"""
multi_risk_matrix.py
=====================
Builds liquidity and market "impact matrices" analogous to build_pd_matrix()
(scenario_selector.py) — {scenario_name: pd.Series(index=year)} — so the NGFS
adverse/severe scenario-pair selection can be driven by credit + liquidity +
market signals jointly, instead of credit PD alone.

Both builders reuse ALREADY-CALIBRATED satellites (liquidity's
LiquidityStressEngine, market's IRSatellite BetaSatellite), fit once on
historical data before this module runs — nothing here re-fits any model; it
only replays the fitted models across every NGFS candidate scenario (not just
the 2-3 scenarios a selection has already picked).

select_scenario_pair_multi_risk() then combines the three matrices: a
candidate (adverse, severe) pair must pass the existing hard constraints
(C1-C5, unchanged, reused as-is from scenario_selector.py) for a MAJORITY of
the risks that have data for it, and is ranked by the equal-weighted average
of each available risk's existing composite soft-score formula. Any risk
missing data for this run (no liquidity/market file uploaded, or projection
failed) is simply left out — the algorithm degrades gracefully to fewer
risks, down to credit-only (byte-identical to the original
find_optimal_scenario_pair behaviour) if liquidity and market are both
unavailable.
"""
from __future__ import annotations

import logging
from itertools import permutations
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("climate.multi_risk_matrix")


# ─────────────────────────────────────────────────────────────────────────────
# Generic NGFS tidy → per-scenario macro DataFrame pivot (ALL candidates)
# ─────────────────────────────────────────────────────────────────────────────

def build_ngfs_macro_by_scenario(
    ngfs_country: pd.DataFrame,
    candidate_scenarios: List[str],
    var_mapper: Callable[[str], Optional[str]],
) -> Dict[str, pd.DataFrame]:
    """
    Pivot the tidy NGFS frame (columns: year, scenario, variable_base, value —
    the same shape run_three_scenarios() already loads as `ngfs_country`) into
    {scenario_name: DataFrame(index=year, columns=mapped_var)} for EVERY
    candidate scenario at once, instead of just an already-chosen pair.

    Reuses whatever tidy data run_three_scenarios() already loaded — no
    re-read of the NGFS file.

    Parameters
    ----------
    ngfs_country        : tidy NGFS frame, already country/region-filtered.
    candidate_scenarios : scenario names to pivot (baseline + all non-baseline
                          candidates under consideration).
    var_mapper          : callable(variable_base) -> mapped column name or
                          None if not usable — pass ngfs_credit_engine's
                          _map_ngfs_variable for a market/credit-style macro
                          universe, or ngfs_liquidity_engine's for the
                          liquidity MACRO_VARS universe.

    Returns
    -------
    {scenario_name: DataFrame} — scenarios with no mappable variable or no
    data are simply absent from the result (graceful — never raises).
    """
    if ngfs_country is None or ngfs_country.empty or not candidate_scenarios:
        return {}

    tidy = ngfs_country[ngfs_country["scenario"].isin(candidate_scenarios)].copy()
    if tidy.empty:
        return {}

    tidy["mapped_col"] = tidy["variable_base"].apply(var_mapper)
    tidy = tidy[tidy["mapped_col"].notna()].copy()
    if tidy.empty:
        return {}

    result: Dict[str, pd.DataFrame] = {}
    for scen_name, df_s in tidy.groupby("scenario"):
        df_agg = (
            df_s.groupby(["year", "mapped_col"])["value"].mean().reset_index()
        )
        pivoted = df_agg.pivot(index="year", columns="mapped_col", values="value")
        pivoted.index = pivoted.index.astype(int)
        pivoted = pivoted.sort_index()
        pivoted.index.name = "year"
        pivoted.columns.name = None
        if not pivoted.empty:
            result[str(scen_name)] = pivoted

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Liquidity matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_liquidity_matrix(
    liquidity_engine,
    ngfs_macro_by_scenario: Dict[str, pd.DataFrame],
    metric: str = "nsfr",
) -> Dict[str, pd.Series]:
    """
    Parameters
    ----------
    liquidity_engine       : LiquidityStressEngine, already calibrated
                             (.calibrate() already called on historical data).
    ngfs_macro_by_scenario : {scenario_name: DataFrame(index=year, columns=MACRO_VARS)}
                             — ALL NGFS candidate scenarios (not just an
                             already-chosen adverse/severe pair).
    metric                 : "lcr" or "nsfr".

    Returns
    -------
    {scenario_name: pd.Series(index=year)} — only scenarios with >= 2 valid
    points; {} if liquidity_engine is None, uncalibrated, or compute_stress
    fails (graceful degradation — never raises).
    """
    if not ngfs_macro_by_scenario or liquidity_engine is None:
        return {}
    try:
        # enforce_sign=False: NGFS climate scenarios can show short-term macro
        # better than baseline for some candidates (same rationale already
        # used in ngfs_liquidity_engine.py's own compute_stress call).
        results = liquidity_engine.compute_stress(
            ngfs_macro_by_scenario, enforce_sign=False,
        )
    except Exception as exc:
        log.warning(
            "build_liquidity_matrix: compute_stress échoué (%s) — matrice vide.",
            exc,
        )
        return {}

    matrix: Dict[str, pd.Series] = {}
    for scen_name, risk_output in results.items():
        try:
            ts = risk_output.time_series.set_index("year")[metric].dropna()
        except Exception:
            continue
        if len(ts) >= 2:
            matrix[scen_name] = ts

    log.info(
        "build_liquidity_matrix: %d/%d scénarios avec trajectoire '%s' valide.",
        len(matrix), len(ngfs_macro_by_scenario), metric,
    )
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# Market matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_market_matrix(
    beta1_satellite,
    last_beta1: float,
    ngfs_macro_by_scenario: Dict[str, pd.DataFrame],
) -> Dict[str, pd.Series]:
    """
    Walks the already-calibrated beta1 (Nelson-Siegel level factor) satellite
    forward year-by-year over each NGFS candidate scenario's macro
    trajectory, using BetaSatellite.predict_next() — the exact same AR(1)+
    macro formula used in production (ir_satellite.py), just walked annually
    here (NGFS trajectories are annual) instead of monthly. No model is
    re-fit; only replayed.

    beta1 (the level factor) is used as the market severity proxy: it drives
    the dominant, roughly-parallel shift of the whole yield curve, so its
    magnitude is a reasonable single scalar for RANKING scenario severity.
    The full β₁/β₂/β₃ combination is still used later, downstream, for the
    actual repricing/valuation in MarketEngineAdapter — this matrix only
    needs a monotonic severity signal for scenario selection, not a final
    valuation.

    Parameters
    ----------
    beta1_satellite        : BetaSatellite for k=1, already fit on historical
                              yield data (ir_satellite.py). None → returns {}.
    last_beta1              : last observed historical beta1 value (AR(1)
                              recursion starting point).
    ngfs_macro_by_scenario : {scenario_name: DataFrame(index=year, columns=MACRO_VARS)}

    Returns
    -------
    {scenario_name: pd.Series(index=year)} — one projected beta1 level per
    year; {} if beta1_satellite is None or projection fails for every
    scenario (graceful degradation — never raises).
    """
    if not ngfs_macro_by_scenario or beta1_satellite is None:
        return {}

    matrix: Dict[str, pd.Series] = {}
    for scen_name, macro_df in ngfs_macro_by_scenario.items():
        try:
            years = sorted(int(y) for y in macro_df.index)
            if len(years) < 2:
                continue
            beta_prev = float(last_beta1)
            values: Dict[int, float] = {}
            for yr in years:
                row = macro_df.loc[yr].to_dict()
                beta_prev = beta1_satellite.predict_next(beta_prev, row)
                values[yr] = beta_prev
            matrix[scen_name] = pd.Series(values)
        except Exception as exc:
            log.debug(
                "build_market_matrix: projection échouée pour '%s' (%s) — ignoré.",
                scen_name, exc,
            )
            continue

    log.info(
        "build_market_matrix: %d/%d scénarios projetés (beta1).",
        len(matrix), len(ngfs_macro_by_scenario),
    )
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# Multi-risk pair selection
# ─────────────────────────────────────────────────────────────────────────────

def select_scenario_pair_multi_risk(
    pd_matrix: Dict[str, pd.Series],
    baseline_name: str,
    liquidity_matrix: Optional[Dict[str, pd.Series]] = None,
    market_matrix: Optional[Dict[str, pd.Series]] = None,
    cfg=None,
    enforce_cross_family: bool = False,
):
    """
    Select the (adverse, severe) NGFS scenario pair using credit (PD) +
    liquidity (NSFR) + market (beta1) signals jointly, instead of credit PD
    alone.

    Hard constraints (C1-C5, reused as-is from scenario_selector.py, never
    modified): a candidate pair must pass for a MAJORITY of the risks that
    have data for it (not unanimity — liquidity/market projections can be
    noisier or have fewer statistically significant macro drivers than
    credit's, so requiring all 3 to agree would too often collapse to the
    last-resort fallback).

    Soft score: for each risk, the exact same composite-score formula as
    find_optimal_scenario_pair() (_compute_soft_scores + min-max
    normalisation via _normalise_scores + cfg.weight_vector) — then the
    pair's final score is the equal-weighted average across whichever risks
    have data for that specific pair.

    Degrades gracefully: if liquidity_matrix and market_matrix are both
    empty/None (no data available this run), this is byte-identical to
    calling find_optimal_scenario_pair(pd_matrix, ...) directly.

    Returns
    -------
    ScenarioPairResult (same type as find_optimal_scenario_pair).
    """
    from .scenario_selector import (
        ScenarioSelectorConfig, ScenarioPairResult,
        _check_constraints, _compute_soft_scores, _normalise_scores,
        find_optimal_scenario_pair,
    )

    if cfg is None:
        cfg = ScenarioSelectorConfig()

    risk_matrices: Dict[str, Dict[str, pd.Series]] = {"credit": pd_matrix}
    if liquidity_matrix:
        risk_matrices["liquidity"] = liquidity_matrix
    if market_matrix:
        risk_matrices["market"] = market_matrix

    if len(risk_matrices) == 1:
        log.info(
            "Sélection multi-risque: liquidité/marché indisponibles pour ce "
            "run — repli sur sélection crédit seul."
        )
        return find_optimal_scenario_pair(
            pd_matrix, baseline_name, cfg, enforce_cross_family,
        )

    stressed_names = [k for k in pd_matrix if k != baseline_name]
    if len(stressed_names) < 2:
        return ScenarioPairResult(valid=False, baseline=baseline_name)

    weights = cfg.weight_vector
    score_keys = list(weights.keys())

    # ── Per-risk: hard-constraint pass/fail + raw soft scores per pair ────
    per_risk_pass: Dict[str, Dict[tuple, bool]] = {r: {} for r in risk_matrices}
    per_risk_raw:  Dict[str, Dict[tuple, dict]] = {r: {} for r in risk_matrices}

    for risk_name, matrix in risk_matrices.items():
        if baseline_name not in matrix:
            log.info(
                "Sélection multi-risque: baseline absente de la matrice "
                "'%s' — risque ignoré pour ce run.", risk_name,
            )
            continue
        base_s = matrix[baseline_name].dropna().sort_index()

        for adv_name, sev_name in permutations(stressed_names, 2):
            if adv_name not in matrix or sev_name not in matrix:
                continue
            if enforce_cross_family:
                fam_adv = adv_name.split("_")[0].upper()
                fam_sev = sev_name.split("_")[0].upper()
                if fam_adv == fam_sev:
                    continue

            adv_s = matrix[adv_name].dropna().sort_index()
            sev_s = matrix[sev_name].dropna().sort_index()

            passed, _detail, _reason = _check_constraints(base_s, adv_s, sev_s, cfg)
            per_risk_pass[risk_name][(adv_name, sev_name)] = passed
            if passed:
                per_risk_raw[risk_name][(adv_name, sev_name)] = _compute_soft_scores(
                    base_s, adv_s, sev_s,
                    severity_ranks=cfg.scenario_severity_ranks,
                    adv_name=adv_name, sev_name=sev_name,
                )

    # ── Normalise each risk's raw scores across ITS OWN constraint-passing
    #    pairs (mirrors find_optimal_scenario_pair exactly, per risk) ──────
    per_risk_score: Dict[str, Dict[tuple, float]] = {}
    for risk_name, raw_by_pair in per_risk_raw.items():
        if not raw_by_pair:
            per_risk_score[risk_name] = {}
            continue
        pairs = list(raw_by_pair.keys())
        records = [dict(raw_by_pair[p]) for p in pairs]
        records = _normalise_scores(records, score_keys)
        per_risk_score[risk_name] = {
            pairs[i]: sum(weights[k] * records[i][f"{k}_norm"] for k in score_keys)
            for i in range(len(pairs))
        }

    # ── Combine: majority hard-constraint pass among AVAILABLE risks for
    #    that pair + equal-weighted average of available composite scores ──
    all_pairs = set()
    for d in per_risk_pass.values():
        all_pairs.update(d.keys())

    candidates = []
    for pair in all_pairs:
        available = [r for r in risk_matrices if pair in per_risk_pass[r]]
        if not available:
            continue
        n_pass = sum(1 for r in available if per_risk_pass[r][pair])
        # Majority = strictly more than half of the risks that had data for
        # this pair (e.g. 2/3 when all three are available).
        if n_pass * 2 <= len(available):
            continue
        scores = [
            per_risk_score[r][pair] for r in available
            if pair in per_risk_score.get(r, {})
        ]
        if not scores:
            continue
        candidates.append({
            "adverse": pair[0], "severe": pair[1],
            "score": sum(scores) / len(scores),
            "n_risks_pass": n_pass, "n_risks_available": len(available),
        })

    if not candidates:
        log.warning(
            "Sélection multi-risque: aucune paire ne passe la majorité des "
            "contraintes dures sur les risques disponibles (%s) — repli sur "
            "sélection crédit seul.", list(risk_matrices.keys()),
        )
        return find_optimal_scenario_pair(
            pd_matrix, baseline_name, cfg, enforce_cross_family,
        )

    ranking_df = (
        pd.DataFrame(candidates)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    best = ranking_df.iloc[0]

    log.info(
        "Sélection multi-risque ✓ | adverse='%s' severe='%s' score=%.4f "
        "(risques utilisés: %s ; %d/%d passent la majorité pour la paire "
        "retenue)",
        best["adverse"], best["severe"], float(best["score"]),
        list(risk_matrices.keys()),
        int(best["n_risks_pass"]), int(best["n_risks_available"]),
    )

    return ScenarioPairResult(
        valid=True,
        baseline=baseline_name,
        adverse=str(best["adverse"]),
        severe=str(best["severe"]),
        score=float(best["score"]),
        constraints={},
        soft_scores={},
        ranking_df=ranking_df,
        rejected_df=pd.DataFrame(),
        n_candidates=len(all_pairs),
    )
