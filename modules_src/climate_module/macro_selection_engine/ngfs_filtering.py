"""
ngfs_filtering.py
=================
STAGE 1 — NGFS-only variable filtering. NO target variable is used here.

The user's clarification is: 'combined' is a *risk channel* extracted from
the NGFS variable name (the trailing parenthesis), found within each of the
7 NGFS scenarios. So Stage 1 always:

  1. selects rows where channel == cfg.risk_channel (default: 'combined' strict)
  2. pivots the trajectories to a year x variable matrix per scenario
  3. applies three filters:
        a) variability (top X% by coefficient of variation)
        b) scenario sensitivity = |stressed - baseline|
        c) economic relevance (whitelist via ECONOMIC_PRIORS)

Output: a list of NGFS variable names (variable_base) that survive.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import (ECONOMIC_PRIORS, EngineConfig, expected_sign, get_logger,
                    normalize_label)

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Sélection du channel le plus intense (Option B)
# ──────────────────────────────────────────────────────────────────────────────

def _pick_most_intense_channel(
    ngfs_country: pd.DataFrame,
    scenario: str,
    baseline_scenario: str,
    available_channels: List[str],
) -> str:
    """
    Parmi les channels disponibles pour un scénario, retourne celui dont
    les variables s'écartent le plus du Baseline.

    Métrique : déviation absolue moyenne sur toutes les variables et années.

        amplitude(channel) = mean_{var, year} |stressed(var,year) - baseline(var,year)|

    Le channel avec la plus haute amplitude est retenu.

    Parameters
    ----------
    ngfs_country       : DataFrame NGFS filtré sur le pays
    scenario           : scénario stressé à évaluer
    baseline_scenario  : scénario de référence (Baseline)
    available_channels : liste des channels disponibles pour ce scénario

    Returns
    -------
    str : nom du channel le plus intense
    """
    scores: Dict[str, float] = {}

    for ch in available_channels:
        # Pivot stressed
        stressed_rows = ngfs_country[
            (ngfs_country["scenario"] == scenario)
            & (ngfs_country["channel"] == ch)
        ]
        # Pivot baseline : essayer même channel, sinon NaN channel (baseline NGFS)
        baseline_rows = ngfs_country[
            (ngfs_country["scenario"] == baseline_scenario)
            & (ngfs_country["channel"] == ch)
        ]
        if baseline_rows.empty:
            baseline_rows = ngfs_country[
                (ngfs_country["scenario"] == baseline_scenario)
                & (ngfs_country["channel"].isna())
            ]

        if stressed_rows.empty or baseline_rows.empty:
            scores[ch] = 0.0
            continue

        # Pivoter year × variable_base
        stressed_pivot = (
            stressed_rows.groupby(["year", "variable_base"])["value"]
            .mean().unstack("variable_base")
        )
        baseline_pivot = (
            baseline_rows.groupby(["year", "variable_base"])["value"]
            .mean().unstack("variable_base")
        )

        # Intersection variables + années
        common_vars  = stressed_pivot.columns.intersection(baseline_pivot.columns)
        common_years = stressed_pivot.index.intersection(baseline_pivot.index)

        if common_vars.empty or common_years.empty:
            scores[ch] = 0.0
            continue

        s = stressed_pivot.loc[common_years, common_vars]
        b = baseline_pivot.loc[common_years, common_vars]
        amplitude = float((s - b).abs().mean().mean())
        scores[ch] = amplitude

    best_channel = max(scores, key=scores.get)
    log.info(
        "Channel selection for '%s': amplitude scores = %s → picked '%s'.",
        scenario,
        {ch: f"{v:.4f}" for ch, v in scores.items()},
        best_channel,
    )
    return best_channel


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic combined channel
# ──────────────────────────────────────────────────────────────────────────────

def _synthesize_combined(
    ngfs_country: pd.DataFrame,
    scenario: str,
    source_channels: Tuple[str, ...] = ("physical", "transition"),
) -> pd.DataFrame:
    """
    Reconstruit le channel 'combined' = somme des channels source (additif).
    Conforme à la définition NGFS Phase 5 : combined = physical + transition.

    Appliqué uniquement quand (combined) est absent pour ce scénario.
    Les lignes synthétiques reçoivent channel='combined_synthetic' pour
    permettre le traçage et la distinction dans les logs.

    Retourne un DataFrame de lignes supplémentaires à concat dans ngfs_country,
    ou un DataFrame vide si la synthèse est impossible.
    """
    scenario_df = ngfs_country[ngfs_country["scenario"] == scenario]

    # Vérifier si combined existe déjà
    has_combined = (scenario_df["channel"] == "combined").any()
    if has_combined:
        return pd.DataFrame()  # rien à faire

    # Récupérer les variables disponibles pour chaque channel source
    channel_vars: Dict[str, set] = {}
    for ch in source_channels:
        ch_df = scenario_df[scenario_df["channel"] == ch]
        if not ch_df.empty:
            channel_vars[ch] = set(ch_df["variable_base"].unique())

    if len(channel_vars) < 2:
        log.warning(
            "Synthèse combined impossible pour '%s': "
            "channels disponibles=%s (besoin de %s)",
            scenario, list(channel_vars.keys()), list(source_channels),
        )
        return pd.DataFrame()

    # Variables communes aux deux channels
    common_vars = set.intersection(*channel_vars.values())
    if not common_vars:
        log.warning(
            "Synthèse combined pour '%s': aucune variable commune entre %s.",
            scenario, list(source_channels),
        )
        return pd.DataFrame()

    log.info(
        "Synthèse combined pour '%s': %d variables communes (%s + %s).",
        scenario, len(common_vars), source_channels[0], source_channels[1],
    )

    synthetic_rows = []
    for var in sorted(common_vars):
        # Récupérer toutes les lignes de chaque channel pour cette variable
        slices = [
            scenario_df[
                (scenario_df["channel"] == ch)
                & (scenario_df["variable_base"] == var)
            ]
            for ch in source_channels
        ]

        # Aligner sur les mêmes années et additionner les valeurs
        pivots = [s.set_index("year")["value"] for s in slices if not s.empty]
        if len(pivots) < 2:
            continue

        common_years = pivots[0].index.intersection(pivots[1].index)
        if common_years.empty:
            continue

        combined_values = sum(p.loc[common_years] for p in pivots)

        # Construire les lignes synthétiques en copiant les métadonnées du 1er channel
        ref_slice = slices[0].set_index("year").loc[common_years].reset_index()
        for _, ref_row in ref_slice.iterrows():
            row = ref_row.to_dict()
            row["value"]   = float(combined_values.loc[row["year"]])
            row["channel"] = "combined_synthetic"
            synthetic_rows.append(row)

    if not synthetic_rows:
        return pd.DataFrame()

    synth_df = pd.DataFrame(synthetic_rows)
    log.info(
        "Synthèse combined: %d lignes créées pour '%s'.",
        len(synth_df), scenario,
    )
    return synth_df


@dataclass
class Stage1Result:
    selected_variables: List[str] = field(default_factory=list)
    variability_table: Optional[pd.DataFrame] = None
    shock_table: Optional[pd.DataFrame] = None
    economic_relevance_table: Optional[pd.DataFrame] = None
    divergence_table: Optional[pd.DataFrame] = None        # filtre divergence inter-scénarios
    scenario_pivot: Optional[pd.DataFrame] = None          # stressed scenario, channel=combined
    baseline_pivot: Optional[pd.DataFrame] = None          # baseline scenario, channel=combined


def _filter_scenario_divergence(
    ngfs_country: pd.DataFrame,
    candidate_variables: List[str],
    channel: str,
    baseline_scenario: str,
    min_divergence: float,
) -> pd.DataFrame:
    """
    Filtre Option B — Divergence inter-scénarios.

    Pour chaque variable candidate, calcule à quel point ses projections NGFS
    divergent entre scénarios. Une variable qui reste plate sous tous les
    scénarios ne peut pas produire un spread adverse/severe significatif
    dans le modèle satellite, même si elle est corrélée historiquement avec la PD.

    Métrique : Coefficient de Variation inter-scénarios (CV)
        Pour chaque année t : std_over_scenarios(var_t) / |mean_over_scenarios(var_t)|
        Score final : moyenne temporelle de ce CV → résistant aux outliers d'une année

    Paramètres
    ----------
    ngfs_country        : DataFrame NGFS filtré sur le pays (tous scénarios)
    candidate_variables : variables à évaluer (noms NGFS variable_base)
    channel             : channel NGFS à utiliser (ex. 'combined')
    baseline_scenario   : scénario baseline exclu du calcul de divergence
    min_divergence      : seuil minimum de CV inter-scénarios (0 = désactivé)

    Retourne
    --------
    DataFrame avec colonnes [variable_base, cv_inter_scenario, pass_divergence]
    """
    if min_divergence <= 0.0:
        # Filtre désactivé : toutes les variables passent
        if not candidate_variables:
            return pd.DataFrame(columns=["cv_inter_scenario", "pass_divergence"])
        table = pd.DataFrame({
            "cv_inter_scenario": [np.nan] * len(candidate_variables),
            "pass_divergence":   [True]   * len(candidate_variables),
        }, index=candidate_variables)
        return table

    # Récupérer tous les scénarios stressés (hors baseline)
    all_scenarios = ngfs_country["scenario"].dropna().unique().tolist()
    stressed_scenarios = [s for s in all_scenarios if s != baseline_scenario]

    if len(stressed_scenarios) < 2:
        # Pas assez de scénarios pour calculer une divergence significative
        log.warning(
            "Stage 1 / divergence: seulement %d scénarios stressés — filtre ignoré.",
            len(stressed_scenarios),
        )
        table = pd.DataFrame({
            "cv_inter_scenario": [np.nan] * len(candidate_variables),
            "pass_divergence":   [True]   * len(candidate_variables),
        }, index=candidate_variables)
        return table

    records: Dict[str, Dict] = {}

    for var in candidate_variables:
        scenario_series: Dict[str, pd.Series] = {}

        for sc in stressed_scenarios:
            # Chercher avec le channel demandé, puis channel NaN en fallback
            mask = (
                (ngfs_country["scenario"] == sc)
                & (ngfs_country["variable_base"] == var)
            )
            if channel:
                rows_ch = ngfs_country[mask & (ngfs_country["channel"] == channel)]
                if rows_ch.empty:
                    rows_ch = ngfs_country[mask & ngfs_country["channel"].isna()]
            else:
                rows_ch = ngfs_country[mask & ngfs_country["channel"].isna()]

            if rows_ch.empty:
                continue

            s = rows_ch.groupby("year")["value"].mean()
            scenario_series[sc] = s

        if len(scenario_series) < 2:
            # Pas assez de données pour ce scénario → passe par défaut
            records[var] = {"cv_inter_scenario": np.nan, "pass_divergence": True}
            continue

        # Aligner les séries sur les années communes
        combined = pd.DataFrame(scenario_series).dropna()
        if combined.empty or combined.shape[0] < 2:
            records[var] = {"cv_inter_scenario": np.nan, "pass_divergence": True}
            continue

        # CV inter-scénarios par année, moyenné sur le temps
        year_std  = combined.std(axis=1, ddof=1)
        year_mean = combined.mean(axis=1).abs()
        # Éviter division par zéro : si mean ≈ 0, utiliser std brute
        year_cv   = np.where(year_mean > 1e-10, year_std / year_mean, year_std)
        mean_cv   = float(np.nanmean(year_cv))

        records[var] = {
            "cv_inter_scenario": mean_cv,
            "pass_divergence":   mean_cv >= min_divergence,
        }

    if not records:
        return pd.DataFrame(columns=["cv_inter_scenario", "pass_divergence"])

    table = pd.DataFrame(records).T
    table.index.name = "variable_base"
    return table


def _pivot(df_subset: pd.DataFrame) -> pd.DataFrame:
    """Pivot tidy slice (already filtered by region+scenario+channel) to
    year x variable_base. If a variable has multiple unit rows, average them.
    """
    p = (df_subset
         .groupby(["year", "variable_base"], dropna=False)["value"]
         .mean()
         .unstack("variable_base"))
    p.index.name = "year"
    return p.sort_index()


def _filter_variability(pivot: pd.DataFrame, keep_top_pct: float) -> pd.DataFrame:
    """Keep variables in the top X% by coefficient of variation (|std/mean|).
    Variables with zero std are dropped.
    """
    std = pivot.std(axis=0, skipna=True)
    mean = pivot.mean(axis=0, skipna=True).abs().replace(0, np.nan)
    cv = (std / mean).replace([np.inf, -np.inf], np.nan)
    table = pd.DataFrame({"std": std, "abs_mean": mean, "cv": cv}).dropna()
    table = table[table["std"] > 0]
    if table.empty:
        return table
    threshold = table["cv"].quantile(1.0 - keep_top_pct)
    table["pass_variability"] = table["cv"] >= threshold
    return table.sort_values("cv", ascending=False)


def _filter_shock(stressed: pd.DataFrame, baseline: Optional[pd.DataFrame],
                  min_magnitude: float) -> pd.DataFrame:
    """Compute shock = max_t |stressed_t - baseline_t| per variable.
    If baseline is missing, skip gracefully (return empty marker)."""
    if baseline is None or baseline.empty:
        log.warning("Shock filter: baseline scenario not available — skipping.")
        return pd.DataFrame(columns=["max_abs_shock", "pass_shock"])
    common_cols = stressed.columns.intersection(baseline.columns)
    common_idx = stressed.index.intersection(baseline.index)
    if not len(common_cols) or not len(common_idx):
        log.warning("Shock filter: no overlap between stressed and baseline.")
        return pd.DataFrame(columns=["max_abs_shock", "pass_shock"])
    diff = (stressed.loc[common_idx, common_cols]
            - baseline.loc[common_idx, common_cols]).abs()
    mx = diff.max(axis=0, skipna=True)
    table = pd.DataFrame({"max_abs_shock": mx})
    table["pass_shock"] = table["max_abs_shock"] > min_magnitude
    return table.sort_values("max_abs_shock", ascending=False)


def _filter_economic_relevance(variables: List[str],
                               risk_type: str) -> pd.DataFrame:
    """Keep variables that match ANY pattern in ECONOMIC_PRIORS[risk_type]."""
    priors = ECONOMIC_PRIORS.get(risk_type.lower(), {})
    rows = []
    for v in variables:
        sign = expected_sign(v, risk_type)
        # match if variable matches any prior pattern (sign 0 still counts as known)
        norm = normalize_label(v)
        matched_pattern = None
        for pat in sorted(priors.keys(), key=len, reverse=True):
            if normalize_label(pat) in norm:
                matched_pattern = pat
                break
        rows.append({
            "variable_base": v,
            "matched_pattern": matched_pattern,
            "expected_sign": sign,
            "pass_economic": matched_pattern is not None,
        })
    return pd.DataFrame(rows).set_index("variable_base")


def stage1_filter(ngfs_country: pd.DataFrame,
                  cfg: EngineConfig) -> Stage1Result:
    """Run Stage 1 on a country-filtered NGFS frame.

    Parameters
    ----------
    ngfs_country : tidy NGFS frame already filtered by country/region.
    cfg          : EngineConfig.
    """
    log.info("=== STAGE 1: NGFS-only filtering (channel='%s', scenario='%s') ===",
             cfg.risk_channel, cfg.scenario)

    # ── Mode CT : pas de channels (GEM-E3 = niveaux directs) ─────────────────
    is_ct = getattr(cfg, "ngfs_mode", "LT") == "CT"

    if is_ct:
        # CT mode : pas de channel du tout — filtre uniquement sur scenario
        stressed_slice = ngfs_country[ngfs_country["scenario"] == cfg.scenario]
        baseline_slice = ngfs_country[
            ngfs_country["scenario"] == cfg.baseline_scenario
        ]
        log.info("Stage 1 [CT mode]: skip channel filter. "
                 "stressed=%d rows, baseline=%d rows.",
                 len(stressed_slice), len(baseline_slice))
    else:
        # ── LT mode : synthèse combined si absent (ex. Current Policies) ─────
        scenario_channels = (
            ngfs_country.loc[ngfs_country["scenario"] == cfg.scenario, "channel"]
            .dropna().unique().tolist()
        )
        if cfg.risk_channel not in scenario_channels:
            log.warning(
                "Channel '%s' absent pour '%s'. Channels disponibles: %s. "
                "Tentative de synthèse additive.",
                cfg.risk_channel, cfg.scenario, scenario_channels,
            )
            synth_df = _synthesize_combined(ngfs_country, cfg.scenario)
            if not synth_df.empty:
                synth_df["channel"] = cfg.risk_channel
                ngfs_country = pd.concat([ngfs_country, synth_df], ignore_index=True)
                log.info(
                    "Channel '%s' synthétisé: %d lignes ajoutées.",
                    cfg.risk_channel, len(synth_df),
                )
            else:
                # Synthèse impossible → prendre le channel le plus intense
                log.warning(
                    "Synthèse combined impossible pour '%s' — "
                    "sélection du channel le plus intense parmi %s.",
                    cfg.scenario, scenario_channels,
                )
                best_ch = _pick_most_intense_channel(
                    ngfs_country=ngfs_country,
                    scenario=cfg.scenario,
                    baseline_scenario=cfg.baseline_scenario,
                    available_channels=scenario_channels,
                )
                # Réutiliser ce channel comme proxy du combined
                cfg = copy.copy(cfg)
                cfg.risk_channel = best_ch
                log.info(
                    "Channel proxy retenu pour '%s': '%s'.",
                    cfg.scenario, best_ch,
                )

        # Filter on risk channel STRICTLY (option a)
        chan_mask = ngfs_country["channel"] == cfg.risk_channel
        if not chan_mask.any():
            raise ValueError(
                f"No rows for risk_channel='{cfg.risk_channel}'. "
                f"Available channels here: "
                f"{ngfs_country['channel'].dropna().unique().tolist()}"
            )

        stressed_slice = ngfs_country[chan_mask
                                      & (ngfs_country["scenario"] == cfg.scenario)]
        if stressed_slice.empty:
            raise ValueError(
                f"No rows for scenario='{cfg.scenario}' with channel='{cfg.risk_channel}'. "
                f"Available scenarios: "
                f"{ngfs_country['scenario'].unique().tolist()}"
            )

        if cfg.baseline_scenario.lower() == "baseline":
            baseline_slice = ngfs_country[
                (ngfs_country["scenario"] == cfg.baseline_scenario)
                & (ngfs_country["channel"].isna())
            ]
        else:
            baseline_slice = ngfs_country[chan_mask
                                          & (ngfs_country["scenario"] == cfg.baseline_scenario)]

    stressed_pivot = _pivot(stressed_slice)
    baseline_pivot = _pivot(baseline_slice) if not baseline_slice.empty else None

    log.info("Stage 1: %d candidate variables in stressed pivot.",
             stressed_pivot.shape[1])

    # 1) variability filter
    var_table = _filter_variability(stressed_pivot, cfg.s1_keep_top_variance_pct)
    pass_var = set(var_table.index[var_table["pass_variability"]]) \
        if not var_table.empty else set()
    log.info("Stage 1 / variability: %d/%d pass (top %.0f%% by CV).",
             len(pass_var), stressed_pivot.shape[1], 100*cfg.s1_keep_top_variance_pct)

    # 2) shock filter
    shock_table = _filter_shock(stressed_pivot, baseline_pivot,
                                cfg.s1_min_shock_magnitude)
    if not shock_table.empty:
        pass_shock = set(shock_table.index[shock_table["pass_shock"]])
        log.info("Stage 1 / shock: %d pass (vs baseline '%s').",
                 len(pass_shock), cfg.baseline_scenario)
    else:
        # graceful skip — accept all
        pass_shock = set(stressed_pivot.columns)
        log.info("Stage 1 / shock: SKIPPED (no baseline) — all %d variables pass.",
                 len(pass_shock))

    # 3) economic relevance
    econ_table = _filter_economic_relevance(stressed_pivot.columns.tolist(),
                                            cfg.risk_type)
    if cfg.s1_apply_economic_whitelist:
        pass_econ = set(econ_table.index[econ_table["pass_economic"]])
        log.info("Stage 1 / economic relevance: %d pass for risk_type='%s'.",
                 len(pass_econ), cfg.risk_type)
    else:
        pass_econ = set(stressed_pivot.columns)
        log.info("Stage 1 / economic relevance: DISABLED — all %d pass.",
                 len(pass_econ))

    # 4) divergence inter-scénarios (Option B — activé si s1_min_scenario_divergence > 0)
    min_div = getattr(cfg, "s1_min_scenario_divergence", 0.0)
    candidates_for_div = sorted(pass_var & pass_shock & pass_econ)
    div_table = _filter_scenario_divergence(
        ngfs_country=ngfs_country,
        candidate_variables=candidates_for_div,
        channel=cfg.risk_channel,
        baseline_scenario=cfg.baseline_scenario,
        min_divergence=min_div,
    )

    if min_div > 0.0 and not div_table.empty and "pass_divergence" in div_table.columns:
        pass_div = set(div_table.index[div_table["pass_divergence"].astype(bool)])
        n_rejected = len(candidates_for_div) - len(pass_div)
        log.info(
            "Stage 1 / divergence inter-scénarios: %d/%d pass "
            "(seuil CV=%.4f) — %d rejetées (trajectoires plates entre scénarios).",
            len(pass_div), len(candidates_for_div), min_div, n_rejected,
        )
    else:
        pass_div = set(candidates_for_div)
        if min_div <= 0.0:
            log.info("Stage 1 / divergence: DÉSACTIVÉ (s1_min_scenario_divergence=0).")

    # Intersection finale
    selected = sorted(pass_var & pass_shock & pass_econ & pass_div)
    log.info("Stage 1: %d variables survive all filters.", len(selected))

    return Stage1Result(
        selected_variables=selected,
        variability_table=var_table,
        shock_table=shock_table,
        economic_relevance_table=econ_table,
        divergence_table=div_table if not div_table.empty else None,
        scenario_pivot=stressed_pivot,
        baseline_pivot=baseline_pivot,
    )
