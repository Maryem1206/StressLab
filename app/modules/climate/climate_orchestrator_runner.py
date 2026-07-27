"""
climate_orchestrator_runner.py
===============================
Entry-point function run_climate_orchestration() called by the climate module
wrapper after the NGFS computation completes.

Responsibility chain:
  1. Optionally build MultiTargetSatelliteFactory (Layer 2) from historical
     macro data + target series present in ngfs_projections.
  2. Instantiate ClimateMacroAdapter (passe la factory si disponible).
  3. Si factory disponible, appeler prepare_satellite_projections() pour
     pré-calculer les deltas cibles (PD, ΔLCR, ΔNSFR, β₁, β₂, β₃).
  4. Instantiate the three engine adapters (Credit, Liquidity, Market).
  5. Instantiate ClimateOrchestrator.
  6. Derive horizon_years and scenario names from ngfs_projections.
  7. Call run_all_scenarios().
  8. Serialise results for render_template().

Dégradation gracieuse :
  Si ngfs_projections ne contient pas "macro_df" et/ou "target_series",
  la factory n'est pas construite et l'adapter fonctionne en Layer 1 seule
  (comportement d'avant l'intégration de la factory).

Clés optionnelles dans ngfs_projections qui activent la Layer 2 :
  "macro_df"      : pd.DataFrame — historique macro (index=année int,
                    colonnes ⊇ CLIMATE_MACRO_SOCLE)
  "target_series" : dict {nom: pd.Series} — séries historiques des variables
                    cibles (pd, lcr, nsfr, beta1, beta2, beta3 ; index=année)
  "ngfs_dfs"      : dict {scénario: pd.DataFrame} — trajectoires NGFS par
                    scénario (index=année, colonnes ⊇ CLIMATE_MACRO_SOCLE).
                    Si absent, construit automatiquement depuis macro_trajectories.
  "satellite_params" : dict — kwargs optionnels passés à MultiTargetSatelliteFactory
                    (min_abs_corr, pvalue_threshold, vif_threshold, harrell_factor)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

LOG = logging.getLogger("climate.orchestrator_runner")


def run_climate_orchestration(
    config: Dict[str, Any],
    ngfs_projections: Dict[str, Any],
    institution_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Instantiate and run the Climate Orchestrator.  Returns a dict ready for
    render_template() under the key "climate_stress_results".

    Parameters
    ----------
    config : dict
        Must include keys:
          upload_paths : Dict[str, str]   — same as ClimateModuleWrapper.upload_paths
          params       : Dict[str, Any]   — same as ClimateModuleWrapper.params
          scenario     : Dict[str, Any]   — base scenario dict

    ngfs_projections : dict
        Output from the climate wrapper _parse() method.  Standard keys:
          ngfs_credit, ngfs_liquidity, baseline_scenario, adverse_scenario,
          severe_scenario, macro_trajectories.
        Optional Layer 2 keys: macro_df, target_series, ngfs_dfs,
          satellite_params (see module docstring).

    institution_data : dict
        Arbitrary extra context (reserved for future use).

    Returns
    -------
    dict with keys:
      "results"           : {scenario: {year: dashboard_dict}}
      "run_summary"       : {engine: {successes, failures, skipped}}
      "satellite_report"  : FactoryReport dict | None
      "error"             : str | None
    """
    from .climate_macro_adapter import ClimateMacroAdapter
    from .engine_adapters import (
        CreditEngineAdapter,
        LiquidityEngineAdapter,
        MarketEngineAdapter,
    )
    from .climate_orchestrator import ClimateOrchestrator

    upload_paths  = config.get("upload_paths", {})
    params        = config.get("params", {})
    base_scenario = config.get("scenario", {})
    baseline_name = ngfs_projections.get("baseline_scenario", "Baseline")

    # ── 1. Satellite factory (Layer 2 — optionnelle) ──────────────────────────
    factory        = _build_satellite_factory(ngfs_projections, config)
    satellite_report = None

    # ── 2. ClimateMacroAdapter ────────────────────────────────────────────────
    adapter = ClimateMacroAdapter(
        satellite_factory   = factory,
        plausibility_bounds = config.get("plausibility_bounds"),
    )

    # ── 3. Projection satellite → deltas cibles ───────────────────────────────
    if factory is not None:
        ngfs_dfs = _extract_ngfs_dfs(ngfs_projections, baseline_name)
        if ngfs_dfs:
            # prepare_satellite_projections déclenche aussi update_monotony_test()
            # (B3.3) pour mettre à jour les rapports post-estimation après projection
            adapter.prepare_satellite_projections(
                ngfs_dfs, baseline_name=baseline_name
            )
            satellite_report = _serialise_factory_report(factory.report)
            LOG.info(
                "Layer 2 active: %d/%d satellites calibrés pour %d scénarios NGFS",
                factory.report.n_fitted,
                factory.report.n_requested,
                len(ngfs_dfs) - 1,  # -1 baseline
            )
        else:
            LOG.warning(
                "run_climate_orchestration: ngfs_dfs vide — "
                "Layer 2 désactivée malgré la factory."
            )
    else:
        LOG.info(
            "run_climate_orchestration: factory absente — "
            "Layer 1 (macro proxy) uniquement."
        )

    # ── 4. Engine adapters ────────────────────────────────────────────────────
    # Injection directe NGFS : credit_engine/liquidity_engine lisent les
    # trajectoires PD/LGD/RWA (ngfs_credit_engine.py) et LCR/NSFR
    # (ngfs_liquidity_engine.py) déjà calculées une fois par run — plus de
    # rampe/décroissance synthétique. adverse_scenario/severe_scenario
    # (noms NGFS réels) servent à résoudre l'alias baseline/adverse/severe
    # depuis MacroDeltaVector.scenario_name.
    adverse_scenario = ngfs_projections.get("adverse_scenario", "")
    severe_scenario  = ngfs_projections.get("severe_scenario", "")
    credit_engine    = CreditEngineAdapter(
        upload_paths, params, base_scenario,
        ngfs_credit=ngfs_projections.get("ngfs_credit"),
        adverse_name=adverse_scenario, severe_name=severe_scenario,
    )
    liquidity_engine = LiquidityEngineAdapter(
        upload_paths, params, base_scenario,
        ngfs_liquidity=ngfs_projections.get("ngfs_liquidity"),
        adverse_name=adverse_scenario, severe_name=severe_scenario,
    )
    market_engine    = MarketEngineAdapter(upload_paths, params, base_scenario)

    # ── 5. Orchestrateur ──────────────────────────────────────────────────────
    orchestrator = ClimateOrchestrator(
        credit_engine=credit_engine,
        liquidity_engine=liquidity_engine,
        market_engine=market_engine,
        adapter=adapter,
        rag_amber_threshold=float(params.get("rag_amber_threshold", 0.10)),
        rag_red_threshold=float(params.get("rag_red_threshold", 0.25)),
    )

    # ── 6. Scénarios et horizons ──────────────────────────────────────────────
    horizon_years  = _extract_horizon_years(ngfs_projections)
    scenario_names = _extract_scenario_names(ngfs_projections)

    if not horizon_years or not scenario_names:
        LOG.warning(
            "run_climate_orchestration: horizon_years=%r scénarios=%r — "
            "orchestration ignorée.",
            horizon_years, scenario_names,
        )
        return {
            "results":            {},
            "run_summary":        {},
            "satellite_report":   satellite_report,
            "leaderboard":        [],
            "leaderboard_summary": [],
            "error":              "Aucun scénario ou horizon NGFS disponible.",
        }

    LOG.info(
        "run_climate_orchestration: %d scénario(s) × %d année(s) → "
        "%d runs moteur",
        len(scenario_names), len(horizon_years),
        len(scenario_names) * len(horizon_years) * 3,
    )

    # ── 7. Exécution ──────────────────────────────────────────────────────────
    try:
        raw_results = orchestrator.run_all_scenarios(
            ngfs_projections = ngfs_projections,
            horizon_years    = horizon_years,
            scenarios        = scenario_names,
        )
    except Exception as exc:
        LOG.error("run_climate_orchestration: run_all_scenarios échoué: %s", exc)
        return {
            "results":            {},
            "run_summary":        orchestrator.run_summary(),
            "satellite_report":   satellite_report,
            "leaderboard":        _build_leaderboard(factory),
            "leaderboard_summary": _build_leaderboard_summary(factory),
            "error":              str(exc),
        }

    # ── 8. Sérialisation Jinja2 / JSON ────────────────────────────────────────
    serialised: Dict[str, Dict[str, Any]] = {}
    for scen_name, year_dict in raw_results.items():
        serialised[scen_name] = {}
        for year, stress_result in year_dict.items():
            try:
                serialised[scen_name][str(year)] = stress_result.to_dashboard_dict()
            except Exception as exc:
                LOG.warning(
                    "run_climate_orchestration: sérialisation échouée "
                    "(scénario=%r année=%d): %s",
                    scen_name, year, exc,
                )
                serialised[scen_name][str(year)] = {
                    "scenario":     scen_name,
                    "horizon_year": year,
                    "error":        str(exc),
                    "rows":         [],
                }

    LOG.info(
        "run_climate_orchestration: terminé — %d scénarios sérialisés",
        len(serialised),
    )

    # ── 9. Leaderboard post-estimation (toutes cibles × tous tests) ──────────
    leaderboard      = _build_leaderboard(factory)
    leaderboard_summ = _build_leaderboard_summary(factory)

    LOG.info(
        "run_climate_orchestration: terminé — %d scénarios sérialisés | "
        "leaderboard: %d lignes (%d satellites)",
        len(serialised),
        len(leaderboard),
        len(leaderboard_summ),
    )

    # ── 10. Courbe des taux de référence (marché) ─────────────────────────────
    # Exposé une seule fois par run (indépendant du scénario/année) pour la
    # reconstruction de la courbe des taux stressée côté dashboard (Row 3
    # Bloc C). Le fitter/baseline_curve ne sont normalement disponibles
    # qu'en mémoire (MarketEngineAdapter._prefitted) ; on les sérialise ici
    # de façon additive, sans toucher au fonctionnement de run_stressed().
    market_baseline_curve: Dict[str, float] = {}
    market_lambda = None
    _pf = getattr(market_engine, "_prefitted", None)
    if _pf is not None:
        try:
            market_baseline_curve = {
                str(k): float(v) for k, v in _pf["baseline_curve"].items()
            }
            market_lambda = float(_pf["fitter"].lambda_)
        except Exception as exc:
            LOG.warning(
                "run_climate_orchestration: exposition courbe de base "
                "marché échouée: %s", exc,
            )

    return {
        "results":            serialised,
        "run_summary":        orchestrator.run_summary(),
        "satellite_report":   satellite_report,
        "leaderboard":        leaderboard,
        "leaderboard_summary": leaderboard_summ,
        "market_baseline_curve": market_baseline_curve,
        "market_lambda":      market_lambda,
        "error":              None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Factory builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_satellite_factory(
    ngfs_projections: Dict[str, Any],
    config: Dict[str, Any],
) -> Optional[Any]:
    """
    Construit et calibre la MultiTargetSatelliteFactory si les données
    nécessaires sont présentes dans ngfs_projections.

    Clés requises :
      ngfs_projections["macro_df"]      : pd.DataFrame
      ngfs_projections["target_series"] : dict {nom: pd.Series}

    Retourne None (dégradation silencieuse) si l'une des deux est absente
    ou si la calibration échoue.
    """
    try:
        from .multi_target_satellite import MultiTargetSatelliteFactory
    except ImportError as exc:
        LOG.warning("_build_satellite_factory: import échoué (%s) — Layer 2 désactivée.", exc)
        return None

    macro_df = ngfs_projections.get("macro_df")
    targets  = ngfs_projections.get("target_series", {})

    if macro_df is None:
        LOG.info(
            "_build_satellite_factory: 'macro_df' absent de ngfs_projections "
            "— Layer 2 désactivée."
        )
        return None

    if not targets:
        LOG.info(
            "_build_satellite_factory: 'target_series' vide — Layer 2 désactivée."
        )
        return None

    if not isinstance(macro_df, pd.DataFrame) or macro_df.empty:
        LOG.warning(
            "_build_satellite_factory: macro_df invalide ou vide — Layer 2 désactivée."
        )
        return None

    # Paramètres optionnels de la factory
    sat_params = config.get("satellite_params", {})

    try:
        factory = MultiTargetSatelliteFactory(macro_df, **sat_params)
        fitted  = factory.fit_all(targets)
        LOG.info(
            "_build_satellite_factory: %d/%d satellites calibrés — cibles: %s",
            factory.report.n_fitted,
            factory.report.n_requested,
            list(fitted.keys()),
        )
        if factory.report.n_fitted == 0:
            LOG.warning(
                "_build_satellite_factory: aucun satellite calibré "
                "(données insuffisantes ?) — Layer 2 désactivée."
            )
            return None
        return factory
    except Exception as exc:
        LOG.warning(
            "_build_satellite_factory: calibration échouée (%s) — "
            "Layer 2 désactivée.",
            exc,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# NGFS DataFrames builder
# ─────────────────────────────────────────────────────────────────────────────

def _extract_ngfs_dfs(
    ngfs_projections: Dict[str, Any],
    baseline_name: str = "Baseline",
) -> Dict[str, pd.DataFrame]:
    """
    Retourne {scenario_name: DataFrame(index=année int, cols=macro vars)}
    pour la projection satellite.

    Priorité :
      1. ngfs_projections["ngfs_dfs"] — fourni directement (format natif)
      2. ngfs_projections["macro_trajectories"] — {scénario: {année: {col: val}}}
         converti en DataFrames

    Retourne {} si aucune source n'est disponible.
    """
    # Source 1 : clé explicite
    ngfs_dfs_raw = ngfs_projections.get("ngfs_dfs")
    if ngfs_dfs_raw and isinstance(ngfs_dfs_raw, dict):
        # Valider que les valeurs sont bien des DataFrames
        valid = {
            k: v for k, v in ngfs_dfs_raw.items()
            if isinstance(v, pd.DataFrame) and not v.empty
        }
        if valid:
            LOG.debug(
                "_extract_ngfs_dfs: %d DataFrames depuis 'ngfs_dfs'", len(valid)
            )
            return valid

    # Source 2 : macro_trajectories → conversion en DataFrames
    macro_traj = ngfs_projections.get("macro_trajectories")
    if not macro_traj or not isinstance(macro_traj, dict):
        LOG.debug(
            "_extract_ngfs_dfs: ni 'ngfs_dfs' ni 'macro_trajectories' — "
            "ngfs_dfs vide."
        )
        return {}

    result: Dict[str, pd.DataFrame] = {}
    for scen_name, year_dict in macro_traj.items():
        if not isinstance(year_dict, dict):
            continue
        try:
            # year_dict : {year_int_or_str: {col: value}}
            records = {}
            for yr_key, row in year_dict.items():
                try:
                    yr = int(yr_key)
                except (ValueError, TypeError):
                    continue
                if isinstance(row, dict):
                    records[yr] = row

            if not records:
                continue

            df = pd.DataFrame.from_dict(records, orient="index")
            df.index = df.index.astype(int)
            df.sort_index(inplace=True)
            result[scen_name] = df.apply(pd.to_numeric, errors="coerce")

        except Exception as exc:
            LOG.warning(
                "_extract_ngfs_dfs: conversion échouée pour scénario '%s': %s",
                scen_name, exc,
            )

    LOG.debug(
        "_extract_ngfs_dfs: %d DataFrames construits depuis macro_trajectories",
        len(result),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _serialise_factory_report(report: Any) -> Dict[str, Any]:
    """Convertit un FactoryReport en dict JSON-sérialisable."""
    try:
        return {
            "n_requested": report.n_requested,
            "n_fitted":    report.n_fitted,
            "details":     {
                target: {
                    k: (list(v) if isinstance(v, (list,)) else v)
                    for k, v in detail.items()
                }
                for target, detail in report.details.items()
            },
        }
    except Exception:
        return {}


def _extract_horizon_years(ngfs_projections: Dict[str, Any]) -> List[int]:
    """Extrait la liste des années horizon depuis les projections NGFS."""
    if "horizon_years" in ngfs_projections:
        return [int(y) for y in ngfs_projections["horizon_years"]]

    ngfs_credit = ngfs_projections.get("ngfs_credit", {})
    for alias in ("adverse", "severe", "baseline"):
        years = (ngfs_credit.get("scenarios", {})
                            .get(alias, {})
                            .get("years", []))
        if years:
            return sorted({int(y) for y in years})

    ngfs_liq = ngfs_projections.get("ngfs_liquidity", {})
    for alias in ("adverse", "severe", "baseline"):
        years = (ngfs_liq.get("scenarios", {})
                         .get(alias, {})
                         .get("years", []))
        if years:
            return sorted({int(y) for y in years})

    return []


def _extract_scenario_names(ngfs_projections: Dict[str, Any]) -> List[str]:
    """Retourne les noms de scénarios NGFS à stresser (hors baseline)."""
    if "scenarios" in ngfs_projections:
        return list(ngfs_projections["scenarios"])

    names: List[str] = []
    adverse = ngfs_projections.get("adverse_scenario", "")
    severe  = ngfs_projections.get("severe_scenario",  "")

    if adverse and adverse not in names:
        names.append(adverse)
    if severe and severe not in names and severe != adverse:
        names.append(severe)

    return names


def _build_leaderboard(factory: Optional[Any]) -> List[Dict[str, Any]]:
    """
    Construit le leaderboard complet (toutes cibles × tous tests) depuis la
    factory calibrée.

    Retourne une liste de dicts JSON-sérialisables (une entrée par ligne du
    DataFrame leaderboard). Retourne [] si la factory est None ou si aucun
    rapport n'est disponible.
    """
    if factory is None:
        return []
    try:
        df = factory.leaderboard()
        if df.empty:
            return []
        return df.to_dict(orient="records")
    except Exception as exc:
        LOG.warning("_build_leaderboard: échec (%s) — leaderboard vide", exc)
        return []


def _build_leaderboard_summary(factory: Optional[Any]) -> List[Dict[str, Any]]:
    """
    Construit le résumé du leaderboard (une ligne par satellite) depuis la
    factory calibrée.

    Retourne une liste de dicts JSON-sérialisables.
    Retourne [] si la factory est None ou si aucun rapport n'est disponible.
    Colonnes : Satellite | Famille | n_obs | R² | n_tests | PASS | WARN | FAIL | Verdict.
    Trié : REJECTED en premier, R² décroissant.
    """
    if factory is None:
        return []
    try:
        df = factory.leaderboard_summary()
        if df.empty:
            return []
        return df.to_dict(orient="records")
    except Exception as exc:
        LOG.warning("_build_leaderboard_summary: échec (%s) — résumé vide", exc)
        return []
