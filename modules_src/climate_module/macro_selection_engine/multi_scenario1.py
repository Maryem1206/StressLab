"""
multi_scenario.py
=================
3-scenario stress test : Baseline / Adverse / Severe.

Mise à jour :
  - L'utilisateur choisit le modèle satellite (pas de sélection automatique)
  - Logit sigmoid dans la projection
  - Anchor avant toute transformation
  - best_per_var filtré aux variables du modèle choisi
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .baseline_projector import (
    BaselineProjection,
    build_baseline_macro_df,
    project_baseline,
    summarize_projection,
    _adf,
    _best_ar_order,
    _forecast_ar,
    _forecast_arima,
    _min_diff_order,
)
from .data_loader import load_historical, load_ngfs_for_stress_test
from .geo_selector import select_geography
from .historical_validation import Stage2Result, run_stage2
from .variable_resolver import resolve_variables, print_resolution_log, ResolverResult
from .main import _df_to_json, _model_to_json
from .model_estimation import FittedModel
from .model_ranking import select_model_interactive, select_scenario_pair_interactive
from .ngfs_filtering import Stage1Result, stage1_filter, _pick_most_intense_channel
from .model_stress_validator import (
    validate_models as validate_stress_models,
    get_stress_scores_dict,
)
from .scenario_family_filter import filter_scenario_families
from .transform_selector import (
    VariableBestTransform,
    apply_transform,
    parse_transformed_colname,
    replay_transforms_on_ngfs,
)
from .utils import EngineConfig, get_logger
from .scenario_selector import (
    find_optimal_scenario_pair,
    build_pd_matrix,
    ScenarioSelectorConfig,
)
log = get_logger(__name__)


DEFAULT_SCENARIO_MAP: Dict[str, str] = {
    "baseline": "Net Zero 2050",
    "adverse":  "Below 2°C",
    "severe":   "Fragmented World",
}

CT_SCENARIO_MAP: Dict[str, str] = {
    "baseline": "Baseline",
    "adverse":  "SWUC",          # Sudden Wake-Up Call
    "severe":   "DIRE",          # Diverging Realities
}


def _project_ct_levels(
    model: FittedModel,
    ngfs_country: pd.DataFrame,
    scenario_name: str,
    ngfs_to_hist: Dict[str, str],
    best_per_var: Dict[str, VariableBestTransform],
    hist_df: pd.DataFrame,
    hist_year_col: str = "year",
    target_logit_transformed: bool = False,
) -> List[Dict[str, Any]]:
    """
    Projection CT (SIMPLIFIÉE) : utilise les niveaux NGFS directement.
    
    Fonctionne pour TOUS les scénarios CT (Baseline ET stressés) car
    GEM-E3 fournit des niveaux absolus dans tous les cas.
    
    Flux simplifié (sans transformations) :
      1. Pivot des niveaux NGFS CT pour le scénario donné
      2. Mapping NGFS → colonnes historiques
      3. Rebasing pour aligner échelle NGFS (régionale) sur historique (pays)
      4. Satellite model directement sur niveaux → PD projeté
    """
    # ── 1. Pivot du scénario CT ──────────────────────────────────────────
    rows = ngfs_country[ngfs_country["scenario"] == scenario_name]
    if rows.empty:
        log.warning("CT projection: no data for scenario '%s'.", scenario_name)
        return []

    pivot = (rows.groupby(["year", "variable_base"])["value"]
             .mean().unstack("variable_base").sort_index())
    log.info("CT pivot [%s]: %d years × %d vars.", scenario_name,
             pivot.shape[0], pivot.shape[1])

    # ── 2. Mapping NGFS → hist cols ──────────────────────────────────────
    hist_to_ngfs = {v: k for k, v in ngfs_to_hist.items()}

    # Identifier les colonnes historiques nécessaires au modèle
    needed_hist_cols = set()
    for var_name in model.variables:
        hist_col, _ = parse_transformed_colname(var_name)
        needed_hist_cols.add(hist_col)

    # ── 3. Préparer l'historique pour le rebasing ────────────────────────
    df_hist = hist_df.copy()
    if hist_year_col in df_hist.columns:
        df_hist = df_hist.set_index(hist_year_col)
    df_hist.index = df_hist.index.astype(int)
    df_hist = df_hist.sort_index()

    horizon_years = sorted(pivot.index.tolist())

    # ── 4. Extraction + rebasing des niveaux NGFS ─────────────────────────
    proj_cols: Dict[str, pd.Series] = {}

    for hist_col in needed_hist_cols:
        if hist_col not in best_per_var:
            log.warning("CT [%s]: '%s' absent de best_per_var.", scenario_name, hist_col)
            continue

        # Trouver la variable NGFS correspondante
        ngfs_var = hist_to_ngfs.get(hist_col)
        if ngfs_var is None or ngfs_var not in pivot.columns:
            log.warning("CT [%s]: pas de variable NGFS pour '%s' (cherché: %s).",
                        scenario_name, hist_col, ngfs_var)
            continue

        # Série NGFS CT en niveaux
        ngfs_series = pivot[ngfs_var].dropna()

        # Série historique
        hist_series = df_hist[hist_col].dropna() if hist_col in df_hist.columns \
            else pd.Series(dtype=float)

        # ── REBASING : aligner l'échelle NGFS (agrégat régional) ─────────
        # sur l'échelle historique (pays) via l'année de chevauchement.
        overlap_years = sorted(set(hist_series.index) & set(ngfs_series.index))
        if overlap_years:
            anchor_year = overlap_years[-1]
            hist_val = float(hist_series.loc[anchor_year])
            ngfs_val = float(ngfs_series.loc[anchor_year])
            if abs(ngfs_val) > 1e-12 and abs(hist_val) > 1e-12:
                ratio = hist_val / ngfs_val
                ngfs_series = ngfs_series * ratio
                log.debug("CT rebase [%s]: année=%d, ratio=%.4f",
                          hist_col, anchor_year, ratio)
        else:
            # Pas de chevauchement : utiliser dernier niveau historique
            if not hist_series.empty and len(ngfs_series) >= 2:
                last_hist = float(hist_series.iloc[-1])
                first_ngfs = float(ngfs_series.iloc[0])
                if abs(first_ngfs) > 1e-12:
                    ratio = last_hist / first_ngfs
                    ngfs_series = ngfs_series * ratio
                    log.debug("CT rebase [%s]: pas de chevauchement, ratio=%.4f",
                              hist_col, ratio)

        # Ne garder que les années CT dans l'horizon
        ngfs_in_horizon = ngfs_series[ngfs_series.index.isin(horizon_years)]
        if ngfs_in_horizon.empty:
            log.warning("CT [%s]: aucune année CT pour '%s'.", scenario_name, hist_col)
            continue

        proj_cols[hist_col] = ngfs_in_horizon

    if not proj_cols:
        log.warning("CT [%s]: aucune variable disponible.", scenario_name)
        return []

    proj_df = pd.DataFrame(proj_cols).sort_index()

    # Vérifier que toutes les variables du modèle sont présentes
    missing = [v for v in model.variables if v not in proj_df.columns]
    if missing:
        log.warning("CT [%s]: variables manquantes: %s", scenario_name, missing)
        return []

    # ── 5. Prédicteur linéaire + lien inverse ────────────────────────────
    coefs = model.coefficients
    intercept = float(coefs.get("const", 0.0))
    eta = pd.Series(intercept, index=proj_df.index, dtype=float)
    for v in model.variables:
        x_v = proj_df[v].astype(float)
        # Si le modèle a été estimé sur des variables standardisées,
        # appliquer la même standardisation sur les données de projection.
        if model.scaler and v in model.scaler:
            mean_v, std_v = model.scaler[v]
            x_v = (x_v - mean_v) / (std_v if std_v != 0.0 else 1.0)
        eta = eta + float(coefs.get(v, 0.0)) * x_v

    if target_logit_transformed:
        y_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
    elif model.family == "OLS":
        y_hat = eta
    elif model.family in ("Logit", "Beta"):
        y_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
    elif model.family == "Vasicek-OLS":
        from scipy.stats import norm
        y_hat = pd.Series(norm.cdf(eta.values), index=eta.index)
    else:
        y_hat = eta

    log.info("CT [%s]: %d points projetés (raw levels).", scenario_name, len(y_hat))
    return [
        {"year": int(yr), "value": (None if pd.isna(v) else float(v))}
        for yr, v in y_hat.items()
    ]


def _build_ngfs_pivot(ngfs_country, scenario_name, scenario_channel,
                      baseline_name: Optional[str] = None):
    """
    Pivot NGFS data for a given scenario + channel.

    Si le channel demandé est absent pour ce scénario, sélectionne
    automatiquement le channel disponible le plus intense (Option B).
    """
    if scenario_channel is None:
        rows = ngfs_country[
            (ngfs_country["scenario"] == scenario_name)
            & (ngfs_country["channel"].isna())
        ]
    else:
        rows = ngfs_country[
            (ngfs_country["scenario"] == scenario_name)
            & (ngfs_country["channel"] == scenario_channel)
        ]
        # Fallback niveau 1 : channel absent → prendre le plus intense disponible
        if rows.empty:
            available = (
                ngfs_country.loc[
                    ngfs_country["scenario"] == scenario_name, "channel"
                ]
                .dropna().unique().tolist()
            )
            if available and baseline_name and scenario_name != baseline_name:
                # Scénarios stressés : comparer à baseline pour trouver le plus intense
                best_ch = _pick_most_intense_channel(
                    ngfs_country=ngfs_country,
                    scenario=scenario_name,
                    baseline_scenario=baseline_name,
                    available_channels=available,
                )
                rows = ngfs_country[
                    (ngfs_country["scenario"] == scenario_name)
                    & (ngfs_country["channel"] == best_ch)
                ]
                log.info(
                    "_build_ngfs_pivot [%s]: channel '%s' absent → proxy '%s'.",
                    scenario_name, scenario_channel, best_ch,
                )
            elif available:
                # Baseline ou scénario sans référence : prendre le premier channel dispo
                best_ch = available[0]
                rows = ngfs_country[
                    (ngfs_country["scenario"] == scenario_name)
                    & (ngfs_country["channel"] == best_ch)
                ]
                log.info(
                    "_build_ngfs_pivot [%s]: channel '%s' absent → fallback channel '%s'.",
                    scenario_name, scenario_channel, best_ch,
                )

        # Fallback niveau 2 : aucun channel explicite → essayer channel NaN
        # (cas typique du scénario Baseline NGFS qui n'a pas de channel)
        if rows.empty:
            nan_rows = ngfs_country[
                (ngfs_country["scenario"] == scenario_name)
                & (ngfs_country["channel"].isna())
            ]
            if not nan_rows.empty:
                rows = nan_rows
                log.info(
                    "_build_ngfs_pivot [%s]: channel NaN utilisé (pas de channel '%s').",
                    scenario_name, scenario_channel,
                )

    if rows.empty:
        return pd.DataFrame(), {}
    pivot = (rows.groupby(["year", "variable_base"])["value"]
             .mean().unstack("variable_base").sort_index())
    units = (rows.groupby("variable_base")["unit"]
             .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "")
             .to_dict())
    return pivot, units


# ──────────────────────────────────────────────────────────────────────────────
# Vérification de cohérence du Baseline projeté
# ──────────────────────────────────────────────────────────────────────────────

def _check_baseline_coherence(
    baseline_proj: List[Dict],
    stressed_pd_dict: Optional[Dict[str, List[Dict]]] = None,
    max_pd_value: float = 0.40,
    max_annual_change: float = 0.15,
) -> Tuple[bool, List[str]]:
    """
    Vérifie la cohérence économique d'une projection Baseline de PD.

    Trois critères indépendants — un seul échec suffit à rejeter :

    C1 — Plage de valeurs
         Aucune PD projetée ne dépasse max_pd_value.
         Une PD Baseline > 40% est économiquement absurde comme référence.

    C2 — Stabilité inter-annuelle (cap sur la dérivée annuelle)
         La variation absolue de PD d'une année à l'autre ne dépasse pas
         max_annual_change (ex. 0.15 = 15 points de pourcentage par an).
         Détecte les oscillations chaotiques AR explosifs sans exception.

    C3 — Cohérence directionnelle avec les scénarios stressés
         Si des projections stressées sont fournies :
         mean(PD_baseline) ne doit pas être supérieur à mean(PD_stressed)
         pour la majorité des scénarios stressés.
         Baseline > Stressed = inversion économique.

    Parameters
    ----------
    baseline_proj    : liste de dict {year, value} — projection PD Baseline
    stressed_pd_dict : {scenario_name: [{year, value}]} — projections stressées (optionnel)
    max_pd_value     : seuil C1 (défaut 0.40)
    max_annual_change: seuil C2 (défaut 0.15 = 15pp/an)

    Returns
    -------
    (is_coherent: bool, failed_criteria: List[str])
    """
    failed: List[str] = []

    # Extraire les valeurs de PD en série temporelle ordonnée
    pd_series = pd.Series(
        {p["year"]: p["value"] for p in baseline_proj if p.get("value") is not None}
    ).sort_index().astype(float).dropna()

    if pd_series.empty:
        return False, ["C0: projection Baseline vide"]

    # ── C1 — Plage de valeurs ────────────────────────────────────────────────
    pd_max = float(pd_series.max())
    if pd_max > max_pd_value:
        failed.append(
            f"C1: PD max={pd_max:.4f} > seuil={max_pd_value:.4f} "
            f"(année {pd_series.idxmax()})"
        )

    # ── C2 — Stabilité inter-annuelle ────────────────────────────────────────
    annual_changes = pd_series.diff().abs().dropna()
    if not annual_changes.empty:
        max_change = float(annual_changes.max())
        worst_year = int(annual_changes.idxmax())
        if max_change > max_annual_change:
            failed.append(
                f"C2: variation annuelle max={max_change:.4f} > seuil={max_annual_change:.4f} "
                f"(année {worst_year}→{worst_year+1})"
            )

    # ── C3 — Cohérence directionnelle ────────────────────────────────────────
    if stressed_pd_dict:
        base_mean = float(pd_series.mean())
        n_inverted = 0
        n_total    = 0
        for sc_name, sc_proj in stressed_pd_dict.items():
            sc_vals = [p["value"] for p in sc_proj if p.get("value") is not None]
            if not sc_vals:
                continue
            sc_mean = float(np.mean(sc_vals))
            n_total += 1
            if base_mean > sc_mean:
                n_inverted += 1

        if n_total > 0 and n_inverted > n_total / 2:
            failed.append(
                f"C3: Baseline (mean={base_mean:.4f}) > scénarios stressés "
                f"pour {n_inverted}/{n_total} scénarios — inversion économique"
            )

    return len(failed) == 0, failed


def _project_baseline_model(
    model: FittedModel,
    hist_df: pd.DataFrame,
    selected_hist_vars: List[str],
    horizon_years: List[int],
    best_per_var: Dict[str, VariableBestTransform],
    hist_year_col: str = "year",
    allow_var: bool = True,
    target_logit_transformed: bool = False,
    target_col: str = "Default rate",
) -> List[Dict[str, Any]]:
    """
    Projette le Baseline via AR/ARIMA directement sur la série historique de PD.
    Indépendant du modèle satellite — évite le mismatch d'unités WB vs NGFS.
    """
    # ── Préparer hist_df ─────────────────────────────────────────────────────
    df = hist_df.copy()
    if hist_year_col in df.columns:
        df = df.set_index(hist_year_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = df.index.astype(int)
        except Exception:
            pass
    df = df.sort_index()

    log.info(
        "Baseline AR: hist_df shape=%s, columns=%s, target_col='%s'",
        df.shape, list(df.columns)[:8], target_col,
    )

    # ── Trouver la colonne PD ─────────────────────────────────────────────────
    # Stratégie 1 : correspondance exacte sur target_col
    pd_col = None
    if target_col in df.columns:
        pd_col = target_col

    # Stratégie 2 : correspondance insensible à la casse
    if pd_col is None:
        for col in df.columns:
            if col.strip().lower() == target_col.strip().lower():
                pd_col = col
                break

    # Stratégie 3 : première colonne qui n'est pas une variable macro sélectionnée
    if pd_col is None:
        for col in df.columns:
            if col not in selected_hist_vars:
                pd_col = col
                break

    if pd_col is None:
        log.warning(
            "Baseline AR: impossible de trouver la colonne PD dans hist_df "
            "(colonnes: %s, macro: %s). Retour à la moyenne historique.",
            list(df.columns), selected_hist_vars[:3],
        )
        return [{"year": yr, "value": 0.03} for yr in horizon_years]

    pd_series = df[pd_col].dropna().astype(float)
    log.info(
        "Baseline AR: colonne PD = '%s', %d obs, mean=%.4f, min=%.4f, max=%.4f",
        pd_col, len(pd_series),
        float(pd_series.mean()) if len(pd_series) > 0 else 0,
        float(pd_series.min()) if len(pd_series) > 0 else 0,
        float(pd_series.max()) if len(pd_series) > 0 else 0,
    )

    if len(pd_series) < 3:
        hist_mean = max(float(pd_series.mean()), 0.001) if len(pd_series) > 0 else 0.03
        log.warning("Baseline AR: série courte (%d obs) → constante %.4f.", len(pd_series), hist_mean)
        return [{"year": yr, "value": hist_mean} for yr in horizon_years]

    hist_mean = max(float(pd_series.mean()), 0.001)
    horizon = len(horizon_years)

    # ── Projection AR/ARIMA ───────────────────────────────────────────────────
    try:
        if target_logit_transformed:
            pd_clipped = pd_series.clip(lower=0.001, upper=0.999)
            logit_series = np.log(pd_clipped / (1.0 - pd_clipped))
            is_stat, adf_pval = _adf(logit_series.values)
            if is_stat:
                p = _best_ar_order(logit_series.values)
                fc_logit = _forecast_ar(logit_series.values, p, horizon)
                log.info("Baseline AR: AR(%d) sur logit(PD) | ADF p=%.3f.", p, adf_pval)
            else:
                d = _min_diff_order(logit_series.values)
                fc_logit = _forecast_arima(logit_series.values, d, horizon)
                log.info("Baseline AR: ARIMA sur logit(PD) | ADF p=%.3f (d=%d).", adf_pval, d)
            fc_pd = 1.0 / (1.0 + np.exp(-np.clip(fc_logit, -10, 10)))
        else:
            pd_clipped = pd_series.clip(lower=0.001, upper=0.999)
            is_stat, adf_pval = _adf(pd_clipped.values)
            if is_stat:
                p = _best_ar_order(pd_clipped.values)
                fc_pd = _forecast_ar(pd_clipped.values, p, horizon)
                log.info("Baseline AR: AR(%d) sur PD | ADF p=%.3f.", p, adf_pval)
            else:
                d = _min_diff_order(pd_clipped.values)
                fc_pd = _forecast_arima(pd_clipped.values, d, horizon)
                log.info("Baseline AR: ARIMA sur PD | ADF p=%.3f (d=%d).", adf_pval, d)
            fc_pd = np.clip(fc_pd, 0.0001, 0.9999)
    except Exception as e:
        log.warning("Baseline AR: exception dans AR/ARIMA (%s) → fallback hist_mean=%.4f.", e, hist_mean)
        return [{"year": int(yr), "value": hist_mean} for yr in horizon_years]

    # ── Garde-fou : si projection converge vers 0 → hist_mean ────────────────
    fc_mean = float(np.mean(fc_pd))
    if fc_mean < 0.001:
        log.warning(
            "Baseline AR: projection dégénérée (mean=%.6f) → fallback hist_mean=%.4f.",
            fc_mean, hist_mean,
        )
        fc_pd = np.full(horizon, hist_mean)

    log.info(
        "Baseline AR: %d pts | mean=%.4f  min=%.4f  max=%.4f",
        horizon, float(np.mean(fc_pd)), float(np.min(fc_pd)), float(np.max(fc_pd)),
    )
    return [{"year": int(yr), "value": float(v)} for yr, v in zip(horizon_years, fc_pd)]


def _project_baseline_via_satellite(
    model: FittedModel,
    hist_df: pd.DataFrame,
    selected_hist_vars: List[str],
    horizon_years: List[int],
    best_per_var: Dict[str, VariableBestTransform],
    hist_year_col: str = "year",
    allow_var: bool = True,
    target_logit_transformed: bool = False,
) -> List[Dict[str, Any]]:
    """Fallback satellite — conservé pour compatibilité mais NE DEVRAIT PAS être appelé."""
    log.warning("_project_baseline_via_satellite appelé — méthode dépréciée.")
    bp = project_baseline(
        hist_df=hist_df, selected_hist_vars=selected_hist_vars,
        horizon_years=horizon_years, hist_year_col=hist_year_col, allow_var=allow_var,
    )
    macro_levels_df = build_baseline_macro_df(bp)
    if macro_levels_df.empty:
        return []
    needed = set()
    for v in model.variables:
        h, _ = parse_transformed_colname(v)
        needed.add(h)
    proj_cols: Dict[str, pd.Series] = {}
    for h in needed:
        if h not in best_per_var or h not in macro_levels_df.columns:
            continue
        s = macro_levels_df[h]
        s2 = s[s.index.isin(horizon_years)]
        if not s2.empty:
            proj_cols[h] = s2
    if not proj_cols:
        return []
    proj_df = pd.DataFrame(proj_cols).sort_index()
    if any(v not in proj_df.columns for v in model.variables):
        return []
    coefs = model.coefficients
    eta = pd.Series(float(coefs.get("const", 0.0)), index=proj_df.index, dtype=float)
    for v in model.variables:
        x_v = proj_df[v].astype(float)
        if model.scaler and v in model.scaler:
            m_, s_ = model.scaler[v]
            x_v = (x_v - m_) / (s_ if s_ != 0.0 else 1.0)
        eta = eta + float(coefs.get(v, 0.0)) * x_v
    if target_logit_transformed or model.family in ("Logit", "Beta"):
        y_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
    elif model.family == "Vasicek-OLS":
        from scipy.stats import norm
        y_hat = pd.Series(norm.cdf(eta.values), index=eta.index)
    else:
        y_hat = eta
    return [{"year": int(yr), "value": (None if pd.isna(v) else float(v))}
            for yr, v in y_hat.items()]


def _project_model(
    model: FittedModel,
    ngfs_country: pd.DataFrame,
    scenario_name: str,
    scenario_channel: Optional[str],
    ngfs_to_hist: Dict[str, str],
    best_per_var: Dict[str, VariableBestTransform],
    hist_anchor: Optional[pd.Series] = None,
    target_logit_transformed: bool = False,
    baseline_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Projette le modèle satellite choisi sur un scénario NGFS."""

    pivot, units = _build_ngfs_pivot(
        ngfs_country, scenario_name, scenario_channel,
        baseline_name=baseline_name,
    )
    if pivot.empty:
        log.warning("Projection: no rows for scenario='%s'.", scenario_name)
        return []

    # Filtrer best_per_var aux variables du modèle choisi
    needed_hist_cols = set()
    for var_name in model.variables:
        hist_col, _ = parse_transformed_colname(var_name)
        needed_hist_cols.add(hist_col)

    best_per_var_filtered = {
        k: v for k, v in best_per_var.items()
        if k in needed_hist_cols
    }

    log.info(
        "Projection [%s]: using %d raw level variable(s): %s",
        scenario_name, len(best_per_var_filtered),
        list(best_per_var_filtered.keys()),
    )

    proj_df = replay_transforms_on_ngfs(
        ngfs_pivot=pivot,
        best_per_var=best_per_var_filtered,
        ngfs_to_hist=ngfs_to_hist,
        units=units,
        hist_anchor=hist_anchor,
    )

    if proj_df.empty:
        log.warning("Projection: empty DataFrame for '%s'.", scenario_name)
        return []

    missing = [v for v in model.variables if v not in proj_df.columns]
    if missing:
        log.warning("Projection: missing %s — skipping '%s'.",
                    missing, scenario_name)
        return []

    # Prédicteur linéaire
    coefs = model.coefficients
    intercept = float(coefs.get("const", 0.0))
    eta = pd.Series(intercept, index=proj_df.index, dtype=float)
    for v in model.variables:
        x_v = proj_df[v].astype(float)
        # Si le modèle a été estimé sur des variables standardisées,
        # appliquer la même standardisation sur les données de projection.
        if model.scaler and v in model.scaler:
            mean_v, std_v = model.scaler[v]
            x_v = (x_v - mean_v) / (std_v if std_v != 0.0 else 1.0)
        eta = eta + float(coefs.get(v, 0.0)) * x_v

    # Lien inverse
    if target_logit_transformed:
        y_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
        log.info("Projection [%s]: sigmoid applied (logit target).",
                 scenario_name)
    elif model.family == "OLS":
        y_hat = eta
    elif model.family in ("Logit", "Beta"):
        y_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
    elif model.family == "Vasicek-OLS":
        from scipy.stats import norm
        y_hat = pd.Series(norm.cdf(eta.values), index=eta.index)
    else:
        y_hat = eta

    return [
        {"year": int(yr), "value": (None if pd.isna(v) else float(v))}
        for yr, v in y_hat.items()
    ]


def _project_baseline_ngfs_lt(
    model: FittedModel,
    ngfs_country: pd.DataFrame,
    baseline_name: str,
    ngfs_to_hist: Dict[str, str],
    best_per_var: Dict[str, VariableBestTransform],
    hist_df: pd.DataFrame,
    hist_year_col: str = "year",
    target_logit_transformed: bool = False,
) -> List[Dict[str, Any]]:
    """
    Projection LT Baseline via NGFS Baseline — même logique que CT MAIS
    en utilisant les TAUX DE VARIATION NGFS appliqués aux dernières valeurs
    historiques, au lieu des niveaux absolus.

    Pourquoi cette approche :
      Les données NiGEM (LT) sont en unités régionales PPP 2017 USD Bn.
      Les données historiques sont en unités pays (EGP Bn, current USD, %).
      Appliquer les niveaux absolus NGFS directement (même avec rebasing) place
      les variables hors du domaine d'entraînement du satellite model → PD → 0.

      En utilisant les VARIATIONS relatives (niveaux/niveau_ancre) ou
      additives (taux - taux_ancre), les variables projetées restent dans
      l'espace historique (même unités, même ordre de grandeur).

    Formule :
      Variables niveau  : proj_t = hist_last × (ngfs_t / ngfs_last_hist_year)
      Variables taux (%) : proj_t = hist_last + (ngfs_t  − ngfs_last_hist_year)
    """
    # ── 1. Pivot NGFS Baseline — tous channels, moyenne ─────────────────────
    rows = ngfs_country[ngfs_country["scenario"] == baseline_name]
    if rows.empty:
        log.warning("Baseline LT: aucune donnée pour scénario '%s'.", baseline_name)
        return []

    pivot = (rows.groupby(["year", "variable_base"])["value"]
             .mean().unstack("variable_base").sort_index())
    log.info("Baseline LT pivot [%s]: %d années × %d variables.",
             baseline_name, pivot.shape[0], pivot.shape[1])

    # ── 2. Préparer l'historique ─────────────────────────────────────────────
    df_hist = hist_df.copy()
    if hist_year_col in df_hist.columns:
        df_hist = df_hist.set_index(hist_year_col)
    df_hist.index = df_hist.index.astype(int)
    df_hist = df_hist.sort_index()
    last_hist_year = int(df_hist.index.max())

    hist_to_ngfs = {v: k for k, v in ngfs_to_hist.items()}

    # Identifier les variables nécessaires au modèle
    needed_hist_cols: set = set()
    for var_name in model.variables:
        hist_col, _ = parse_transformed_colname(var_name)
        needed_hist_cols.add(hist_col)

    # Années futures à projeter (après la dernière année historique)
    future_years = sorted([int(y) for y in pivot.index if int(y) > last_hist_year])
    if not future_years:
        log.warning("Baseline LT: aucune année future dans le pivot NGFS.")
        return []

    # ── 3. Construction des variables projetées (espace historique) ──────────
    proj_cols: Dict[str, pd.Series] = {}

    for hist_col in needed_hist_cols:
        # Variable NGFS correspondante
        ngfs_var = hist_to_ngfs.get(hist_col)
        if ngfs_var is None or ngfs_var not in pivot.columns:
            log.warning("Baseline LT: pas de variable NGFS pour '%s' (cherché: %s).",
                        hist_col, ngfs_var)
            continue

        ngfs_series = pivot[ngfs_var].dropna()

        # Valeur historique de référence (dernière année historique)
        if hist_col not in df_hist.columns:
            log.warning("Baseline LT: '%s' absent de hist_df.", hist_col)
            continue
        hist_series = df_hist[hist_col].dropna()
        if hist_series.empty:
            continue
        hist_last = float(hist_series.iloc[-1])

        # Valeur NGFS à l'année d'ancrage (last_hist_year ou plus proche)
        overlap = sorted(set(hist_series.index) & set(ngfs_series.index))
        if overlap:
            anchor_yr = overlap[-1]
        elif not ngfs_series.empty:
            anchor_yr = int(ngfs_series.index[0])
        else:
            continue

        ngfs_anchor = float(ngfs_series.loc[anchor_yr]) if anchor_yr in ngfs_series.index \
            else float(ngfs_series.iloc[0])

        # Déterminer si la variable est un taux (%) ou un niveau
        # Heuristique : valeurs historiques < 200 ET nom contient indicateur de taux
        _rate_keywords = ("rate", "ratio", "pct", "percent", "inflation",
                          "interest", "unemployment", "exchange", "cpi", "policy")
        is_rate = (
            any(kw in hist_col.lower() for kw in _rate_keywords)
            or (hist_series.abs().max() < 200)
        )

        proj: Dict[int, float] = {}
        for yr in future_years:
            if yr not in ngfs_series.index:
                continue
            ngfs_val = float(ngfs_series.loc[yr])

            if is_rate:
                # Variation additive : hist_last + (ngfs_t − ngfs_anchor)
                proj[yr] = hist_last + (ngfs_val - ngfs_anchor)
            else:
                # Variation multiplicative : hist_last × (ngfs_t / ngfs_anchor)
                if abs(ngfs_anchor) > 1e-12:
                    proj[yr] = hist_last * (ngfs_val / ngfs_anchor)
                else:
                    proj[yr] = hist_last

        if proj:
            proj_cols[hist_col] = pd.Series(proj)
            log.debug("Baseline LT '%s' (%s): anchor=%.4f hist_last=%.4f "
                      "proj_first=%.4f proj_last=%.4f",
                      hist_col, "taux" if is_rate else "niveau",
                      ngfs_anchor, hist_last,
                      list(proj.values())[0], list(proj.values())[-1])

    if not proj_cols:
        log.warning("Baseline LT: aucune variable disponible pour la projection.")
        return []

    proj_df = pd.DataFrame(proj_cols).sort_index()

    # ── 4. Vérifier que toutes les variables du modèle sont présentes ────────
    missing = [v for v in model.variables if v not in proj_df.columns]
    if missing:
        log.warning("Baseline LT: variables manquantes dans proj_df: %s.", missing)
        return []

    # ── 5. Prédicteur linéaire + lien inverse ────────────────────────────────
    coefs     = model.coefficients
    intercept = float(coefs.get("const", 0.0))
    eta = pd.Series(intercept, index=proj_df.index, dtype=float)
    for v in model.variables:
        x_v = proj_df[v].astype(float)
        if model.scaler and v in model.scaler:
            mean_v, std_v = model.scaler[v]
            x_v = (x_v - mean_v) / (std_v if std_v != 0.0 else 1.0)
        eta = eta + float(coefs.get(v, 0.0)) * x_v

    if target_logit_transformed:
        y_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
    elif model.family == "OLS":
        y_hat = eta
    elif model.family in ("Logit", "Beta"):
        y_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
    elif model.family == "Vasicek-OLS":
        from scipy.stats import norm
        y_hat = pd.Series(norm.cdf(eta.values), index=eta.index)
    else:
        y_hat = eta

    log.info("Baseline LT [%s]: %d points projetés | mean=%.4f  min=%.4f  max=%.4f",
             baseline_name, len(y_hat),
             float(y_hat.mean()), float(y_hat.min()), float(y_hat.max()))

    return [
        {"year": int(yr), "value": (None if pd.isna(v) else float(v))}
        for yr, v in y_hat.items()
    ]


def run_three_scenarios(
    cfg: EngineConfig,
) -> Dict[str, Any]:
    is_ct = getattr(cfg, "ngfs_mode", "LT") == "CT"

    # ── Seul le Baseline est fixé — adverse/severe seront auto-sélectionnés
    baseline_name = getattr(cfg, "baseline_scenario", "Baseline")
    scn_map: Dict[str, str] = {
        "baseline": baseline_name,
        "adverse":  "__pending__",
        "severe":   "__pending__",
    }

    log.info(
        "=== 3-SCENARIO STRESS TEST [%s] ===  "
        "baseline=%s | adverse=AUTO | severe=AUTO",
        "CT" if is_ct else "LT",
        scn_map["baseline"],
    )

    # ── Chargement ────────────────────────────────────────────────────
    ngfs_long = load_ngfs_for_stress_test(cfg, scn_map)
    ngfs_country, resolved_region = select_geography(ngfs_long, cfg)
    hist_df = load_historical(cfg)

    # ── Stage 1 ──────────────────────────────────────────────────────
    # Tourne sur TOUS les scénarios non-baseline disponibles dans le fichier
    # (adverse/severe encore inconnus à ce stade)
    all_scenarios = [
        s for s in ngfs_country["scenario"].dropna().unique()
        if s != scn_map["baseline"]
    ]
    if not all_scenarios:
        raise RuntimeError(
            f"Aucun scénario non-baseline trouvé dans les données NGFS. "
            f"Vérifiez le fichier chargé."
        )

    # ── Filtre famille de scénarios (CT uniquement) ───────────────────
    # En LT (NGFS NiGEM), les scénarios n'ont pas de structure famille/run.
    # En CT (GEM-E3), on filtre : exclure _run1, garder 1 représentant
    # par famille (DAPS, DIRE, HWTP, SWUC).
    if is_ct:
        all_scenarios = filter_scenario_families(
            scenarios=all_scenarios,
            baseline_name=scn_map["baseline"],
            cfg=cfg,
        )
        if not all_scenarios:
            raise RuntimeError(
                "ScenarioFamilyFilter: aucun scénario retenu après filtrage "
                "des familles CT. Vérifiez le fichier GEM-E3."
            )

    log.info("Stage 1: %d scénarios candidats: %s", len(all_scenarios), all_scenarios)

    stage1_per_scn: Dict[str, Stage1Result] = {}
    for sc in all_scenarios:
        try:
            s1_cfg = copy.copy(cfg)
            s1_cfg.scenario = sc
            s1_cfg.baseline_scenario = scn_map["baseline"]
            s1 = stage1_filter(ngfs_country, s1_cfg)
            stage1_per_scn[sc] = s1
            log.info("Stage 1 [%s]: %d variables.", sc, len(s1.selected_variables))
        except Exception as e:
            log.warning("Stage 1 [%s]: ignoré (%s).", sc, e)

    if not stage1_per_scn:
        raise RuntimeError(
            "Stage 1 a échoué pour tous les scénarios candidats. "
            "Vérifiez le channel configuré et les données NGFS."
        )

    union_vars = sorted(set().union(
        *(s.selected_variables for s in stage1_per_scn.values())
    ))
    log.info("Stage 1 UNION: %d variables sur %d scénarios valides.",
             len(union_vars), len(stage1_per_scn))

    # ── Variable Resolution (file → WB API → IMF API → DROP) ────────
    # En mode CT : utiliser les données régionales pour éviter le rebasing
    # qui efface les différences entre scénarios NGFS.
    override_iso2 = None
    if is_ct and getattr(cfg, "use_regional_data_ct", False):
        # Priorité : code explicite > auto-dérivé depuis le pays
        explicit = getattr(cfg, "ct_region_code", None)
        if explicit:
            override_iso2 = explicit
            log.info("CT mode: regional data (explicit) '%s'.", override_iso2)
        else:
            from .variable_resolver import get_wb_region_code
            override_iso2 = get_wb_region_code(cfg.country)
            if override_iso2:
                log.info(
                    "CT mode: regional data (auto-derived) '%s' for country '%s'.",
                    override_iso2, cfg.country,
                )
            else:
                log.warning(
                    "CT mode: could not derive WB region for '%s'. "
                    "Falling back to country data.", cfg.country,
                )
                override_iso2 = None

    resolver_result: ResolverResult = resolve_variables(
        ngfs_vars=union_vars,
        cfg=cfg,
        hist_df=hist_df,
        override_iso2=override_iso2,
    )
    print_resolution_log(resolver_result)

    # Utiliser le hist_df enrichi par l'API
    hist_df = resolver_result.enriched_hist_df

    # ── Stage 2 ──────────────────────────────────────────────────────
    s2: Stage2Result = run_stage2(
        union_vars, hist_df, cfg,
        pre_mapping=resolver_result.mapping,
    )

    # ── Variables communes nécessaires avant la validation et la projection
    ngfs_to_hist = s2.mapping.matches if s2.mapping else {}
    best_per_var = s2.transform_result.best_per_var \
        if s2.transform_result else {}

    # Anchor LT : moyenne des dernières années historiques par variable
    hist_anchor: Optional[pd.Series] = None
    if not is_ct and best_per_var:
        anchor_n_years = 3
        available_cols = [c for c in best_per_var.keys()
                          if c in hist_df.columns]
        if available_cols:
            df_h = hist_df.copy()
            if cfg.hist_year_col in df_h.columns:
                df_h = df_h.set_index(cfg.hist_year_col)
            df_h.index = df_h.index.astype(int)
            hist_anchor = (
                df_h.sort_index().tail(anchor_n_years)[available_cols]
                .mean(axis=0)
            )

    # Horizon années + variables historiques sélectionnées
    horizon_years = sorted(
        ngfs_country["year"].dropna().astype(int).unique().tolist()
    )

    # ── Stress validation des modèles ─────────────────────────────────
    if getattr(cfg, "enable_stress_validator", True):
        log.info("── Model Stress Validation ──────────────────────────────────")

        val_result = validate_stress_models(
            fitted_models=s2.fitted_models,
            hist_df=hist_df,
            ngfs_country=ngfs_country,
            baseline_name=scn_map["baseline"],
            ngfs_to_hist=ngfs_to_hist,
            best_per_var=best_per_var,
            cfg=cfg,
            is_ct=is_ct,
            project_fn_lt=lambda **kw: _project_model(
                **kw, baseline_name=scn_map["baseline"]
            ),
            project_fn_ct=_project_ct_levels,
            hist_anchor=hist_anchor,
            target_logit_transformed=s2.target_logit_transformed,
        )

        if val_result.n_validated == 0:
            log.error(
                "ModelStressValidator: AUCUN modèle ne passe les filtres F0–F4 "
                "(%d modèles testés). "
                "Détail rejets → F0(stat)=%d F1(PD)=%d F2(signes)=%d "
                "F3(stress)=%d F4(saturation)=%d. "
                "Causes probables: données historiques insuffisantes, "
                "R² trop faible, ou variables sans pouvoir de transmission du choc. "
                "Fallback sur tous les modèles estimés — RÉSULTATS NON FIABLES.",
                val_result.n_input,
                val_result.rejected_f0,
                val_result.rejected_f1,
                val_result.rejected_f2,
                val_result.rejected_f3,
                val_result.rejected_f4,
            )
            validated_models  = s2.fitted_models
            stress_scores_map: Dict[str, float] = {}
        else:
            validated_models  = val_result.validated_models
            stress_scores_map = get_stress_scores_dict(val_result)
            log.info(
                "ModelStressValidator: %d/%d modèles validés. "
                "Présentation à l'utilisateur du meilleur en premier.",
                val_result.n_validated, val_result.n_input,
            )
    else:
        log.info("ModelStressValidator: désactivé (cfg.enable_stress_validator=False).")
        validated_models  = s2.fitted_models
        stress_scores_map = {}

    from .model_ranking import build_model_table
    validated_table = build_model_table(
        validated_models, cfg, stress_scores=stress_scores_map
    )

    # ── SÉLECTION DU MODÈLE PAR L'UTILISATEUR ────────────────────────
    auto_select = getattr(cfg, "auto_select_model", False)
    chosen_model = select_model_interactive(
        models=validated_models,
        table=validated_table,
        auto_select=auto_select,
    )

    # ── Projection ───────────────────────────────────────────────────
    projections: Dict[str, List[Dict[str, Any]]] = {}

    if chosen_model is not None and s2.transform_result is not None:
        # ngfs_to_hist, best_per_var, hist_anchor, horizon_years
        # déjà définis avant la validation du modèle

        # selected_hist_vars propre à chosen_model
        selected_hist_vars = []
        for var_name in chosen_model.variables:
            hist_col, _ = parse_transformed_colname(var_name)
            if hist_col not in selected_hist_vars:
                selected_hist_vars.append(hist_col)

        # ── Projection Baseline ───────────────────────────────────────────
        # CT : niveaux NGFS absolus via _project_ct_levels (GEM-E3 = données pays)
        # LT : variations NGFS Baseline appliquées aux valeurs historiques
        #      via _project_baseline_ngfs_lt (NiGEM = données régionales PPP)
        if is_ct:
            baseline_proj = _project_ct_levels(
                model=chosen_model,
                ngfs_country=ngfs_country,
                scenario_name=scn_map["baseline"],
                ngfs_to_hist=ngfs_to_hist,
                best_per_var=best_per_var,
                hist_df=hist_df,
                hist_year_col=cfg.hist_year_col,
                target_logit_transformed=s2.target_logit_transformed,
            )
        else:
            baseline_proj = _project_baseline_ngfs_lt(
                model=chosen_model,
                ngfs_country=ngfs_country,
                baseline_name=scn_map["baseline"],
                ngfs_to_hist=ngfs_to_hist,
                best_per_var=best_per_var,
                hist_df=hist_df,
                hist_year_col=cfg.hist_year_col,
                target_logit_transformed=s2.target_logit_transformed,
            )

        log.info(
            "Baseline [%s | %s]: %d points projetés (hors ancrage historique).",
            "CT" if is_ct else "LT", scn_map["baseline"], len(baseline_proj),
        )
        if not baseline_proj:
            log.error(
                "Baseline vide — vérifiez que le scénario '%s' existe "
                "dans les données NGFS et que les variables sont mappées.",
                scn_map["baseline"],
            )

        baseline_series = pd.Series(
            {p["year"]: p["value"] for p in baseline_proj if p["value"] is not None}
        ).sort_index().astype(float)

        log.info(
            "Baseline pre-projection: %d points (will be injected into pd_matrix).",
            len(baseline_series.dropna()),
        )

        # ── Sélection automatique des scénarios ──────────────────────
        pd_matrix = build_pd_matrix(
            ngfs_country=ngfs_country,
            chosen_model=chosen_model,
            baseline_name=scn_map["baseline"],
            ngfs_to_hist=ngfs_to_hist,
            best_per_var=best_per_var,
            hist_df=hist_df,
            hist_year_col=cfg.hist_year_col,
            target_logit_transformed=s2.target_logit_transformed,
            is_ct=is_ct,
            scenario_channel=cfg.risk_channel,
            hist_anchor=hist_anchor,
            project_fn_lt=lambda **kw: _project_model(
                **kw, baseline_name=scn_map["baseline"]
            ),
            project_fn_ct=_project_ct_levels,
            baseline_series=baseline_series,
            candidate_scenarios=all_scenarios + [scn_map["baseline"]],
        )
        sel_result = find_optimal_scenario_pair(
            pd_matrix=pd_matrix,
            baseline_name=scn_map["baseline"],
            enforce_cross_family=is_ct,
        )
        if sel_result.valid:
            scn_map["adverse"] = sel_result.adverse
            scn_map["severe"]  = sel_result.severe
            log.info("Auto-selected: adverse='%s'  severe='%s'  score=%.4f",
                     sel_result.adverse, sel_result.severe, sel_result.score)
        else:
            log.error(
                "ScenarioSelector: no valid pair found after all fallbacks "
                "(%d candidates, %d rejected). "
                "Using the 2 most deviated scenarios from Baseline as fallback.",
                sel_result.n_candidates,
                len(sel_result.rejected_df),
            )
            # Fallback final : sélectionner par déviation absolue maximale
            non_baseline = [s for s in ngfs_country["scenario"].dropna().unique()
                           if s != scn_map["baseline"]]
            if len(non_baseline) >= 2:
                base_proj = {p["year"]: p["value"] for p in baseline_proj
                            if p["value"] is not None}
                base_mean = np.mean(list(base_proj.values())) if base_proj else 0.0
                deviations = {}
                for sc in non_baseline:
                    sc_data = ngfs_country[ngfs_country["scenario"] == sc]
                    if not sc_data.empty:
                        deviations[sc] = abs(sc_data["value"].mean() - base_mean)
                if deviations:
                    sorted_scn = sorted(deviations, key=deviations.get, reverse=True)
                    scn_map["severe"]  = sorted_scn[0]
                    scn_map["adverse"] = sorted_scn[1] if len(sorted_scn) > 1 else sorted_scn[0]
                    log.warning(
                        "Fallback assignment: adverse='%s'  severe='%s'",
                        scn_map["adverse"], scn_map["severe"],
                    )

        # ── Exploration interactive des scénarios ────────────────────
        # Permet à l'utilisateur de parcourir tt les paires depuis pd_matrix
        # et de choisir la combinaison qui lui convient.
        auto_accept_scn = getattr(cfg, "auto_accept_scenarios", False)
        adv_chosen, sev_chosen = select_scenario_pair_interactive(
            sel_result=sel_result,
            pd_matrix=pd_matrix,
            baseline_name=scn_map["baseline"],
            auto_accept=auto_accept_scn,
        )
        scn_map["adverse"] = adv_chosen
        scn_map["severe"]  = sev_chosen

        # ── Projections finales ──────────────────────────────────────
        if is_ct:
            projections["baseline"] = baseline_proj
            log.info("Projection CT [baseline]: %d points (reused).", len(baseline_proj))
            for tag in ("adverse", "severe"):
                sc = scn_map[tag]
                projections[tag] = _project_ct_levels(
                    model=chosen_model,
                    ngfs_country=ngfs_country,
                    scenario_name=sc,
                    ngfs_to_hist=ngfs_to_hist,
                    best_per_var=best_per_var,
                    hist_df=hist_df,
                    hist_year_col=cfg.hist_year_col,
                    target_logit_transformed=s2.target_logit_transformed,
                )
                log.info("Projection CT [%s → %s]: %d points.", tag, sc, len(projections[tag]))
        else:
            # Tous les scénarios via NGFS (cohérence méthodologique)
            projections["baseline"] = baseline_proj
            log.info("Projection LT [baseline → NGFS]: %d points (reused).",
                     len(baseline_proj))
            for tag in ("adverse", "severe"):
                sc = scn_map[tag]
                projections[tag] = _project_model(
                    model=chosen_model,
                    ngfs_country=ngfs_country,
                    scenario_name=sc,
                    scenario_channel=cfg.risk_channel,
                    ngfs_to_hist=ngfs_to_hist,
                    best_per_var=best_per_var,
                    hist_anchor=hist_anchor,
                    target_logit_transformed=s2.target_logit_transformed,
                    baseline_name=scn_map["baseline"],
                )
                log.info("Projection LT [%s → %s]: %d points.", tag, sc, len(projections[tag]))

        # ── Ancrage historique : remplacer les années < projection_start_year ─
        # Les années pour lesquelles on a des données historiques réelles
        # sont remplacées dans TOUS les scénarios.
        # projection_start_year = None → auto-dérivé = max(années hist_df) + 1
        target_col = cfg.target_variable

        # Calculer projection_start_year dynamiquement
        configured_start = getattr(cfg, "projection_start_year", None)
        if configured_start is not None:
            proj_start = int(configured_start)
        else:
            # Auto-dériver depuis le dataset historique
            df_tmp = hist_df.copy()
            if cfg.hist_year_col in df_tmp.columns:
                df_tmp = df_tmp.set_index(cfg.hist_year_col)
            df_tmp.index = df_tmp.index.astype(int)
            if target_col in df_tmp.columns:
                last_hist_year = int(df_tmp[target_col].dropna().index.max())
                proj_start = last_hist_year + 1
            else:
                proj_start = int(df_tmp.index.max()) + 1
            log.info(
                "projection_start_year auto-dérivé: %d (max hist=%d).",
                proj_start, proj_start - 1,
            )

        if target_col in hist_df.columns:
            df_h = hist_df.copy()
            if cfg.hist_year_col in df_h.columns:
                df_h = df_h.set_index(cfg.hist_year_col)
            df_h.index = df_h.index.astype(int)
        else:
            # target_col absent du hist_df enrichi → recharger l'historique brut
            log.info(
                "Historical anchoring: '%s' absent du hist_df enrichi "
                "→ rechargement depuis '%s'.",
                target_col, cfg.historical_path,
            )
            df_h = load_historical(cfg)
            if cfg.hist_year_col in df_h.columns:
                df_h = df_h.set_index(cfg.hist_year_col)
            df_h.index = df_h.index.astype(int)
            hist_pd = df_h[target_col].dropna()
            hist_pd = hist_pd[hist_pd.index < proj_start]

            if not hist_pd.empty:
                n_anchored_total = 0
                for tag in ("baseline", "adverse", "severe"):
                    if tag not in projections or not projections[tag]:
                        continue
                    anchored: List[Dict[str, Any]] = []
                    n_replaced = 0
                    for pt in projections[tag]:
                        yr = int(pt["year"])
                        if yr in hist_pd.index:
                            anchored.append({"year": yr,
                                            "value": float(hist_pd.loc[yr])})
                            n_replaced += 1
                        else:
                            anchored.append(pt)
                    projections[tag] = anchored
                    n_anchored_total += n_replaced

                log.info(
                    "Historical anchoring: years < %d → actual historical PD "
                    "(%d values). Projection starts at %d.",
                    proj_start, len(hist_pd), proj_start,
                )

    else:
        raise RuntimeError(
            "Aucun modèle satellite sélectionné ou transform_result manquant. "
            "Impossible de lancer la sélection automatique des scénarios."
        )
    # ── Output JSON ──────────────────────────────────────────────────
    # Convertir le model_table pour JSON (sans la colonne interne _name)
    table_for_json = s2.model_table.drop(columns=["_name"], errors="ignore")

    output: Dict[str, Any] = {
        "metadata": {
            "country": cfg.country,
            "resolved_region": resolved_region,
            "ngfs_mode": "CT" if is_ct else "LT",
            "scenario_map": scn_map,
            "risk_channel_for_stressed": (None if is_ct else cfg.risk_channel),
            "risk_type": cfg.risk_type,
            "target_variable": cfg.target_variable,
            "target_logit_transformed": s2.target_logit_transformed,
        },
        "stage1_per_scenario": {
            sc: {
                "ngfs_scenario": sc,
                "selected_variables": s1.selected_variables,
                "n_selected": len(s1.selected_variables),
            }
            for sc, s1 in stage1_per_scn.items()
        },
        "stage1_union": union_vars,
        "stage2_historical": {
            "mapping": {
                "ngfs_to_historical": s2.mapping.matches if s2.mapping else {},
                "method_per_match": s2.mapping.method_per_match if s2.mapping else {},
                "unmatched_dropped": s2.mapping.unmatched if s2.mapping else [],
            },
            "best_transform_per_variable": {
                var: {"method": vbt.best_method, "abs_r": round(vbt.best_abs_r, 4)}
                for var, vbt in (
                    s2.transform_result.best_per_var.items()
                    if s2.transform_result else {}
                )
            },
            "correlation_with_target": _df_to_json(s2.correlation_table),
            "vif_removed": s2.vif_removed,
            "vif_final": _df_to_json(s2.vif_final),
            "backward_kept": s2.backward_kept,
        },
        "model_comparison_table": _df_to_json(table_for_json, max_rows=100),
        "selected_model": _model_to_json(chosen_model) if chosen_model else None,
        "insample_fit": {
            "years":  chosen_model.obs_years     if chosen_model else [],
            "actual": chosen_model.actual_values if chosen_model else [],
            "fitted": chosen_model.fitted_values if chosen_model else [],
            "family": chosen_model.family        if chosen_model else "",
            "vars":   chosen_model.variables     if chosen_model else [],
        },
        "insample_fits": [
            {
                "years":  m.obs_years,
                "actual": m.actual_values,
                "fitted": m.fitted_values,
                "family": m.family,
                "vars":   list(m.variables),
            }
            for m in (validated_models or [])
        ],
        # ══ INSERTION ICI ════════════════════════════════════════════
        "scenario_selection": {
            "method":            "automatic" if sel_result.valid else "default",
            "n_candidates":      sel_result.n_candidates,
            "selected_adverse":  sel_result.adverse,
            "selected_severe":   sel_result.severe,
            "score":             sel_result.score,
            "soft_scores":       sel_result.soft_scores,
            "ranking": (
                sel_result.ranking_df[["adverse", "severe", "score"]]
                .head(10)
                .to_dict(orient="records")
            ) if not sel_result.ranking_df.empty else [],
            "rejected_count": len(sel_result.rejected_df),
        },
        # ══ FIN INSERTION ════════════════════════════════════════════
        "projections": {
            tag: {
                "ngfs_scenario": scn_map[tag],
                "channel": (None if is_ct or tag == "baseline" else cfg.risk_channel),
                "path": projections.get(tag, []),
            }
            for tag in ("baseline", "adverse", "severe")
        },
        "diagnostics": {
            "correlation_matrix": (
                s2.correlation_matrix.replace({np.nan: None}).to_dict()
            ),
        },
    }

    if cfg.output_path:
        import pathlib, tempfile, shutil, os
        out_path = pathlib.Path(cfg.output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json_str = json.dumps(output, indent=2, default=str)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=str(out_path.parent))
        try:
            os.write(tmp_fd, json_str.encode("utf-8"))
            os.close(tmp_fd)
            shutil.move(tmp_path, str(out_path))
        except Exception:
            os.close(tmp_fd)
            os.unlink(tmp_path)
            raise
        log.info("Wrote 3-scenario output: %s", out_path)

    return output
