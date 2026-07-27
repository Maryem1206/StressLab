"""
scenario_shock_builder.py
=========================
Construit les scénarios stressés (Adverse, Severe) à partir du Baseline
time-series + chocs NGFS calibrés.

Formule :
    Δ(t) = NGFS_stress(t) - NGFS_baseline(t)      ← choc pur (pas de tendance)
    x_stress(t) = x_baseline_ts(t) + Δ(t)          ← macro stressée

Avantages vs scénario indépendant :
  - Cohérence économique : le choc est ancré sur NOTRE baseline TS
  - Ordre Baseline ≤ Adverse ≤ Severe garanti via monotonicity enforcement
  - Interprétable : Δ(t) mesure l'écart pur dû au scénario climatique
  - Générique : fonctionne pour tout pays, tout risque, tout dataset

Généricité :
  - Aucune hypothèse sur le nombre de scénarios
  - Le signe du choc est dérivé automatiquement depuis les β du satellite model
  - Le mode de monotonicity enforcement est configurable
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .model_estimation import FittedModel
from .transform_selector import (
    VariableBestTransform,
    apply_transform,
    parse_transformed_colname,
)
from .utils import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────────────────

def _ngfs_pivot(
    ngfs_country: pd.DataFrame,
    scenario_name: str,
    scenario_channel: Optional[str],
) -> pd.DataFrame:
    """
    Pivote les données NGFS pour un scénario donné.

    Returns
    -------
    DataFrame (index=year, columns=variable_base) ou DataFrame vide.
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

    if rows.empty:
        return pd.DataFrame()

    return (
        rows.groupby(["year", "variable_base"])["value"]
        .mean()
        .unstack("variable_base")
        .sort_index()
    )


def _derive_col_beta_sign(model_coefs: Dict[str, float]) -> Dict[str, int]:
    """
    Dérive le signe du coefficient β par hist_col depuis les coefs du satellite.

    Les coefs sont indexés sur les noms transformés (ex: "gdp__lag1_pct"),
    on remonte au hist_col (ex: "gdp") via parse_transformed_colname.

    En cas de conflit entre transforms d'une même variable,
    le signe est mis à 0 (ambigu → monotonicity non appliquée).

    Returns
    -------
    Dict[hist_col, int]  où int ∈ {-1, 0, +1}
    """
    col_sign: Dict[str, int] = {}
    for var, coef in model_coefs.items():
        if var == "const":
            continue
        try:
            hist_col, _ = parse_transformed_colname(var)
        except Exception:
            hist_col = var

        sign = int(np.sign(coef)) if coef != 0.0 else 0

        if hist_col not in col_sign:
            col_sign[hist_col] = sign
        elif col_sign[hist_col] != sign:
            col_sign[hist_col] = 0  # conflit → ambigu

    return col_sign


# ─────────────────────────────────────────────────────────────────────────────
# API publique
# ─────────────────────────────────────────────────────────────────────────────

def build_shocked_macro(
    baseline_macro_df: pd.DataFrame,
    ngfs_country: pd.DataFrame,
    stress_scenario: str,
    baseline_scenario: str,
    ngfs_to_hist: Dict[str, str],
    scenario_channel: Optional[str],
    model_coefs: Optional[Dict[str, float]] = None,
    enforce_monotone_vs: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Retourne un DataFrame de niveaux macro stressés :

        Δ(t)         = NGFS_stress(t) - NGFS_baseline(t)
        x_stress(t)  = x_baseline_ts(t) + Δ(t)

    Parameters
    ----------
    baseline_macro_df   : niveaux macro issus du Baseline time-series
                          (AR/ARIMA/VAR). Index = année (int), colonnes = hist_col.
    ngfs_country        : données NGFS longues (toutes vars, tous scénarios).
    stress_scenario     : nom du scénario NGFS stressé (ex: "Delayed transition").
    baseline_scenario   : nom du scénario NGFS baseline (ex: "Baseline").
    ngfs_to_hist        : mapping ngfs_var → hist_col.
    scenario_channel    : channel NGFS (ex: "combined") ou None.
    model_coefs         : coefficients du satellite model — nécessaires pour
                          le monotonicity enforcement (optionnel).
    enforce_monotone_vs : DataFrame de niveaux macro d'un scénario moins sévère
                          déjà calculé. Si fourni, la sévérité est bornée pour
                          être au moins aussi stressante que ce scénario.
                          (ex: passer shocked_adverse lors du calcul de severe)

    Returns
    -------
    DataFrame (index=année, colonnes=hist_col) avec niveaux macro stressés.
    Si les données NGFS sont absentes, retourne baseline_macro_df inchangé.
    """
    # ── Pivots NGFS ───────────────────────────────────────────────────────────
    pivot_stress   = _ngfs_pivot(ngfs_country, stress_scenario,   scenario_channel)
    pivot_baseline = _ngfs_pivot(ngfs_country, baseline_scenario, scenario_channel)

    if pivot_stress.empty:
        log.warning(
            "ShockBuilder [%s]: aucune donnée NGFS — baseline retourné sans choc.",
            stress_scenario,
        )
        return baseline_macro_df.copy()

    if pivot_baseline.empty:
        log.warning(
            "ShockBuilder [%s]: NGFS Baseline absent — choc = 0 (baseline retourné).",
            stress_scenario,
        )
        return baseline_macro_df.copy()

    # ── Calcul des chocs : Δ = NGFS_stress - NGFS_baseline ───────────────────
    common_years = sorted(set(pivot_stress.index) & set(pivot_baseline.index))
    common_vars  = sorted(set(pivot_stress.columns) & set(pivot_baseline.columns))

    if not common_years or not common_vars:
        log.warning(
            "ShockBuilder [%s]: pas d'années/variables communes avec le "
            "Baseline NGFS — baseline retourné sans choc.",
            stress_scenario,
        )
        return baseline_macro_df.copy()

    delta_ngfs = (
        pivot_stress.loc[common_years, common_vars]
        - pivot_baseline.loc[common_years, common_vars]
    )

    log.info(
        "ShockBuilder [%s]: Δ calculé sur %d années × %d variables NGFS.",
        stress_scenario, len(common_years), len(common_vars),
    )

    # ── Application des chocs sur le Baseline TS ─────────────────────────────
    shocked = baseline_macro_df.copy()

    for ngfs_var, hist_col in ngfs_to_hist.items():
        if ngfs_var not in delta_ngfs.columns:
            continue
        if hist_col not in shocked.columns:
            log.debug(
                "ShockBuilder [%s]: '%s' absent du baseline macro — ignoré.",
                stress_scenario, hist_col,
            )
            continue

        delta_series = delta_ngfs[ngfs_var]
        overlap      = sorted(set(shocked.index) & set(delta_series.index))

        if not overlap:
            log.warning(
                "ShockBuilder [%s]: pas d'années communes pour '%s' — ignoré.",
                stress_scenario, hist_col,
            )
            continue

        shocked.loc[overlap, hist_col] = (
            baseline_macro_df.loc[overlap, hist_col].values
            + delta_series.loc[overlap].values
        )
        log.debug(
            "ShockBuilder [%s | %s]: Δ moyen = %+.4f sur %d années.",
            stress_scenario, hist_col,
            float(delta_series.loc[overlap].mean()), len(overlap),
        )

    # ── Monotonicity enforcement (optionnel) ──────────────────────────────────
    if enforce_monotone_vs is not None and model_coefs is not None:
        shocked = _enforce_monotonicity(
            shocked_stress=shocked,
            shocked_less_severe=enforce_monotone_vs,
            model_coefs=model_coefs,
            label=stress_scenario,
        )

    return shocked


def _enforce_monotonicity(
    shocked_stress: pd.DataFrame,
    shocked_less_severe: pd.DataFrame,
    model_coefs: Dict[str, float],
    label: str,
) -> pd.DataFrame:
    """
    S'assure que le scénario `shocked_stress` est au moins aussi adverse
    que `shocked_less_severe` pour chaque variable et chaque année.

    Règle par variable :
      - β > 0 (hausse → PD↑) : on veut shocked_stress ≤ shocked_less_severe
        (choc plus négatif = plus de stress)
      - β < 0 (baisse → PD↑) : on veut shocked_stress ≥ shocked_less_severe
        (choc plus positif = plus de stress)
      - β ambigu (0) : pas de correction

    En cas de violation, la valeur du scénario moins sévère est utilisée
    (conservatisme : on ne suramplifie jamais le choc).
    """
    result    = shocked_stress.copy()
    col_signs = _derive_col_beta_sign(model_coefs)

    common_cols = [c for c in result.columns if c in shocked_less_severe.columns]
    common_idx  = sorted(set(result.index) & set(shocked_less_severe.index))

    for hist_col in common_cols:
        beta_sign = col_signs.get(hist_col, 0)
        if beta_sign == 0:
            continue  # signe ambigu → pas de correction

        sev = result.loc[common_idx, hist_col]
        adv = shocked_less_severe.loc[common_idx, hist_col]

        if beta_sign > 0:
            # Variable hausse → PD↑ → on veut severe PLUS BAS qu'adverse
            violation_mask = sev > adv
        else:
            # Variable baisse → PD↑ → on veut severe PLUS HAUT qu'adverse
            violation_mask = sev < adv

        n_viol = int(violation_mask.sum())
        if n_viol > 0:
            log.info(
                "Monotonicity [%s | %s]: %d violation(s) corrigée(s) "
                "(β_sign=%+d).",
                label, hist_col, n_viol, beta_sign,
            )
            result.loc[common_idx, hist_col] = np.where(
                violation_mask, adv.values, sev.values
            )

    return result


def project_satellite_on_macro(
    model: FittedModel,
    macro_levels_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    best_per_var: Dict[str, VariableBestTransform],
    hist_year_col: str = "year",
    target_logit_transformed: bool = False,
) -> List[Dict[str, Any]]:
    """
    Applique le satellite model sur un DataFrame de niveaux macro
    (Baseline TS ou stressés).

    Flux :
      1. Reconstitue pour chaque variable : historique + niveaux projetés
         (concat pour que les transforms avec lags soient correctes)
      2. Rejoue les mêmes transforms que Stage 2 (best_per_var)
      3. Prédicteur linéaire η = intercept + Σ β_i × x_i_transformed
      4. Lien inverse selon la famille du modèle (OLS / Logit / Beta / Vasicek)

    Parameters
    ----------
    model                    : FittedModel satellite sélectionné.
    macro_levels_df          : niveaux macro (index=année int, colonnes=hist_col).
    hist_df                  : historique réel (pour concat + transforms à lags).
    best_per_var             : dict hist_col → VariableBestTransform (Stage 2).
    hist_year_col            : nom de la colonne année dans hist_df.
    target_logit_transformed : si True, applique sigmoid sur η.

    Returns
    -------
    List[{"year": int, "value": float}]
    """
    horizon_years = sorted(macro_levels_df.index.tolist())
    if not horizon_years:
        return []

    # ── Préparer l'historique indexé par année ────────────────────────────────
    df_hist = hist_df.copy()
    if hist_year_col in df_hist.columns:
        df_hist = df_hist.set_index(hist_year_col)
    df_hist.index = df_hist.index.astype(int)
    df_hist = df_hist.sort_index()

    # ── Identifier les hist_col nécessaires au modèle ─────────────────────────
    needed_hist_cols: set = set()
    for var_name in model.variables:
        hist_col, _ = parse_transformed_colname(var_name)
        needed_hist_cols.add(hist_col)

    # ── Rejouer les transformations : hist + macro stressée ───────────────────
    transformed_cols: Dict[str, pd.Series] = {}

    for hist_col in needed_hist_cols:
        if hist_col not in best_per_var:
            log.warning(
                "SatProjection: '%s' absent de best_per_var — ignoré.", hist_col
            )
            continue
        if hist_col not in macro_levels_df.columns:
            log.warning(
                "SatProjection: '%s' absent du macro_df — ignoré.", hist_col
            )
            continue

        vbt    = best_per_var[hist_col]
        method = vbt.best_method

        # Historique réel
        hist_series = (
            df_hist[hist_col].dropna()
            if hist_col in df_hist.columns
            else pd.Series(dtype=float)
        )

        # Niveaux projetés (Baseline TS ou stressés)
        proj_series = macro_levels_df[hist_col]
        proj_only   = proj_series[~proj_series.index.isin(hist_series.index)]
        full_series = pd.concat([hist_series, proj_only]).sort_index()

        # Transformation sur la série complète (pour cohérence des lags)
        transformed = apply_transform(full_series, method)

        # Ne garder que les années de projection
        mask = transformed.index.isin(horizon_years)
        if not mask.any():
            log.warning(
                "SatProjection: aucune année de projection dans '%s' transformé.",
                hist_col,
            )
            continue

        col_name = f"{hist_col}__{method}"
        transformed_cols[col_name] = transformed[mask]

    if not transformed_cols:
        log.warning("SatProjection: aucune variable transformée disponible.")
        return []

    proj_df = pd.DataFrame(transformed_cols).sort_index()

    # ── Vérification des variables du modèle ─────────────────────────────────
    missing = [v for v in model.variables if v not in proj_df.columns]
    if missing:
        log.warning("SatProjection: variables manquantes: %s", missing)
        return []

    # ── Prédicteur linéaire ───────────────────────────────────────────────────
    coefs     = model.coefficients
    intercept = float(coefs.get("const", 0.0))
    eta       = pd.Series(intercept, index=proj_df.index, dtype=float)
    for v in model.variables:
        x_v = proj_df[v].astype(float)
        # Standardisation si le modèle a été estimé sur des vars standardisées
        if model.scaler and v in model.scaler:
            mean_v, std_v = model.scaler[v]
            x_v = (x_v - mean_v) / (std_v if std_v != 0.0 else 1.0)
        eta = eta + float(coefs.get(v, 0.0)) * x_v

    # ── Lien inverse selon la famille ────────────────────────────────────────
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

    log.info("SatProjection: %d points projetés.", len(y_hat))
    return [
        {"year": int(yr), "value": (None if pd.isna(v) else float(v))}
        for yr, v in y_hat.items()
    ]
