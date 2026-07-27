"""
model_stress_validator.py
=========================
Filtre et classe les modèles satellites selon leur cohérence sous stress.

Pipeline (cascade de 4 filtres durs + scoring composite) :

1. FILTRE 1 — Plausibilité des niveaux de PD historiques
   PD ajustée sur historique ∈ [pd_min_plausible, pd_max_plausible]

2. FILTRE 2 — Cohérence des signes économiques
   Pour chaque variable : signe(β) conforme au prior éco

3. FILTRE 3 — Direction du stress
   Au moins N scénarios NGFS produisent PD_stressed > PD_baseline

4. FILTRE 4 — Non-saturation
   PD projetée ne touche pas les bornes 0 ou 1

Score composite (sur modèles survivants) :
- S1 stress_direction   : Δmean(stressed - baseline) normalisé
- S2 stress_spread      : amplitude max - min sur scénarios stressés
- S3 scenario_dispersion: std des PD moyennes par scénario
- S4 pd_plausibility    : distance au centre de la plage plausible
- S5 sign_consistency   : % de signes corrects
- S6 fit_quality        : R² du modèle

Design :
- Générique : agnostique au risque, au pays, à la cible
- Découplé : reçoit les fonctions de projection en paramètres
- Configurable : tous les seuils via EngineConfig
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from .model_estimation import FittedModel
from .transform_selector import parse_transformed_colname, VariableBestTransform
from .utils import EngineConfig, expected_sign

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ModelValidationResult:
    """Résultat de validation pour un modèle individuel."""

    model_name: str

    # 4 hard filters
    f1_pd_plausible:     bool = False
    f2_sign_consistency: bool = False
    f3_stress_direction: bool = False
    f4_no_saturation:    bool = False

    # Diagnostics
    pd_fitted_mean: float = np.nan
    pd_fitted_min:  float = np.nan
    pd_fitted_max:  float = np.nan
    sign_correct_ratio:    float = 0.0
    n_stress_scenarios_ok: int = 0
    pd_projected_max: float = np.nan
    pd_projected_min: float = np.nan

    # Soft scores
    s1_stress_direction:    float = 0.0
    s2_stress_spread:       float = 0.0
    s3_scenario_dispersion: float = 0.0
    s4_pd_plausibility:     float = 0.0
    s5_sign_consistency:    float = 0.0
    s6_fit_quality:         float = 0.0

    composite_score: float = 0.0
    rejection_reason: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return (
            self.f1_pd_plausible
            and self.f2_sign_consistency
            and self.f3_stress_direction
            and self.f4_no_saturation
        )


@dataclass
class StressValidationReport:
    validated_models: List[FittedModel] = field(default_factory=list)
    all_results: Dict[str, ModelValidationResult] = field(default_factory=dict)
    n_input: int = 0
    n_validated: int = 0
    rejected_f0: int = 0
    rejected_f1: int = 0
    rejected_f2: int = 0
    rejected_f3: int = 0
    rejected_f4: int = 0

    def summary(self) -> str:
        return (
            f"Stress validator: {self.n_validated}/{self.n_input} models passed. "
            f"Rejected: F0(stat quality)={self.rejected_f0}, "
            f"F1(PD plausible)={self.rejected_f1}, "
            f"F2(signs)={self.rejected_f2}, "
            f"F3(stress direction)={self.rejected_f3}, "
            f"F4(saturation)={self.rejected_f4}."
        )


# ──────────────────────────────────────────────────────────────────────
# F0 — Qualité statistique minimale (nouveau filtre, avant F1)
# ──────────────────────────────────────────────────────────────────────

def _filter_statistical_quality(
    model: FittedModel,
    cfg: EngineConfig,
) -> Tuple[bool, Dict[str, Any]]:
    """
    F0 : Qualité statistique minimale du modèle satellite.

    Critères (cascade, premier échec → rejet) :
      F0a — R² ≥ cfg.min_r2
            Un modèle qui n'explique pas la variance historique ne doit pas
            être projeté, quel que soit son comportement sous stress.
      F0b — Au moins 1 variable (hors constante) avec p-value < cfg.min_pvalue_any
            Garantit qu'au moins un signal statistique existe dans le modèle.
      F0c — Aucun coefficient non-constante exactement nul
            Un coefficient nul = variable économiquement morte dans le modèle.

    Ces critères sont indépendants du comportement sous stress (F1–F4) :
    ils filtrent les modèles statistiquement non-informatifs avant tout.
    """
    diag: Dict[str, Any] = {}

    # F0a — R² minimum
    r2 = model.r2 if not np.isnan(model.r2) else 0.0
    diag["r2"] = r2
    if r2 < cfg.min_r2:
        diag["reason"] = f"R²={r2:.4f} < min_r2={cfg.min_r2}"
        return False, diag

    # F0b — Au moins 1 p-value significative (hors constante)
    pvalues = {k: v for k, v in model.pvalues.items() if k != "const"}
    if pvalues:
        min_pval = float(min(pvalues.values()))
        diag["min_pvalue"] = min_pval
        if min_pval > cfg.min_pvalue_any:
            diag["reason"] = (
                f"Aucune variable significative: min p-value={min_pval:.4f} "
                f"> min_pvalue_any={cfg.min_pvalue_any}"
            )
            return False, diag

    # F0c — Aucun coefficient non-constante exactement nul
    coefs = {k: v for k, v in model.coefficients.items() if k != "const"}
    zero_coefs = [k for k, v in coefs.items() if abs(v) < 1e-10]
    if zero_coefs:
        diag["zero_coefs"] = zero_coefs
        diag["reason"] = f"Coefficients nuls: {zero_coefs}"
        return False, diag

    return True, diag


# ──────────────────────────────────────────────────────────────────────
# Linear predictor + link function
# ──────────────────────────────────────────────────────────────────────

def _apply_link(eta: pd.Series, family: str,
                target_logit_transformed: bool = False) -> pd.Series:
    """Lien inverse : transforme eta en PD selon la famille."""
    if target_logit_transformed:
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
    if family == "OLS":
        return eta
    if family in ("Logit", "Beta"):
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
    if family == "Vasicek-OLS":
        return pd.Series(norm.cdf(eta.values), index=eta.index)
    return eta


def _compute_fitted_pd(model: FittedModel, hist_df: pd.DataFrame,
                       target_logit_transformed: bool = False) -> pd.Series:
    """Reconstruit la PD ajustée sur l'historique."""
    needed_cols = []
    for v in model.variables:
        hist_col, _ = parse_transformed_colname(v)
        needed_cols.append(hist_col)

    available = [c for c in needed_cols if c in hist_df.columns]
    if not available:
        return pd.Series(dtype=float)

    df = hist_df[available].dropna()
    if df.empty:
        return pd.Series(dtype=float)

    coefs = model.coefficients
    intercept = float(coefs.get("const", 0.0))
    eta = pd.Series(intercept, index=df.index, dtype=float)

    for v in model.variables:
        hist_col, _ = parse_transformed_colname(v)
        if hist_col not in df.columns:
            continue
        x = df[hist_col].astype(float)
        if model.scaler and v in model.scaler:
            mean_v, std_v = model.scaler[v]
            x = (x - mean_v) / (std_v if std_v != 0.0 else 1.0)
        eta = eta + float(coefs.get(v, 0.0)) * x

    return _apply_link(eta, model.family, target_logit_transformed)


# ──────────────────────────────────────────────────────────────────────
# 4 hard filters
# ──────────────────────────────────────────────────────────────────────

def _filter_pd_plausibility(model: FittedModel, hist_df: pd.DataFrame,
                             cfg: EngineConfig,
                             target_logit_transformed: bool = False,
                             ) -> Tuple[bool, Dict[str, float]]:
    """F1 : PD ajustée doit être dans [pd_min_plausible, pd_max_plausible]."""
    fitted_pd = _compute_fitted_pd(model, hist_df, target_logit_transformed)
    if fitted_pd.empty:
        return False, {"mean": np.nan, "min": np.nan, "max": np.nan}

    pd_mean = float(fitted_pd.mean())
    pd_min  = float(fitted_pd.min())
    pd_max  = float(fitted_pd.max())
    diag = {"mean": pd_mean, "min": pd_min, "max": pd_max}

    # OLS pur (cible non-bornée) : relâchement
    if model.family == "OLS" and not target_logit_transformed:
        return pd_min >= 0.0, diag

    passed = (
        cfg.pd_min_plausible <= pd_mean <= cfg.pd_max_plausible
        and pd_max <= cfg.pd_max_plausible * 1.5
        and pd_min >= 0.0
    )
    return passed, diag


def _filter_sign_consistency(model: FittedModel,
                              cfg: EngineConfig) -> Tuple[bool, float]:
    """F2 : signes des coefs conformes aux priors économiques."""
    correct = 0
    total = 0

    for var in model.variables:
        hist_col, _ = parse_transformed_colname(var)
        expected = expected_sign(hist_col, cfg.risk_type)
        if expected == 0:
            continue

        beta = model.coefficients.get(var, 0.0)
        if beta == 0.0:
            continue

        actual_sign = 1 if beta > 0 else -1
        if actual_sign == expected:
            correct += 1
        total += 1

    if total == 0:
        return True, 1.0

    ratio = correct / total
    return ratio >= cfg.min_sign_consistency, ratio


def _filter_stress_direction(
    projections_per_scenario: Dict[str, List[Dict[str, Any]]],
    baseline_name: str,
    cfg: EngineConfig,
) -> Tuple[bool, int, Dict[str, float]]:
    """F3 : au moins N scénarios stressent correctement."""
    scenario_means: Dict[str, float] = {}
    for sc, proj in projections_per_scenario.items():
        vals = [p["value"] for p in proj if p.get("value") is not None]
        scenario_means[sc] = float(np.mean(vals)) if vals else np.nan

    base_mean = scenario_means.get(baseline_name, np.nan)
    if np.isnan(base_mean):
        return False, 0, scenario_means

    n_ok = 0
    for sc, mean_v in scenario_means.items():
        if sc == baseline_name or np.isnan(mean_v):
            continue
        if mean_v > base_mean + cfg.min_stress_delta:
            n_ok += 1

    return n_ok >= cfg.min_stress_scenarios, n_ok, scenario_means


def _filter_no_saturation(
    projections_per_scenario: Dict[str, List[Dict[str, Any]]],
    cfg: EngineConfig,
) -> Tuple[bool, float, float]:
    """F4 : PD ne sature pas aux bornes."""
    all_vals: List[float] = []
    for proj in projections_per_scenario.values():
        for p in proj:
            v = p.get("value")
            if v is not None and not np.isnan(v):
                all_vals.append(float(v))

    if not all_vals:
        return False, np.nan, np.nan

    arr = np.array(all_vals)
    pd_min = float(arr.min())
    pd_max = float(arr.max())

    near_zero = pd_min < cfg.pd_saturation_tol
    near_one  = pd_max > (1.0 - cfg.pd_saturation_tol)
    above_max = pd_max > cfg.pd_max_plausible * 2.0

    return (not near_zero) and (not near_one) and (not above_max), pd_min, pd_max


# ──────────────────────────────────────────────────────────────────────
# Soft scores
# ──────────────────────────────────────────────────────────────────────

def _compute_soft_scores(
    fitted_diag: Dict[str, float],
    sign_ratio: float,
    scenario_means: Dict[str, float],
    baseline_name: str,
    model: FittedModel,
    cfg: EngineConfig,
) -> Dict[str, float]:
    """6 soft scores [0, 1]."""
    base_mean = scenario_means.get(baseline_name, np.nan)
    stressed = {k: v for k, v in scenario_means.items()
                if k != baseline_name and not np.isnan(v)}

    # S1 — stress direction
    if stressed and not np.isnan(base_mean):
        delta = np.mean([v - base_mean for v in stressed.values()])
        s1 = float(np.clip(delta / cfg.pd_max_plausible, 0.0, 1.0))
    else:
        s1 = 0.0

    # S2 — stress spread
    if stressed:
        spread = max(stressed.values()) - min(stressed.values())
        s2 = float(np.clip(spread / (cfg.pd_max_plausible * 0.5), 0.0, 1.0))
    else:
        s2 = 0.0

    # S3 — scenario dispersion
    if len(stressed) >= 2:
        std = float(np.std(list(stressed.values()), ddof=1))
        s3 = float(np.clip(std / (cfg.pd_max_plausible * 0.2), 0.0, 1.0))
    else:
        s3 = 0.0

    # S4 — PD plausibility (distance au centre)
    pd_mean_fitted = fitted_diag.get("mean", np.nan)
    if not np.isnan(pd_mean_fitted):
        mid = (cfg.pd_min_plausible + cfg.pd_max_plausible) / 2.0
        half_range = (cfg.pd_max_plausible - cfg.pd_min_plausible) / 2.0
        dist = abs(pd_mean_fitted - mid) / half_range if half_range > 0 else 1.0
        s4 = float(np.clip(1.0 - dist, 0.0, 1.0))
    else:
        s4 = 0.0

    # S5 — sign consistency
    s5 = float(np.clip(sign_ratio, 0.0, 1.0))

    # S6 — fit quality
    r2 = model.r2 if not np.isnan(model.r2) else 0.0
    s6 = float(np.clip(r2, 0.0, 1.0))

    return {
        "s1_stress_direction":    s1,
        "s2_stress_spread":       s2,
        "s3_scenario_dispersion": s3,
        "s4_pd_plausibility":     s4,
        "s5_sign_consistency":    s5,
        "s6_fit_quality":         s6,
    }


def _composite_score(soft_scores: Dict[str, float]) -> float:
    weights = {
        "s1_stress_direction":    0.25,
        "s2_stress_spread":       0.15,
        "s3_scenario_dispersion": 0.15,
        "s4_pd_plausibility":     0.15,
        "s5_sign_consistency":    0.15,
        "s6_fit_quality":         0.15,
    }
    return float(sum(weights[k] * soft_scores.get(k, 0.0) for k in weights))


# ──────────────────────────────────────────────────────────────────────
# Validator principal
# ──────────────────────────────────────────────────────────────────────

def _validate_one_model(
    model: FittedModel,
    hist_df: pd.DataFrame,
    ngfs_country: pd.DataFrame,
    baseline_name: str,
    ngfs_to_hist: Dict[str, str],
    best_per_var: Dict[str, VariableBestTransform],
    cfg: EngineConfig,
    is_ct: bool,
    project_fn_lt: Optional[Callable],
    project_fn_ct: Optional[Callable],
    hist_anchor: Optional[pd.Series],
    target_logit_transformed: bool,
    available_scenarios: List[str],
) -> Tuple[FittedModel, ModelValidationResult, str, Optional[float]]:
    """
    Validate a single model through the F0–F4 cascade + soft scoring.

    Returns (model, result, status, composite_score_or_None).
    Status values: 'rejected_f0' | 'rejected_f1' | 'rejected_f2' |
                   'rejected_f3' | 'rejected_f4' | 'validated'.
    Designed to be called concurrently — reads only shared read-only objects.
    """
    result = ModelValidationResult(model_name=model.name)

    # F0
    f0_ok, f0_diag = _filter_statistical_quality(model, cfg)
    if not f0_ok:
        result.rejection_reason = (
            f"F0: {f0_diag.get('reason', 'qualité statistique insuffisante')}"
        )
        return model, result, "rejected_f0", None

    # F1
    f1_ok, fit_diag = _filter_pd_plausibility(
        model, hist_df, cfg, target_logit_transformed
    )
    result.f1_pd_plausible = f1_ok
    result.pd_fitted_mean = fit_diag["mean"]
    result.pd_fitted_min = fit_diag["min"]
    result.pd_fitted_max = fit_diag["max"]
    if not f1_ok:
        result.rejection_reason = (
            f"F1: PD fitted mean={fit_diag['mean']:.4f} "
            f"out of [{cfg.pd_min_plausible}, {cfg.pd_max_plausible}]"
        )
        return model, result, "rejected_f1", None

    # F2
    f2_ok, sign_ratio = _filter_sign_consistency(model, cfg)
    result.f2_sign_consistency = f2_ok
    result.sign_correct_ratio = sign_ratio
    if not f2_ok:
        result.rejection_reason = (
            f"F2: sign consistency={sign_ratio:.0%} "
            f"< {cfg.min_sign_consistency:.0%}"
        )
        return model, result, "rejected_f2", None

    # Project all scenarios
    projections_per_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for sc in available_scenarios:
        try:
            if is_ct:
                proj = project_fn_ct(
                    model=model, ngfs_country=ngfs_country, scenario_name=sc,
                    ngfs_to_hist=ngfs_to_hist, best_per_var=best_per_var,
                    hist_df=hist_df, hist_year_col=cfg.hist_year_col,
                    target_logit_transformed=target_logit_transformed,
                )
            else:
                proj = project_fn_lt(
                    model=model, ngfs_country=ngfs_country, scenario_name=sc,
                    scenario_channel=cfg.risk_channel,
                    ngfs_to_hist=ngfs_to_hist, best_per_var=best_per_var,
                    hist_anchor=hist_anchor,
                    target_logit_transformed=target_logit_transformed,
                )
            if proj:
                projections_per_scenario[sc] = proj
        except Exception as exc:
            log.debug(
                "Validator: projection failed for '%s' / '%s': %s",
                model.name, sc, exc,
            )

    if len(projections_per_scenario) < 2:
        result.rejection_reason = (
            f"Insufficient projections: only "
            f"{len(projections_per_scenario)} succeeded"
        )
        return model, result, "rejected_f3", None

    # F3
    f3_ok, n_stress_ok, scenario_means = _filter_stress_direction(
        projections_per_scenario, baseline_name, cfg
    )
    result.f3_stress_direction = f3_ok
    result.n_stress_scenarios_ok = n_stress_ok
    if not f3_ok:
        result.rejection_reason = (
            f"F3: only {n_stress_ok} scenarios produce stress "
            f"(needed: {cfg.min_stress_scenarios})"
        )
        return model, result, "rejected_f3", None

    # F4
    f4_ok, pd_proj_min, pd_proj_max = _filter_no_saturation(
        projections_per_scenario, cfg
    )
    result.f4_no_saturation = f4_ok
    result.pd_projected_min = pd_proj_min
    result.pd_projected_max = pd_proj_max
    if not f4_ok:
        result.rejection_reason = (
            f"F4: projected PD saturated "
            f"(min={pd_proj_min:.4f}, max={pd_proj_max:.4f})"
        )
        return model, result, "rejected_f4", None

    # Soft scores + composite
    soft = _compute_soft_scores(
        fitted_diag=fit_diag, sign_ratio=sign_ratio,
        scenario_means=scenario_means, baseline_name=baseline_name,
        model=model, cfg=cfg,
    )
    result.s1_stress_direction    = soft["s1_stress_direction"]
    result.s2_stress_spread       = soft["s2_stress_spread"]
    result.s3_scenario_dispersion = soft["s3_scenario_dispersion"]
    result.s4_pd_plausibility     = soft["s4_pd_plausibility"]
    result.s5_sign_consistency    = soft["s5_sign_consistency"]
    result.s6_fit_quality         = soft["s6_fit_quality"]
    result.composite_score        = _composite_score(soft)

    return model, result, "validated", result.composite_score


def validate_models(
    fitted_models: List[FittedModel],
    hist_df: pd.DataFrame,
    ngfs_country: pd.DataFrame,
    baseline_name: str,
    ngfs_to_hist: Dict[str, str],
    best_per_var: Dict[str, VariableBestTransform],
    cfg: EngineConfig,
    is_ct: bool,
    project_fn_lt: Optional[Callable] = None,
    project_fn_ct: Optional[Callable] = None,
    hist_anchor: Optional[pd.Series] = None,
    target_logit_transformed: bool = False,
) -> StressValidationReport:
    """
    Valide chaque modèle satellite contre 4 hard filters et calcule un
    score composite pour classement.
    """
    report = StressValidationReport(n_input=len(fitted_models))
    validated_with_scores: List[Tuple[FittedModel, float]] = []

    available_scenarios = sorted(ngfs_country["scenario"].dropna().unique().tolist())
    log.info(
        "Stress validator: %d models to validate against %d NGFS scenarios.",
        len(fitted_models), len(available_scenarios),
    )

    # ── Validate all models in parallel (each model is independent) ──────
    converged_models = [m for m in fitted_models if m.converged]
    skipped = len(fitted_models) - len(converged_models)
    if skipped:
        log.debug("%d model(s) skipped (not converged).", skipped)

    n_workers = min(len(converged_models), 8) if converged_models else 1
    futures: Dict[Any, str] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for model in converged_models:
            fut = executor.submit(
                _validate_one_model,
                model, hist_df, ngfs_country, baseline_name, ngfs_to_hist,
                best_per_var, cfg, is_ct, project_fn_lt, project_fn_ct,
                hist_anchor, target_logit_transformed, available_scenarios,
            )
            futures[fut] = model.name

    for fut in as_completed(futures):
        try:
            model, result, status, score = fut.result()
        except Exception as exc:
            log.error(
                "Validator: unexpected error for '%s': %s",
                futures[fut], exc,
            )
            continue

        report.all_results[model.name] = result

        if status == "rejected_f0":
            report.rejected_f0 += 1
            log.info("  ✗ '%s' [F0]: %s", model.name, result.rejection_reason)
        elif status == "rejected_f1":
            report.rejected_f1 += 1
            log.info("  ✗ '%s' [F1]: %s", model.name, result.rejection_reason)
        elif status == "rejected_f2":
            report.rejected_f2 += 1
            log.info("  ✗ '%s' [F2]: %s", model.name, result.rejection_reason)
        elif status == "rejected_f3":
            report.rejected_f3 += 1
            log.info("  ✗ '%s': %s", model.name, result.rejection_reason)
        elif status == "rejected_f4":
            report.rejected_f4 += 1
            log.info("  ✗ '%s' [F4]: %s", model.name, result.rejection_reason)
        elif status == "validated":
            validated_with_scores.append((model, score))
            log.info(
                "  ✓ '%s': PASSED — composite=%.3f "
                "(S1=%.2f S2=%.2f S3=%.2f S4=%.2f S5=%.2f S6=%.2f)",
                model.name, score,
                result.s1_stress_direction, result.s2_stress_spread,
                result.s3_scenario_dispersion, result.s4_pd_plausibility,
                result.s5_sign_consistency, result.s6_fit_quality,
            )

    # Trier par score composite décroissant
    validated_with_scores.sort(key=lambda x: x[1], reverse=True)
    report.validated_models = [m for m, _ in validated_with_scores]
    report.n_validated = len(report.validated_models)

    log.info(report.summary())
    return report


def get_stress_scores_dict(report: StressValidationReport) -> Dict[str, float]:
    """{model_name: composite_score} pour les modèles validés."""
    return {
        name: res.composite_score
        for name, res in report.all_results.items()
        if res.is_valid
    }
