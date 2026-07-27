"""
baseline_projector.py
=====================
Projette les variables macro sur l'horizon Baseline en utilisant
UNIQUEMENT les données historiques — aucune donnée NGFS.

Architecture (par variable sélectionnée en Stage 2) :
  1. ADF test        → décision stationnarité
  2. AIC sur p=0..4  → ordre AR optimal
  3. d minimal       → si non-stationnaire → ARIMA(p, d, 0)
  4. VAR             → si n_obs > 4 × n_vars et allow_var=True
  5. Random Walk     → fallback ultime (< MIN_OBS_FOR_MODEL obs propres)

Hiérarchie fallback par variable :
    VAR (si éligible) → AR(p) → ARIMA(p, d, 0) → Random Walk

Les projections (niveaux) sont ensuite converties en DataFrame
et consommées par multi_scenario.py qui rejoue les mêmes transformations
(lag1_pct, diff, etc.) avant d'alimenter le modèle satellite.

Générique : fonctionne pour n'importe quel pays, risque, dataset.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.api import VAR as VARModel
    _SM_OK = True
except ImportError:
    _SM_OK = False

try:
    from statsmodels.tsa.stattools import kpss as _kpss_fn
    _KPSS_OK = True
except ImportError:
    _KPSS_OK = False

from .utils import get_logger

log = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
MAX_AR_ORDER      = 4      # p testé : 0, 1, 2, 3, 4
VAR_OBS_PER_VAR   = 4      # seuil : n_obs > VAR_OBS_PER_VAR × n_vars
ADF_PVAL_THRESH   = 0.05   # seuil stationnarité ADF
MAX_DIFF_ORDER    = 2      # différenciation max
MIN_OBS_FOR_MODEL = 5      # en-dessous → random walk
VAR_STABILITY_MARGIN    = 0.95   # module max des racines inverses VAR accepté
MIN_OBS_FOR_VAR_STABILITY = 20   # en-dessous + I(1+) → VECM peu fiable → AR univarié
KPSS_PVAL_THRESH      = 0.05   # seuil KPSS (H0 = stationnaire ; rejet si p < seuil)
# 90% des points de l'horizon à la borne = effondrement quasi-total de la
# trajectoire, distinct d'un cas limite ponctuel.
F4_COLLAPSE_RATIO     = 0.90


# ──────────────────────────────────────────────────────────────────────────────
# Containers de résultats
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VariableProjection:
    """Résultat de projection pour une variable macro."""
    variable:          str
    model_type:        str                    # "VAR" | "AR" | "ARIMA" | "RandomWalk"
    model_order:       Tuple                  # ex. (2,) pour AR(2), (1,1,0) pour ARIMA
    is_stationary:     bool
    adf_pvalue:        float
    projected_levels:  Dict[int, float]       # {année: valeur en niveau}
    notes:             str = ""


@dataclass
class BaselineProjection:
    """Résultat complet passé à multi_scenario."""
    variable_projections: Dict[str, VariableProjection] = field(default_factory=dict)
    model_used:           str = ""            # "VAR" | "univariate"
    var_variables:        List[str] = field(default_factory=list)
    horizon_years:        List[int] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers statistiques
# ──────────────────────────────────────────────────────────────────────────────

def _adf(series: np.ndarray) -> Tuple[bool, float]:
    """ADF test. Retourne (is_stationary, p_value)."""
    if not _SM_OK:
        return False, 1.0
    clean = series[~np.isnan(series)]
    if len(clean) < 5:
        return False, 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = adfuller(clean, autolag="AIC")
        pval = float(res[1])
        return pval < ADF_PVAL_THRESH, pval
    except Exception as e:
        log.debug(f"ADF failed: {e}")
        return False, 1.0


def _best_ar_order(series: np.ndarray, max_p: int = MAX_AR_ORDER) -> int:
    """Sélectionne l'ordre AR par minimisation AIC sur p = 0..max_p."""
    if not _SM_OK:
        return 1
    clean = series[~np.isnan(series)]
    if len(clean) < max_p + 5:
        return min(1, len(clean) - 2)
    best_p, best_aic = 1, np.inf
    for p in range(0, max_p + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = AutoReg(clean, lags=p, old_names=False).fit()
            if m.aic < best_aic:
                best_aic, best_p = m.aic, p
        except Exception:
            continue
    return best_p if best_p > 0 else 1


def _min_diff_order(series: np.ndarray) -> int:
    """Itère la différenciation jusqu'à stationnarité → retourne d minimal."""
    cur = series[~np.isnan(series)].copy()
    for d in range(MAX_DIFF_ORDER + 1):
        is_stat, _ = _adf(cur)
        if is_stat:
            return d
        cur = np.diff(cur)
    return MAX_DIFF_ORDER


def _forecast_ar(series: np.ndarray, p: int, h: int) -> np.ndarray:
    """AR(p) forecast h steps. Fallback: random walk."""
    clean = series[~np.isnan(series)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = AutoReg(clean, lags=p, old_names=False).fit()
        return np.array(m.forecast(h))
    except Exception as e:
        log.debug(f"AR({p}) failed: {e}")
        return np.full(h, clean[-1])


def _forecast_arima(series: np.ndarray, d: int, h: int) -> np.ndarray:
    """ARIMA(p_opt, d, 0) forecast h steps. Fallback: random walk."""
    clean = series[~np.isnan(series)]
    diff_s = np.diff(clean, n=d) if d > 0 else clean
    p = _best_ar_order(diff_s)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ARIMA(clean, order=(p, d, 0)).fit()
        return np.array(m.forecast(steps=h))
    except Exception as e:
        log.debug(f"ARIMA({p},{d},0) failed: {e}")
        return np.full(h, clean[-1])


def _random_walk(series: np.ndarray, h: int) -> np.ndarray:
    """Dernière valeur observée projetée en plateau (fallback ultime)."""
    clean = series[~np.isnan(series)]
    last = float(clean[-1]) if len(clean) > 0 else 0.0
    return np.full(h, last)


def _kpss_test(series: np.ndarray) -> bool:
    """KPSS test. Retourne True si stationnaire (non-rejet H0 : stationnaire)."""
    if not _KPSS_OK:
        return True  # pas de test possible → conservatif
    clean = series[~np.isnan(series)]
    if len(clean) < 5:
        return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, pval, _, _ = _kpss_fn(clean, regression="c", nlags="auto")
        return pval > KPSS_PVAL_THRESH  # non-rejet H0 → stationnaire
    except Exception:
        return True


def _check_f4_collapse(fc: np.ndarray, bounds: tuple) -> bool:
    """F4 : True si ≥ F4_COLLAPSE_RATIO des points projetés sont à une borne."""
    lo, hi = bounds
    at_bound = np.sum(
        (np.abs(fc - lo) < 1e-9) | (np.abs(fc - hi) < 1e-9)
    )
    return (at_bound / max(len(fc), 1)) >= F4_COLLAPSE_RATIO


# ──────────────────────────────────────────────────────────────────────────────
# VAR
# ──────────────────────────────────────────────────────────────────────────────

def _integration_orders(hist_df: pd.DataFrame, variables: List[str]) -> Dict[str, int]:
    """
    Retourne l'ordre d'intégration estimé pour chaque variable :
      0 → I(0) stationnaire en niveaux
      1 → I(1) stationnaire en différences premières
      2 → I(2) (rare)

    Utilisé pour valider la compatibilité d'un VAR en niveaux.
    """
    orders: Dict[str, int] = {}
    for var in variables:
        series = hist_df[var].dropna().values.astype(float)
        for d in range(MAX_DIFF_ORDER + 1):
            is_stat, _ = _adf(series if d == 0 else np.diff(series, n=d))
            if is_stat:
                orders[var] = d
                break
        else:
            orders[var] = MAX_DIFF_ORDER
    return orders


def _check_var_stability(fitted_var) -> bool:
    """
    Vérifie que toutes les racines inverses du VAR sont dans le cercle unité
    (module strict < 1.0 avec marge de sécurité VAR_STABILITY_MARGIN).
    Un VAR instable produit des projections explosives ou oscillantes.
    """
    try:
        roots = fitted_var.roots          # modules des racines inverses
        max_root = float(np.max(np.abs(roots)))
        if max_root >= VAR_STABILITY_MARGIN:
            log.warning(
                f"VAR instable: racine max={max_root:.4f} ≥ {VAR_STABILITY_MARGIN} "
                f"(seuil de stabilité) → rejeté."
            )
            return False
        return True
    except Exception as e:
        log.debug(f"Vérification stabilité VAR impossible: {e}")
        return True  # pas de rejet par défaut si test échoue


def _try_var(
    hist_df: pd.DataFrame,
    variables: List[str],
    horizon: int,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Estime un VAR en niveaux sur `variables` et projette `horizon` steps.
    Retourne {var: forecast_array} ou None si :
      - seuil d'observations non atteint
      - ordres d'intégration mixtes (I(0)/I(1+) mélangés) → AR univarié préféré
      - petit échantillon < MIN_OBS_FOR_VAR_STABILITY avec variables I(1+)
      - VAR instable (racine hors cercle unité)
      - exception statsmodels

    Architecture décisionnelle :
      Toutes I(0)            → VAR en niveaux si stable
      Toutes I(1), n≥20      → VAR en niveaux (cointégration plausible)
      Toutes I(1), n<20      → AR univarié (VECM peu fiable sur petit échantillon)
      Ordres mixtes I(0)/I(1+) → AR univarié (VAR en niveaux invalide)
      Racine VAR ≥ marge      → AR univarié
    """
    if not _SM_OK:
        return None
    data = hist_df[variables].dropna()
    n = len(data)
    threshold = VAR_OBS_PER_VAR * len(variables)
    if n < threshold:
        log.info(f"VAR rejeté: n_obs={n} < {threshold} (={VAR_OBS_PER_VAR}×{len(variables)} vars)")
        return None

    # ── Vérification des ordres d'intégration ────────────────────────────────
    orders = _integration_orders(hist_df, variables)
    unique_orders = set(orders.values())
    order_summary = ", ".join(f"{v}=I({d})" for v, d in orders.items())

    if len(unique_orders) > 1:
        # Ordres mixtes : VAR en niveaux invalide → AR univarié
        log.warning(
            f"VAR rejeté: ordres d'intégration mixtes [{order_summary}]. "
            f"Un VAR en niveaux n'est valide que si toutes les variables "
            f"ont le même ordre. → Fallback AR univarié par variable."
        )
        return None

    max_order = max(unique_orders)
    if max_order >= 1 and n < MIN_OBS_FOR_VAR_STABILITY:
        # Variables I(1+) sur petit échantillon : VECM peu fiable → AR univarié
        log.warning(
            f"VAR rejeté: variables I({max_order}) mais n_obs={n} < "
            f"{MIN_OBS_FOR_VAR_STABILITY} (seuil VECM petit échantillon). "
            f"→ Fallback AR univarié par variable."
        )
        return None

    log.info(f"  Ordres d'intégration: {order_summary}")

    maxlags = max(1, min(4, n // 5))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = VARModel(data).fit(maxlags=maxlags, ic="aic", verbose=False)

        # ── Vérification stabilité racines VAR ───────────────────────────────
        if not _check_var_stability(fitted):
            return None

        k = fitted.k_ar
        fc = fitted.forecast(data.values[-k:], steps=horizon)  # (horizon, n_vars)
        return {col: fc[:, i] for i, col in enumerate(variables)}
    except Exception as e:
        log.warning(f"VAR échoué: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ──────────────────────────────────────────────────────────────────────────────

def project_baseline(
    hist_df: pd.DataFrame,
    selected_hist_vars: List[str],
    horizon_years: List[int],
    hist_year_col: str = "year",
    allow_var: bool = True,
    variable_config: Optional[Dict] = None,
) -> BaselineProjection:
    """
    Projette chaque variable macro sélectionnée sur `horizon_years`
    en utilisant le meilleur modèle time-series disponible.

    Parameters
    ----------
    hist_df           : DataFrame historique (colonne année + variables macro).
    selected_hist_vars: noms des colonnes historiques à projeter.
    horizon_years     : années cibles, ex. list(range(2022, 2051)).
    hist_year_col     : nom de la colonne année dans hist_df.
    allow_var         : si True, tente VAR avant de passer en univarié.
    variable_config   : dict optionnel {var: {bounds, min_obs,
                        conflict_resolution}}. Si None, comportement
                        strictement identique à l'original (non-régression
                        climat garantie).

    Returns
    -------
    BaselineProjection contenant les niveaux projetés par variable.
    """
    if not _SM_OK:
        raise ImportError(
            "statsmodels requis pour baseline_projector. "
            "Installer avec: pip install statsmodels"
        )

    # ── Aligner l'index sur l'année ─────────────────────────────────────────
    df = hist_df.copy()
    if hist_year_col in df.columns:
        df = df.set_index(hist_year_col)
    df.index = df.index.astype(int)
    df = df.sort_index()

    # Ne garder que les variables disponibles
    available = [v for v in selected_hist_vars if v in df.columns]
    missing   = set(selected_hist_vars) - set(available)
    if missing:
        log.warning(f"Variables absentes de l'historique: {missing}")

    horizon = len(horizon_years)
    result  = BaselineProjection(horizon_years=horizon_years)

    if not available:
        log.error("Aucune variable disponible pour la projection Baseline.")
        return result

    n_obs = len(df.dropna(subset=available, how="all"))
    log.info(
        f"Baseline projector: {n_obs} obs | {len(available)} vars | "
        f"horizon {horizon_years[0]}–{horizon_years[-1]}"
    )

    # ── Tentative VAR ────────────────────────────────────────────────────────
    var_forecasts: Dict[str, np.ndarray] = {}
    var_succeeded = False

    if allow_var and len(available) >= 2:
        # Limiter les variables VAR au seuil (prendre les plus corrélées si trop)
        max_var_vars = max(2, n_obs // VAR_OBS_PER_VAR)
        var_candidates = available[:max_var_vars]
        log.info(
            f"Tentative VAR sur {len(var_candidates)} variables "
            f"(seuil: {n_obs} obs ≥ {VAR_OBS_PER_VAR}×{len(var_candidates)})"
        )
        var_forecasts = _try_var(df, var_candidates, horizon) or {}
        if var_forecasts:
            var_succeeded = True
            result.model_used    = "VAR"
            result.var_variables = list(var_forecasts.keys())
            log.info(f"VAR estimé sur: {result.var_variables}")

    if not var_succeeded:
        result.model_used = "univariate"

    # ── Projection par variable ──────────────────────────────────────────────
    for var in available:
        series = df[var].values.astype(float)
        clean  = series[~np.isnan(series)]

        is_stat, adf_pval = _adf(series)

        # Config optionnelle par variable (uniquement si variable_config fourni)
        _vcfg   = variable_config.get(var, {}) if variable_config is not None else {}
        _bounds = _vcfg.get("bounds") if _vcfg else None
        _cr     = _vcfg.get("conflict_resolution") if _vcfg else None

        # ── VAR disponible pour cette variable ───────────────────────────────
        if var_succeeded and var in var_forecasts:
            fc = var_forecasts[var]
            proj = VariableProjection(
                variable=var, model_type="VAR",
                model_order=(len(result.var_variables),),
                is_stationary=is_stat, adf_pvalue=adf_pval,
                projected_levels={yr: float(v) for yr, v in zip(horizon_years, fc)},
                notes=f"VAR joint ({len(result.var_variables)} vars)",
            )
            result.variable_projections[var] = proj
            log.info(f"  {var:<35} VAR    | ADF p={adf_pval:.3f}")
            continue

        # ── Fallback univarié ────────────────────────────────────────────────
        if len(clean) < MIN_OBS_FOR_MODEL:
            fc = _random_walk(series, horizon)
            proj = VariableProjection(
                variable=var, model_type="RandomWalk", model_order=(0,),
                is_stationary=False, adf_pvalue=adf_pval,
                projected_levels={yr: float(v) for yr, v in zip(horizon_years, fc)},
                notes=f"Trop peu d'obs ({len(clean)}) → random walk",
            )
            result.variable_projections[var] = proj
            log.warning(f"  {var:<35} RandomWalk ({len(clean)} obs)")
            continue

        # ── Consensus stationnarité ADF + KPSS (si variable_config fourni) ──
        # Quand variable_config est None, ce bloc est entièrement sauté →
        # comportement strictement identique à l'original (non-régression).
        if variable_config is not None:
            _is_stat_kpss = _kpss_test(series)
            if is_stat != _is_stat_kpss:
                # Conflit ADF / KPSS → conflict_resolution
                _cr_method = _cr or "AR(1)"
                fc_cr = _forecast_ar(series, 1, horizon)
                _notes_cr = (
                    f"AR(1) [conflit ADF(stat={is_stat}, p={adf_pval:.3f}) / "
                    f"KPSS(stat={_is_stat_kpss})] "
                    f"conflict_resolution={_cr_method}"
                )
                log.warning(
                    "  %-35s ADF/KPSS conflit → %s", var, _cr_method
                )
                _mtype, _mord = "AR", (1,)
                if _bounds is not None:
                    fc_cr = np.clip(fc_cr, _bounds[0], _bounds[1])
                    if _check_f4_collapse(fc_cr, _bounds):
                        fc_cr  = _random_walk(series, horizon)
                        _notes_cr += (
                            f" → F4 collapse "
                            f"({F4_COLLAPSE_RATIO:.0%} à borne) → RandomWalk"
                        )
                        _mtype, _mord = "RandomWalk", (0,)
                        log.warning(
                            "  %-35s F4 collapse → RandomWalk", var
                        )
                proj = VariableProjection(
                    variable=var, model_type=_mtype, model_order=_mord,
                    is_stationary=is_stat, adf_pvalue=adf_pval,
                    projected_levels={
                        yr: float(v) for yr, v in zip(horizon_years, fc_cr)
                    },
                    notes=_notes_cr,
                )
                result.variable_projections[var] = proj
                continue

        # ── Modèle univarié standard ─────────────────────────────────────────
        if is_stat:
            p  = _best_ar_order(series)
            fc = _forecast_ar(series, p, horizon)
            model_type  = "AR"
            model_order = (p,)
            notes       = f"AR({p}) | ADF p={adf_pval:.3f} → stationnaire"
            log.info(f"  {var:<35} AR({p})  | ADF p={adf_pval:.3f}")
        else:
            d  = _min_diff_order(series)
            fc = _forecast_arima(series, d, horizon)
            diff_s = np.diff(series[~np.isnan(series)], n=d) if d > 0 else series
            p_diff = _best_ar_order(diff_s)
            model_type  = "ARIMA"
            model_order = (p_diff, d, 0)
            notes       = (
                f"ARIMA({p_diff},{d},0) | ADF p={adf_pval:.3f} → non-stationnaire"
            )
            log.info(f"  {var:<35} ARIMA({p_diff},{d},0) | ADF p={adf_pval:.3f}")

        # ── Clipping F1 + F4 collapse (si variable_config fourni) ───────────
        if variable_config is not None and _bounds is not None:
            fc = np.clip(fc, _bounds[0], _bounds[1])
            if _check_f4_collapse(fc, _bounds):
                fc          = _random_walk(series, horizon)
                notes      += (
                    f" → F4 collapse ({F4_COLLAPSE_RATIO:.0%} à borne) → RandomWalk"
                )
                model_type  = "RandomWalk"
                model_order = (0,)
                log.warning(
                    "  %-35s F4 collapse → RandomWalk [remplace %s]",
                    var, "AR/ARIMA",
                )

        proj = VariableProjection(
            variable=var, model_type=model_type, model_order=model_order,
            is_stationary=is_stat, adf_pvalue=adf_pval,
            projected_levels={yr: float(v) for yr, v in zip(horizon_years, fc)},
            notes=notes,
        )
        result.variable_projections[var] = proj

    log.info(
        f"Projection Baseline complète: {len(result.variable_projections)} variables "
        f"| modèle global: {result.model_used}"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Helpers pour multi_scenario.py
# ──────────────────────────────────────────────────────────────────────────────

def build_baseline_macro_df(projection: BaselineProjection) -> pd.DataFrame:
    """
    Convertit un BaselineProjection en DataFrame (n_years × n_vars)
    indexé par année — format attendu par la couche de replay des transforms
    et le modèle satellite.
    """
    records = {
        var: vp.projected_levels
        for var, vp in projection.variable_projections.items()
    }
    df = pd.DataFrame(records)
    df.index.name = "year"
    return df.sort_index()


def summarize_projection(projection: BaselineProjection) -> str:
    """Résumé lisible des choix de modèles — utilisé dans les logs/rapports."""
    lines = [
        "Baseline Projection Summary",
        f"  Modèle global   : {projection.model_used}",
        f"  Horizon         : {projection.horizon_years[0]}–{projection.horizon_years[-1]}",
        f"  Variables ({len(projection.variable_projections)}) :",
    ]
    for var, vp in projection.variable_projections.items():
        stat = "I(0)" if vp.is_stationary else "I(1+)"
        lines.append(
            f"    {var:<35} {vp.model_type:<10} ordre={vp.model_order}  "
            f"{stat}  ADF_p={vp.adf_pvalue:.3f}"
        )
    return "\n".join(lines)


def summarize_fallbacks(projection: BaselineProjection) -> str:
    """
    Résumé des variables tombées en Random Walk (fallback ultime).

    Si 4 ou 5 variables sur 5 sont en RandomWalk, c'est le signe d'un
    problème en amont (mauvaise transformation, bounds mal calibrés)
    à investiguer avant de considérer le chantier terminé.
    """
    rw = [
        (var, vp)
        for var, vp in projection.variable_projections.items()
        if vp.model_type == "RandomWalk"
    ]
    total = len(projection.variable_projections)
    n_rw  = len(rw)
    lines = [f"Fallbacks RandomWalk : {n_rw}/{total} variables"]
    for var, vp in rw:
        lines.append(f"  {var:<35} → {vp.notes}")
    if n_rw == 0:
        lines.append("  (aucun fallback)")
    return "\n".join(lines)
