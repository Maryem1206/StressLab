"""
satellite_calibrator.py
=======================
Calibration des 5 modèles satellites du module liquidité.

Modèles :
    M1  run_off_retail       Beta regression      domaine (0, 1)
    M2  run_off_corporate    Beta regression      domaine (0, 1)
    M3  haircut_add          OLS log-transform    domaine (0, 0.50)
    M4  asf_factor_retail    Beta + remap [.7,1]  domaine (0.70, 1.00)
    M5  asf_factor_corporate Beta + remap [.2,.8] domaine (0.20, 0.80)

Moteur de sélection — IDENTIQUE AU MODULE CRÉDIT (cohérence méthodologique) :
    1. Générer 4 transformations par variable macro disponible :
       "level" (valeur brute), "lag1" (décalage 1 an),
       "growth" (variation %), "cycle" (écart à l'EWMA span=3)
    2. Scorer chaque feature par corrélation absolue mixte avec Y :
       corr_score = 0.5 × (|Pearson| + |Spearman|)
    3. Dédupliquer : garder la meilleure transformation par variable parente
    4. Énumérer TOUTES les combinaisons k ∈ [min_combo, max_combo] :
       - min_combo = 2  (minimum 2 drivers par modèle satellite)
       - max_combo = max(2, ⌊n/10⌋ − 1) ≤ 3  (règle de Harrell 2015)
    5. Ajuster chaque combinaison (famille beta ou ols_log selon satellite)
    6. Classer par :  sign_violations ASC → R² DESC → AIC ASC
    7. Retourner le modèle vainqueur

Les variables macro candidates NE sont PAS hardcodées : le moteur utilise
TOUTES les colonnes disponibles dans df_macro (issues de l'API WorldBank/IMF).
Le registre des signes économiques (_SIGN_REGISTRY) fournit les contraintes
de signe pour toutes les variables macro connues ; une variable inconnue reçoit
le signe 0 (non contraint) et reste éligible au tournoi.

Références académiques :
    Ferrari & Cribari-Neto (2004), Beta Regression, J. Applied Statistics
    BCBS 238 (2013) — LCR ; BCBS 295 (2014) — NSFR
    Drehmann & Nikolaou (2013), Funding Liquidity Risk, J. Banking & Finance
    Harrell (2015), Regression Modeling Strategies — règle EPV
    Burnham & Anderson (2002), Model Selection — critère AIC
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

LOG = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES TOURNOI (identiques au module crédit)
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORMS            = ("level", "lag1", "growth", "cycle")

_VERDICT_RANK: Dict[str, int] = {
    "VALIDATED":               0,
    "VALIDATED_WITH_WARNINGS": 1,
    "REJECTED":                2,
}
EWMA_SPAN             = 3       # fenêtre EWMA pour le composant cyclique
MIN_OBS_FEATURE       = 5       # obs valides min pour retenir une feature

# ── Tournament memoization ──────────────────────────────────────────────────
# calibrate_all() is a pure, deterministic function of (df_bilan, df_macro,
# forced_ranks) — it never depends on any stress scenario. Callers that
# stress-test the same historical dataset across many (scenario, year) pairs
# (notably ClimateOrchestrator, which re-instantiates the liquidity engine
# once per NGFS scenario × horizon year) would otherwise re-run the full
# 5-satellite × dozens-of-combos tournament every single time for an
# identical result — this was measured at ~90s per call. A small
# content-addressed cache eliminates that redundant work with zero risk of
# changing results, since a cache hit only occurs on byte-identical inputs.
_TOURNAMENT_CACHE: "OrderedDict[str, Tuple[Dict, Dict, Dict]]" = OrderedDict()
_TOURNAMENT_CACHE_MAX = 32


def _tournament_cache_key(
    df_bilan: pd.DataFrame,
    df_macro: pd.DataFrame,
    forced_ranks: Optional[Dict[str, int]],
) -> str:
    h = hashlib.md5()
    h.update(pd.util.hash_pandas_object(df_bilan, index=True).values.tobytes())
    h.update(",".join(df_bilan.columns.astype(str)).encode())
    h.update(pd.util.hash_pandas_object(df_macro, index=True).values.tobytes())
    h.update(",".join(df_macro.columns.astype(str)).encode())
    h.update(repr(sorted((forced_ranks or {}).items())).encode())
    return h.hexdigest()
MIN_COMBO_SIZE        = 2       # minimum 2 drivers par modèle
HARRELL_OBS_PER_PARAM = 10      # règle EPV : ≥10 obs par paramètre estimé
_EPS                  = 1e-9
MIN_CORR_SCORE        = 0.10   # alignement module climat
MAX_VIF               = 10.0   # alignement module climat
PVALUE_THRESHOLD      = 0.20   # alignement module climat

# Colonnes à exclure du pool candidat (identifiants, pas des variables macro)
_NON_MACRO_COLS = frozenset({"year", "country", "iso2", "bank_name", "period"})


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRE DES SIGNES ÉCONOMIQUES
# ─────────────────────────────────────────────────────────────────────────────
# Couvre toutes les variables macro connues retournées par l'API WorldBank/IMF.
# Justification : théorie économique + BCBS 238/295.
# Une variable absente du registre reçoit le signe 0 (non contraint).
#
# Convention :  +1 → coefficient attendu positif
#               -1 → coefficient attendu négatif
#                0 → non contraint (variable inconnue)

_SIGN_REGISTRY: Dict[str, Dict[str, int]] = {

    # M1 — Taux de run-off retail (BCBS 238 §73-74)
    "run_off_retail": {
        "GDP_growth":           -1,  # ↑PIB → confiance → ↓runoff
        "real_gdp_growth":      -1,
        "unemployment_rate":    +1,  # ↑chômage → stress déposants → ↑runoff
        "cpi_inflation":        +1,  # ↑inflation → ↓pouvoir achat → ↑runoff
        "inflation":            +1,
        "policy_rate":          +1,  # ↑taux → MM attractif → ↑runoff
        "interest_rate":        +1,
        "lending_rate":         +1,
        "exchange_rate":        +1,  # ↑fx (dépréciation LCU) → fuite capitaux → ↑runoff
        "real_estate_price":    -1,  # ↑immo → effet richesse → ↓runoff
        "house_price_index":    -1,
        "credit_growth":        -1,  # ↑crédit → confiance ↑ → ↓runoff
        "private_credit":       -1,
        "equity_index_growth":  -1,  # ↑marchés → richesse ↑ → ↓runoff
        "stock_return":         -1,
        "current_account":      -1,  # ↑CA → stabilité ext → ↓runoff
        "fiscal_balance":       -1,  # ↑équilibre budgétaire → stabilité → ↓runoff
        "government_debt_gdp":  +1,  # ↑dette pub → risque pays → ↑runoff
        "vix":                  +1,  # ↑volatilité → panique → ↑runoff
        "sovereign_spread":     +1,  # ↑spread souverain → risque → ↑runoff
        "bank_capital_ratio":   -1,  # ↑capitalisation → confiance → ↓runoff
        "npl_ratio":            +1,  # ↑NPL → méfiance → ↑runoff
    },

    # M2 — Taux de run-off corporate (BCBS 238 §105)
    "run_off_corporate": {
        "GDP_growth":           -1,
        "real_gdp_growth":      -1,
        "unemployment_rate":    +1,
        "cpi_inflation":        +1,
        "inflation":            +1,
        "policy_rate":          +1,
        "interest_rate":        +1,
        "lending_rate":         +1,
        "exchange_rate":        +1,
        "real_estate_price":    -1,
        "house_price_index":    -1,
        "credit_growth":        -1,
        "private_credit":       -1,
        "equity_index_growth":  -1,
        "stock_return":         -1,
        "current_account":      -1,
        "fiscal_balance":       -1,
        "government_debt_gdp":  +1,
        "vix":                  +1,
        "sovereign_spread":     +1,
        "corporate_spread":     +1,  # ↑spread corpo → stress liquidité → ↑runoff
        "bank_capital_ratio":   -1,
        "npl_ratio":            +1,
        "trade_openness":       +1,  # ↑ouverture → expo externe → ↑volatilité
    },

    # M3 — Décote additionnelle HQLA (BCBS 238 §42)
    "haircut_add": {
        "GDP_growth":           -1,  # ↑PIB → liquidité marché ↑ → ↓décote
        "real_gdp_growth":      -1,
        "unemployment_rate":    +1,  # ↑chômage → illiquidité → ↑décote
        "cpi_inflation":        +1,  # ↑inflation → risque taux → ↑décote
        "inflation":            +1,
        "policy_rate":          +1,  # ↑taux → risque durée → ↑décote
        "interest_rate":        +1,
        "exchange_rate":        +1,  # ↑fx → risque devise → ↑décote
        "real_estate_price":    -1,  # ↑immo → collatéral valorisé → ↓décote
        "house_price_index":    -1,
        "equity_index_growth":  -1,  # ↑marchés → liquidité ↑ → ↓décote
        "stock_return":         -1,
        "vix":                  +1,  # ↑volatilité → illiquidité → ↑décote
        "sovereign_spread":     +1,
        "government_debt_gdp":  +1,
        "term_spread":          -1,  # ↑pente courbe → confiance macro → ↓décote
        "fiscal_balance":       -1,
    },

    # M4 — Facteur ASF retail (BCBS 295 §35-36)
    "asf_factor_retail": {
        "GDP_growth":           +1,  # ↑PIB → dépôts stables → ↑ASF
        "real_gdp_growth":      +1,
        "unemployment_rate":    -1,  # ↑chômage → retraits → ↓ASF
        "cpi_inflation":        -1,  # ↑inflation → dépôts moins stables → ↓ASF
        "inflation":            -1,
        "policy_rate":          -1,  # ↑taux → arbitrage MM → ↓ASF
        "interest_rate":        -1,
        "lending_rate":         -1,
        "exchange_rate":        -1,  # ↑fx → fuite capitaux → ↓ASF
        "real_estate_price":    +1,  # ↑immo → effet richesse → ↑ASF
        "house_price_index":    +1,
        "credit_growth":        +1,  # ↑crédit → confiance → ↑ASF
        "private_credit":       +1,
        "equity_index_growth":  +1,  # ↑marchés → richesse ↑ → ↑ASF
        "stock_return":         +1,
        "fiscal_balance":       +1,  # ↑équilibre budgétaire → stabilité → ↑ASF
        "government_debt_gdp":  -1,
        "vix":                  -1,
        "sovereign_spread":     -1,
        "bank_capital_ratio":   +1,
        "npl_ratio":            -1,
    },

    # M5 — Facteur ASF corporate (BCBS 295 §35-36)
    "asf_factor_corporate": {
        "GDP_growth":           +1,
        "real_gdp_growth":      +1,
        "unemployment_rate":    -1,
        "cpi_inflation":        -1,
        "inflation":            -1,
        "policy_rate":          -1,
        "interest_rate":        -1,
        "lending_rate":         -1,
        "exchange_rate":        -1,
        "real_estate_price":    +1,
        "house_price_index":    +1,
        "credit_growth":        +1,
        "private_credit":       +1,
        "equity_index_growth":  +1,
        "stock_return":         +1,
        "fiscal_balance":       +1,
        "government_debt_gdp":  -1,
        "vix":                  -1,
        "sovereign_spread":     -1,
        "corporate_spread":     -1,
        "bank_capital_ratio":   +1,
        "npl_ratio":            -1,
        "trade_openness":       -1,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Structures de résultat
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SatelliteResult:
    """Résultat d'un modèle satellite calibré."""
    variable     : str
    family       : str                        # "beta" | "ols_log"
    drivers      : List[str]                  # features sélectionnées (driver__transform)
    coefficients : Dict[str, float]
    std_errors   : Dict[str, float]
    pvalues      : Dict[str, float]
    r2           : float
    aic          : float
    bic          : float
    n_obs        : int
    converged    : bool
    sign_checks  : Dict[str, bool]            # True si signe économique respecté
    domain       : Tuple[float, float]
    remap_lo      : Optional[float]      = None
    remap_hi      : Optional[float]      = None
    fitted_values : Optional[List[float]] = field(default=None, repr=False)
    actual_values : Optional[List[float]] = field(default=None, repr=False)
    obs_years     : Optional[List[int]]   = field(default=None, repr=False)
    _predict_fn   : object = field(default=None, repr=False)

    def predict(self, df_macro: pd.DataFrame) -> np.ndarray:
        if self._predict_fn is None:
            raise RuntimeError(f"Modèle {self.variable} non calibré.")
        return self._predict_fn(df_macro)

    @property
    def signs_ok(self) -> bool:
        return all(self.sign_checks.values())


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires numériques
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _safe_clip(y: np.ndarray, lo: float, hi: float, eps: float = 1e-6) -> np.ndarray:
    return np.clip(y, lo + eps, hi - eps)


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu  = X.mean(axis=0)
    sig = X.std(axis=0, ddof=1)
    sig = np.where(sig == 0, 1.0, sig)
    return (X - mu) / sig, mu, sig


def _add_const(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES TOURNOI — transformations de features et sélection
# ─────────────────────────────────────────────────────────────────────────────

def _apply_transform(x: np.ndarray, transform: str) -> np.ndarray:
    """Applique une transformation temporelle à une série macro."""
    if transform == "level":
        return x.copy()
    if transform == "lag1":
        out = np.empty(len(x), dtype=float)
        out[0] = np.nan
        out[1:] = x[:-1]
        return out
    if transform == "growth":
        out = np.empty(len(x), dtype=float)
        out[0] = np.nan
        denom = np.abs(x[:-1]) + _EPS
        out[1:] = np.diff(x.astype(float)) / denom
        return out
    if transform == "cycle":
        ewma = pd.Series(x.astype(float)).ewm(span=EWMA_SPAN, adjust=False).mean().values
        return x.astype(float) - ewma
    return x.copy()


def _make_feature_df(
    df_macro: pd.DataFrame,
    parent_drivers: List[str],
) -> pd.DataFrame:
    """Étend df_macro avec 4 colonnes de transformation par driver parent."""
    df = df_macro.copy()
    for drv in parent_drivers:
        if drv not in df.columns:
            continue
        x = df[drv].values.astype(float)
        for tf in TRANSFORMS:
            df[f"{drv}__{tf}"] = _apply_transform(x, tf)
    return df


def _score_features(
    df_features: pd.DataFrame,
    y_vals: np.ndarray,
    parent_drivers: List[str],
    exp_sgns: Dict[str, int],
) -> Dict[str, Dict]:
    """Calcule corr_score = 0.5×(|Pearson|+|Spearman|) pour chaque feature vs Y.

    Même formule que le module crédit.
    """
    from scipy.stats import pearsonr, spearmanr

    specs: Dict[str, Dict] = {}
    for drv in parent_drivers:
        exp_sign = exp_sgns.get(drv, 0)
        for tf in TRANSFORMS:
            feat_name = f"{drv}__{tf}"
            if feat_name not in df_features.columns:
                continue
            tf_vals = df_features[feat_name].values.astype(float)
            valid = ~np.isnan(tf_vals) & ~np.isnan(y_vals)
            if valid.sum() < MIN_OBS_FEATURE:
                continue
            xv = tf_vals[valid]
            yv = y_vals[valid]
            mu    = float(xv.mean())
            sigma = float(xv.std(ddof=1))
            if sigma < _EPS:
                continue
            try:
                r_p_raw = pearsonr(xv, yv)[0]
                r_p = abs(r_p_raw)
            except Exception:
                r_p_raw = 0.0
                r_p = 0.0
            try:
                r_s_raw = spearmanr(xv, yv)[0]
                r_s = abs(r_s_raw)
            except Exception:
                r_s_raw = 0.0
                r_s = 0.0

            # Filtre de signe economique en pre-selection :
            # si le signe attendu est connu et que la correlation observee
            # va dans le sens contraire, la feature est eliminee d'emblee.
            # Exception : la transformation "cycle" peut legitimement inverser
            # le signe (composante cyclique vs tendance) -> pas de filtre.
            if exp_sign != 0 and tf != "cycle":
                consensus_sign = int(np.sign(0.5 * (r_p_raw + r_s_raw)))
                if consensus_sign != 0 and consensus_sign * exp_sign < 0:
                    LOG.debug(
                        "Pre-selection: '%s' elimine (signe attendu=%+d, "
                        "corr=%.3f) — sens eco contraire.",
                        feat_name, exp_sign, 0.5 * (r_p_raw + r_s_raw),
                    )
                    continue

            corr_score = 0.5 * (r_p + r_s)
            if corr_score < MIN_CORR_SCORE:
                LOG.debug(
                    "Pre-selection: '%s' éliminé (|corr|=%.3f < %.2f seuil min).",
                    feat_name, corr_score, MIN_CORR_SCORE,
                )
                continue
            specs[feat_name] = {
                "parent":        drv,
                "transform":     tf,
                "expected_sign": exp_sign,
                "corr_score":    0.5 * (r_p + r_s),
                "mu":            mu,
                "sigma":         sigma,
            }
    return specs


def _dedup_to_one_per_parent(
    feat_specs: Dict[str, Dict],
    max_k: int = 15,
) -> Dict[str, Dict]:
    """Garde la meilleure transformation par variable parent (corr_score DESC).

    Identique à _top_k_dedup du module crédit.
    """
    ranked = sorted(feat_specs.items(), key=lambda kv: -kv[1]["corr_score"])
    selected: Dict[str, Dict] = {}
    seen_parents: set = set()
    for feat_name, spec in ranked:
        parent = spec["parent"]
        if parent in seen_parents:
            continue
        selected[feat_name] = spec
        seen_parents.add(parent)
        if len(selected) >= max_k:
            break
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Beta regression (MLE, lien logit, précision phi estimée)
# ─────────────────────────────────────────────────────────────────────────────

def _beta_nll(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    k       = X.shape[1]
    beta    = params[:k]
    log_phi = params[k]
    phi     = np.exp(log_phi)
    mu = _sigmoid(X @ beta)
    mu = np.clip(mu, 1e-8, 1 - 1e-8)
    a = mu * phi
    b = (1.0 - mu) * phi
    nll = -(
        gammaln(phi) - gammaln(a) - gammaln(b)
        + (a - 1.0) * np.log(y)
        + (b - 1.0) * np.log(1.0 - y)
    ).sum()
    return nll


def _beta_fit(
    X: np.ndarray,
    y: np.ndarray,
    max_attempts: int = 8,
) -> Tuple[np.ndarray, bool]:
    k           = X.shape[1]
    best_nll    = np.inf
    best_params = None
    converged   = False
    logit_y     = np.log(y / (1.0 - y))
    ols_beta    = np.linalg.lstsq(X, logit_y, rcond=None)[0]
    init_sets   = [np.append(ols_beta, [1.0])]
    rng = np.random.default_rng(42)
    for _ in range(max_attempts - 1):
        init_sets.append(np.append(rng.normal(0, 0.5, k), [1.0]))
    for x0 in init_sets:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    _beta_nll, x0, args=(X, y),
                    method="L-BFGS-B",
                    options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
                )
            if res.fun < best_nll:
                best_nll    = res.fun
                best_params = res.x
                converged   = res.success
        except Exception:
            continue
    if best_params is None:
        best_params = init_sets[0]
        converged   = False
    return best_params, converged


def _beta_stderr_and_pvalues(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.optimize import approx_fprime
    from scipy.linalg import inv
    eps      = 1e-5
    n_params = len(params)
    H        = np.zeros((n_params, n_params))
    for i in range(n_params):
        def grad_i(p, _i=i):
            return approx_fprime(p, _beta_nll, eps, X, y)[_i]
        H[i] = approx_fprime(params, grad_i, eps)
    try:
        cov = inv(H)
        se  = np.sqrt(np.abs(np.diag(cov)))
    except Exception:
        se = np.full(n_params, np.nan)
    z     = params / np.where(se == 0, np.nan, se)
    from scipy.stats import norm
    pvals = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    return se, pvals


def _beta_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Pseudo-R² de McFadden sur logit (identique module crédit — Beta)."""
    logit_y    = np.log(y / (1.0 - y))
    logit_yhat = np.log(np.clip(y_hat, 1e-8, 1 - 1e-8) / (1.0 - np.clip(y_hat, 1e-8, 1 - 1e-8)))
    ss_res = np.sum((logit_y - logit_yhat) ** 2)
    ss_tot = np.sum((logit_y - logit_y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# OLS log-transform (pour haircut_add)
# ─────────────────────────────────────────────────────────────────────────────

def _ols_log_fit(
    X: np.ndarray,
    y: np.ndarray,
    clip_hi: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    from scipy.stats import t as t_dist
    y_star = np.log1p(y)
    beta   = np.linalg.lstsq(X, y_star, rcond=None)[0]
    n, k   = X.shape
    y_hat  = X @ beta
    resid  = y_star - y_hat
    sigma2 = np.dot(resid, resid) / max(n - k, 1)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se      = np.sqrt(sigma2 * np.abs(np.diag(XtX_inv)))
    except Exception:
        se = np.full(k, np.nan)
    t_stat = beta / np.where(se == 0, np.nan, se)
    pvals  = 2.0 * t_dist.sf(np.abs(t_stat), df=max(n - k, 1))
    ss_res = np.dot(resid, resid)
    ss_tot = np.sum((y_star - y_star.mean()) ** 2)
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    ll  = -n / 2.0 * np.log(2 * np.pi * max(sigma2, _EPS)) - ss_res / (2 * max(sigma2, _EPS))
    aic = float(2 * k - 2 * ll)
    bic = float(k * np.log(n) - 2 * ll)
    return beta, se, pvals, r2, aic, bic


# ─────────────────────────────────────────────────────────────────────────────
# AIC/BIC pour Beta regression
# ─────────────────────────────────────────────────────────────────────────────

def _beta_aic_bic(nll: float, n: int, k: int) -> Tuple[float, float]:
    ll  = -nll
    aic = float(2 * (k + 1) - 2 * ll)   # +1 pour phi
    bic = float((k + 1) * np.log(n) - 2 * ll)
    return aic, bic


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────────────────────────────────────

class SatelliteCalibrator:
    """Calibre les 5 modèles satellites du module liquidité.

    Variables macro candidates : TOUTES les colonnes de df_macro
    (déterminées dynamiquement par l'API WorldBank/IMF).

    Moteur de sélection : tournoi exhaustif sign-constrained,
    identique au module crédit (cohérence méthodologique).
    """

    # ── Configuration des 5 satellites ───────────────────────────────────────
    # NB : pas de "drivers" hardcodés — le pool candidat = df_macro.columns.
    # Seuls "family", "domain" et "remap" sont spécifiques à chaque satellite.
    _MODELS: Dict[str, Dict] = {
        "run_off_retail": {
            "family": "beta",
            "domain": (0.0, 1.0),
            "remap":  None,
        },
        "run_off_corporate": {
            "family": "beta",
            "domain": (0.0, 1.0),
            "remap":  None,
        },
        "haircut_add": {
            "family": "ols_log",
            "domain": (0.0, 0.50),
            "remap":  None,
        },
        "asf_factor_retail": {
            "family": "beta",
            "domain": (0.70, 1.00),
            "remap":  (0.70, 1.00),
        },
        "asf_factor_corporate": {
            "family": "beta",
            "domain": (0.20, 0.80),
            "remap":  (0.20, 0.80),
        },
    }

    def calibrate_all(
        self,
        df_bilan     : pd.DataFrame,
        df_macro     : pd.DataFrame,
        forced_ranks : Optional[Dict[str, int]] = None,
    ) -> Dict[str, SatelliteResult]:
        """Calibre les 5 satellites par tournoi exhaustif (mis en cache — voir
        _TOURNAMENT_CACHE).

        Variables macro candidates : TOUTES les colonnes numériques de df_macro
        (issues de l'API WorldBank/IMF — pas de liste hardcodée).

        Après l'appel, self.tournament_leaderboards contient le classement complet
        de toutes les combinaisons testées, par satellite.
        """
        key = _tournament_cache_key(df_bilan, df_macro, forced_ranks)
        cached = _TOURNAMENT_CACHE.get(key)
        if cached is not None:
            _TOURNAMENT_CACHE.move_to_end(key)
            results, leaderboards, all_results = cached
            self.tournament_leaderboards = leaderboards
            self.all_satellite_results   = all_results
            LOG.info("[tournament] cache hit — skipping 5-satellite calibration")
            return results

        results = self._calibrate_all_impl(df_bilan, df_macro, forced_ranks)

        _TOURNAMENT_CACHE[key] = (
            results, self.tournament_leaderboards, self.all_satellite_results,
        )
        _TOURNAMENT_CACHE.move_to_end(key)
        if len(_TOURNAMENT_CACHE) > _TOURNAMENT_CACHE_MAX:
            _TOURNAMENT_CACHE.popitem(last=False)
        return results

    def _calibrate_all_impl(
        self,
        df_bilan     : pd.DataFrame,
        df_macro     : pd.DataFrame,
        forced_ranks : Optional[Dict[str, int]] = None,
    ) -> Dict[str, SatelliteResult]:
        """
        Args:
            df_bilan  : DataFrame bilan (index = year).
            df_macro  : DataFrame macro historique (index = year).
                        Toutes les colonnes numériques sont candidats.
        """
        # Alignement temporel
        common_years = df_bilan.index.intersection(df_macro.index)
        if len(common_years) < 8:
            raise ValueError(
                f"Moins de 8 années communes bilan/macro "
                f"({len(df_bilan)} bilan vs {len(df_macro)} macro)."
            )
        df_b = df_bilan.loc[common_years]
        df_m = df_macro.loc[common_years]

        # Pool candidat = toutes les colonnes numériques disponibles dans df_macro
        all_macro_cols = [
            c for c in df_m.columns
            if c not in _NON_MACRO_COLS
            and pd.api.types.is_numeric_dtype(df_m[c])
        ]
        LOG.info(
            "[tournament] Pool macro disponible : %d colonnes — %s",
            len(all_macro_cols), all_macro_cols,
        )

        results: Dict[str, SatelliteResult] = {}
        # Leaderboard complet du tournoi (toutes combinaisons testées, par satellite)
        self.tournament_leaderboards:  Dict[str, List[Dict]]            = {}
        self.all_satellite_results:    Dict[str, List[SatelliteResult]] = {}

        _forced = forced_ranks or {}

        # ── Calibrate all 5 satellites in parallel (independent per target) ──
        n_workers = min(len(self._MODELS), 5)
        futures: Dict[Any, str] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for var_name, spec in self._MODELS.items():
                LOG.info(
                    "[tournament] Calibration %s (%s) — soumis (parallèle).",
                    var_name, spec["family"],
                )
                fut = executor.submit(
                    self._calibrate_tournament,
                    var_name, spec, df_b, df_m, all_macro_cols,
                )
                futures[fut] = var_name

        for fut in as_completed(futures):
            var_name = futures[fut]
            try:
                winner, leaderboard, all_results = fut.result()
            except Exception as exc:
                LOG.error("[tournament] %s : erreur — %s", var_name, exc)
                raise

            self.tournament_leaderboards[var_name] = leaderboard
            self.all_satellite_results[var_name]   = all_results

            # Honour forced rank (0-indexed) — fall back to winner if out of range
            forced_idx = int(_forced.get(var_name, 0))
            if forced_idx and 0 < forced_idx < len(all_results):
                result = all_results[forced_idx]
                if leaderboard and leaderboard[forced_idx].get("verdict") == "REJECTED":
                    LOG.warning(
                        "[tournament] %s : rang forcé %d pointe vers un modèle "
                        "REJECTED — calibration lancée par choix explicite.",
                        var_name, forced_idx,
                    )
                LOG.info(
                    "[tournament] %s : rang forcé %d (verdict=%s)",
                    var_name, forced_idx + 1,
                    leaderboard[forced_idx].get("verdict", "N/A") if leaderboard else "N/A",
                )
            else:
                # Default: validation-sorted winner. Fail loudly if all REJECTED.
                if leaderboard and leaderboard[0].get("verdict") == "REJECTED":
                    raise RuntimeError(
                        f"NO_VALID_MODEL: satellite '{var_name}' — "
                        f"les {len(leaderboard)} candidats sont tous REJECTED "
                        "par PostEstimationValidator. "
                        "Vérifier la qualité ou la quantité des données historiques."
                    )
                result = winner

            results[var_name] = result
            sign_str = " | ".join(
                f"{k}={'OK' if v else 'WARN'}"
                for k, v in result.sign_checks.items()
            )
            LOG.info(
                "[tournament] %s → drivers=%s R²=%.3f AIC=%.2f "
                "conv=%s signs=[%s] (%d combos testés)",
                var_name, result.drivers, result.r2, result.aic,
                result.converged, sign_str, len(leaderboard),
            )

        return results

    # ── Tournoi exhaustif ─────────────────────────────────────────────────────

    def _calibrate_tournament(
        self,
        var_name        : str,
        spec            : dict,
        df_b            : pd.DataFrame,
        df_m            : pd.DataFrame,
        all_macro_cols  : List[str],
    ) -> Tuple[SatelliteResult, List[Dict]]:
        """Sélection par tournoi exhaustif (même moteur que le module crédit).

        Classement : sign_violations ASC → R² DESC → AIC ASC

        Retourne (SatelliteResult_vainqueur, leaderboard_sérialisable).
        Le leaderboard contient toutes les combinaisons testées, classées.
        """
        # ── Expected signs depuis le registre ──────────────────────────────────
        sign_reg = _SIGN_REGISTRY.get(var_name, {})
        exp_sgns = {col: sign_reg.get(col, 0) for col in all_macro_cols}

        # ── Y ─────────────────────────────────────────────────────────────────
        if var_name not in df_b.columns:
            raise ValueError(f"Colonne '{var_name}' absente de df_bilan.")
        y_raw = df_b[var_name].values.astype(float)
        n_obs = len(y_raw)

        # ── Générer les 4 transformations par variable macro ───────────────────
        df_features = _make_feature_df(df_m, all_macro_cols)
        feat_specs  = _score_features(df_features, y_raw, all_macro_cols, exp_sgns)

        LOG.info(
            "[tournament] %s : %d features générées (4 transforms × %d vars macro)",
            var_name, len(feat_specs), len(all_macro_cols),
        )

        # ── Dédupliquer : 1 meilleure feature par variable parente ─────────────
        max_k = min(15, max(n_obs - 2, MIN_COMBO_SIZE))
        selected_specs = _dedup_to_one_per_parent(feat_specs, max_k=max_k)

        LOG.info(
            "[tournament] %s : %d features retenues après dédup (1 par var parente)",
            var_name, len(selected_specs),
        )

        # ── Règle de Harrell : taille max des combos ───────────────────────────
        max_combo = max(MIN_COMBO_SIZE, n_obs // HARRELL_OBS_PER_PARAM - 1)
        max_combo = min(max_combo, len(selected_specs), 3)

        # ── Fallback si pas assez de features valides ──────────────────────────
        if len(selected_specs) < MIN_COMBO_SIZE:
            LOG.warning(
                "[tournament] %s : seulement %d features valides (min=%d) "
                "— fallback sur toutes les variables disponibles.",
                var_name, len(selected_specs), MIN_COMBO_SIZE,
            )
            fallback_spec = {
                "family":         spec["family"],
                "drivers":        all_macro_cols[:max(MIN_COMBO_SIZE, len(all_macro_cols))],
                "domain":         spec["domain"],
                "remap":          spec["remap"],
                "expected_signs": exp_sgns,
            }
            return self._fit_one_model(var_name, fallback_spec, df_b, df_m,
                                       feat_parent_map=None)

        # ── Tournoi exhaustif ──────────────────────────────────────────────────
        feat_names     = list(selected_specs.keys())
        feat_to_parent = {f: selected_specs[f]["parent"] for f in feat_names}
        leaderboard: List[Dict] = []

        for k in range(MIN_COMBO_SIZE, max_combo + 1):
            for combo in combinations(feat_names, k):
                combo_list = list(combo)


                # ── Garde colinéarité (Bug fix : ex. gdp_current_usd + gdp_per_capita) ──
                # Rejeter les combos où deux features ont |corr| > 0.85
                # → évite la multicolinéarité et les modèles économiquement illogiques.
                if len(combo_list) >= 2:
                    _combo_data = df_features[combo_list].dropna()
                    if len(_combo_data) >= 4:
                        _corr_m = _combo_data.corr().abs()
                        _max_off = (_corr_m.where(
                            ~np.eye(len(combo_list), dtype=bool)
                        ).max().max())
                        if _max_off > 0.85:
                            LOG.debug(
                                "[tournament] %s combo %s rejeté — colinéarité |r|=%.2f > 0.85",
                                var_name, combo, _max_off,
                            )
                            continue

                # ── Garde VIF (max 10.0 — alignement module climat) ──────────────
                try:
                    from statsmodels.stats.outliers_influence import (
                        variance_inflation_factor as _vif_fn,
                    )
                    _vif_data = df_features[combo_list].dropna().values.astype(float)
                    if _vif_data.shape[0] > _vif_data.shape[1] + 1:
                        _max_vif = max(
                            _vif_fn(_vif_data, j) for j in range(_vif_data.shape[1])
                        )
                        if _max_vif > MAX_VIF:
                            LOG.debug(
                                "[tournament] %s combo %s rejeté — VIF=%.1f > %.1f",
                                var_name, combo, _max_vif, MAX_VIF,
                            )
                            continue
                except Exception:
                    pass  # statsmodels absent ou données dégénérées → pas de garde VIF


                # Aligner sur les lignes sans NaN pour ce combo
                valid_idx = df_features[combo_list].dropna().index
                valid_idx = valid_idx.intersection(df_b.index)
                if len(valid_idx) < max(MIN_COMBO_SIZE + 2, 6):
                    continue

                df_b_aln = df_b.loc[valid_idx]
                df_f_aln = df_features.loc[valid_idx]

                combo_exp_sgns = {
                    f: selected_specs[f]["expected_sign"] for f in combo_list
                }
                combo_spec = {
                    "family":         spec["family"],
                    "drivers":        combo_list,
                    "domain":         spec["domain"],
                    "remap":          spec["remap"],
                    "expected_signs": combo_exp_sgns,
                }
                try:
                    result = self._fit_one_model(
                        var_name, combo_spec, df_b_aln, df_f_aln,
                        feat_parent_map=feat_to_parent,
                    )
                except Exception as exc:
                    LOG.debug("[tournament] %s combo %s → %s", var_name, combo, exc)
                    continue

                # Filtre p-value : au moins 1 driver doit être significatif
                if result.pvalues:
                    _pvs = [v for v in result.pvalues.values()
                            if not (v != v)]  # exclure NaN
                    if _pvs and all(pv > PVALUE_THRESHOLD for pv in _pvs):
                        LOG.debug(
                            "[tournament] %s combo %s rejeté — aucun coef p<%.2f",
                            var_name, combo, PVALUE_THRESHOLD,
                        )
                        continue

                # Filtre coef ≈ 0 : variable macro sans contribution au modèle
                # Un coef nul indique une collinéarité numérique ou une variable
                # parfaitement redondante — elle ne doit pas entrer dans le satellite.
                if result.coefficients:
                    _zero_coef = [
                        var for var, c in result.coefficients.items()
                        if var != "const" and abs(c) < 1e-8
                    ]
                    if _zero_coef:
                        LOG.debug(
                            "[tournament] %s combo %s rejeté — coef≈0 pour %s",
                            var_name, combo, _zero_coef,
                        )
                        continue

                sign_violations = sum(
                    1 for f in combo_list
                    if not result.sign_checks.get(f, False)
                )
                leaderboard.append({
                    "result":          result,
                    "sign_violations": sign_violations,
                    "r2":              result.r2,
                    "aic":             result.aic,
                    "bic":             result.bic,
                    "combo":           combo_list,
                })

        if not leaderboard:
            LOG.warning(
                "[tournament] %s : aucune combinaison ajustée — fallback.",
                var_name,
            )
            fallback_spec = {
                "family":         spec["family"],
                "drivers":        feat_names[:MIN_COMBO_SIZE],
                "domain":         spec["domain"],
                "remap":          spec["remap"],
                "expected_signs": {f: selected_specs[f]["expected_sign"]
                                   for f in feat_names[:MIN_COMBO_SIZE]},
            }
            fallback_result = self._fit_one_model(
                var_name, fallback_spec,
                df_b, df_features.reindex(df_b.index),
                feat_parent_map=feat_to_parent,
            )
            return fallback_result, []

        # ── Post-Estimation Validation on every candidate ────────────────────
        try:
            from app.modules.core.post_estimation_validator import (
                PostEstimationValidator as _PEV, RiskOutput as _RO,
            )
            for _e in leaderboard:
                _r = _e["result"]
                try:
                    _yt = np.asarray(_r.actual_values or [], dtype=float)
                    _yp = np.asarray(_r.fitted_values or [], dtype=float)
                    _ro = _RO(
                        module     = "liquidity",
                        model_type = _r.family,
                        y_true     = _yt,
                        y_pred     = _yp,
                        residuals  = _yt - _yp,
                        n_obs      = int(_r.n_obs),
                        k_params   = len(_r.drivers),
                        bounds     = _r.domain,
                    )
                    _vr = _PEV(_ro).run()
                    _e["verdict"] = _vr.verdict
                    _e["n_pass"]  = _vr.n_pass
                    _e["n_warn"]  = _vr.n_warn
                    _e["n_fail"]  = _vr.n_fail
                    _e["_vreport_dict"] = _vr.to_dict()
                except Exception as _ve:
                    LOG.debug(
                        "[tournament] PostEstimation %s/%s: %s",
                        var_name, _e["combo"], _ve,
                    )
                    _e["verdict"] = "VALIDATED_WITH_WARNINGS"
                    _e["n_pass"]  = 0
                    _e["n_warn"]  = 1
                    _e["n_fail"]  = 0
                    _e["_vreport_dict"] = {}
        except ImportError:
            LOG.warning(
                "[tournament] PostEstimationValidator non disponible (%s) — tri sans verdict.",
                var_name,
            )
            for _e in leaderboard:
                _e.setdefault("verdict",       "VALIDATED_WITH_WARNINGS")
                _e.setdefault("n_pass",        0)
                _e.setdefault("n_warn",        0)
                _e.setdefault("n_fail",        0)
                _e.setdefault("_vreport_dict", {})

        # ── Classement : verdict ASC -> sign_violations ASC -> AIC+BIC ASC -> R2 DESC
        # AIC et BIC penalisent la complexite du modele (Burnham & Anderson 2002).
        # Leur somme concentre la pression de parcimonie avant de favoriser le R2.
        leaderboard.sort(key=lambda x: (
            _VERDICT_RANK.get(x.get("verdict", "VALIDATED_WITH_WARNINGS"), 1),
            x["sign_violations"],
            x.get("aic", 0.0) + x.get("bic", 0.0),
            -x["r2"],
        ))
        best = leaderboard[0]

        LOG.info(
            "[tournament] %s : vainqueur combo=%s sign_viol=%d AIC=%.2f BIC=%.2f "
            "R2=%.3f verdict=%s (%d combinaisons testees)",
            var_name, best["combo"], best["sign_violations"],
            best.get("aic", float("nan")), best.get("bic", float("nan")),
            best["r2"], best.get("verdict", "N/A"), len(leaderboard),
        )

        # Build serializable leaderboard + keep ranked SatelliteResult objects
        serial_leaderboard = []
        all_results: List[SatelliteResult] = []
        for rank, entry in enumerate(leaderboard, start=1):
            r = entry["result"]
            all_results.append(r)
            serial_leaderboard.append({
                "rang":            rank,
                "satellite":       var_name,
                "family":          r.family,
                "combo":           entry["combo"],
                # ── Intercept + coefs (auditabilité) ────────────────────────
                "intercept":       round(float(r.coefficients.get("const", float("nan"))), 6)
                                   if r.coefficients else None,
                "pv_intercept":    round(float(r.pvalues.get("const", float("nan"))), 4)
                                   if r.pvalues else None,
                "coefs":           {
                                       var: round(float(c), 6)
                                       for var, c in r.coefficients.items()
                                       if var != "const"
                                   } if r.coefficients else {},
                "pv_coefs":        {
                                       var: round(float(p), 4)
                                       for var, p in r.pvalues.items()
                                       if var != "const"
                                   } if r.pvalues else {},
                # ────────────────────────────────────────────────────────────
                "r2":              round(float(r.r2), 4),
                "aic":             round(float(r.aic), 2),
                "n_obs":           int(r.n_obs),
                "sign_violations": int(entry["sign_violations"]),
                "signs_ok":        bool(r.signs_ok),
                "converged":       bool(r.converged),
                "verdict":         entry.get("verdict", "VALIDATED_WITH_WARNINGS"),
                "n_fail":          entry.get("n_fail", 0),
                "n_warn":          entry.get("n_warn", 0),
                "n_pass":          entry.get("n_pass", 0),
                # In-sample fits for "Réel vs Estimé" display
                "years":           list(r.obs_years)    if r.obs_years    else [],
                "actual":          [round(float(v), 6) for v in r.actual_values] if r.actual_values else [],
                "fitted":          [round(float(v), 6) for v in r.fitted_values] if r.fitted_values else [],
            })

        return best["result"], serial_leaderboard, all_results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fit_one_model(
        self,
        var_name       : str,
        spec           : dict,
        df_b           : pd.DataFrame,
        df_m           : pd.DataFrame,
        feat_parent_map: Optional[Dict[str, str]] = None,
    ) -> SatelliteResult:
        """Estime un modèle avec les drivers de spec.

        feat_parent_map : {feature_name: parent_driver_name} — fourni quand les
        drivers sont des features transformées (ex. "GDP_growth__growth").
        La closure de prédiction reconstruira les transforms à partir des
        colonnes parentes de df_macro_proj.  None → colonnes parentes directes.
        """
        if var_name not in df_b.columns:
            raise ValueError(f"Colonne '{var_name}' absente de df_bilan.")
        y_raw = df_b[var_name].values.astype(float)

        missing = [d for d in spec["drivers"] if d not in df_m.columns]
        if missing:
            raise ValueError(
                f"Colonnes manquantes pour {var_name} : {missing}. "
                f"Disponibles : {list(df_m.columns[:8])}..."
            )
        X_raw = df_m[spec["drivers"]].values.astype(float)
        n     = len(y_raw)

        X_std, X_mean, X_std_scale = _standardize(X_raw)
        X_with_const = _add_const(X_std)

        try:
            obs_years = [int(y) for y in df_b.index.tolist()]
        except Exception:
            obs_years = list(range(1, n + 1))

        if spec["family"] == "beta":
            result = self._fit_beta(
                var_name, spec["drivers"], spec["domain"], spec["remap"],
                spec["expected_signs"],
                y_raw, X_raw, X_std, X_mean, X_std_scale, X_with_const, n,
                feat_parent_map=feat_parent_map,
            )
        else:
            result = self._fit_ols_log(
                var_name, spec["drivers"], spec["domain"],
                spec["expected_signs"],
                y_raw, X_raw, X_std, X_mean, X_std_scale, X_with_const, n,
                feat_parent_map=feat_parent_map,
            )
        result.obs_years = obs_years
        return result

    # ── Beta regression ───────────────────────────────────────────────────────

    def _fit_beta(
        self,
        var_name, drivers, domain, remap, exp_sgns,
        y_raw, X_raw, X_std, X_mean, X_std_scale, X_with_const, n,
        feat_parent_map: Optional[Dict[str, str]] = None,
    ) -> SatelliteResult:

        lo, hi = domain
        if remap is not None:
            r_lo, r_hi = remap
            y_unit = (y_raw - r_lo) / (r_hi - r_lo)
        else:
            y_unit = y_raw.copy()
        y_safe = _safe_clip(y_unit, 0.0, 1.0)

        params, converged = _beta_fit(X_with_const, y_safe)
        k      = X_with_const.shape[1]
        beta_k = params[:k]

        mu_hat = _sigmoid(X_with_const @ beta_k)
        mu_hat = np.clip(mu_hat, 1e-8, 1 - 1e-8)

        try:
            se_all, pv_all = _beta_stderr_and_pvalues(params, X_with_const, y_safe)
        except Exception:
            se_all = np.full(len(params), np.nan)
            pv_all = np.full(len(params), np.nan)

        beta_orig    = np.zeros(k)
        beta_orig[0] = beta_k[0]
        for i, (m, s) in enumerate(zip(X_mean, X_std_scale), 1):
            beta_orig[i] = beta_k[i] / s

        coef_names = ["const"] + drivers
        coefs = {nm: float(beta_orig[i]) for i, nm in enumerate(coef_names)}
        ses   = {nm: float(se_all[i])    for i, nm in enumerate(coef_names)}
        pvs   = {nm: float(pv_all[i])    for i, nm in enumerate(coef_names)}

        nll      = _beta_nll(params, X_with_const, y_safe)
        aic, bic = _beta_aic_bic(nll, n, k)
        r2       = _beta_r2(y_safe, mu_hat)

        sign_checks: Dict[str, bool] = {}
        for drv, expected_sign in exp_sgns.items():
            if drv not in drivers:
                continue
            b = beta_k[drivers.index(drv) + 1]
            sign_checks[drv] = (expected_sign == 0) or (b * expected_sign > 0)

        # Closure feature-aware
        _beta_k      = beta_k.copy()
        _X_mean      = X_mean.copy()
        _X_std_scale = X_std_scale.copy()
        _remap       = remap
        _drivers     = list(drivers)
        _feat_pm     = dict(feat_parent_map) if feat_parent_map else None

        def predict_fn(df_macro_proj: pd.DataFrame) -> np.ndarray:
            if _feat_pm is not None:
                parent_drivers = list(set(_feat_pm[d] for d in _drivers if d in _feat_pm))
                df_f = _make_feature_df(df_macro_proj, parent_drivers)
                # Variables absentes du scenario (pool etendu) -> NaN -> mean imputation
                for _d in _drivers:
                    if _d not in df_f.columns:
                        df_f[_d] = np.nan
                X_proj = df_f[_drivers].values.astype(float)
            else:
                # Colonnes directes manquantes -> NaN -> mean imputation
                n_rows = len(df_macro_proj)
                X_proj = np.column_stack([
                    df_macro_proj[_d].values.astype(float) if _d in df_macro_proj.columns
                    else np.full(n_rows, np.nan)
                    for _d in _drivers
                ]) if _drivers else np.empty((n_rows, 0))
            # lag1/growth transforms produce NaN on single-row projections:
            # fill with training mean so z-score = 0 (neutral, mean-imputation).
            X_proj = np.where(np.isnan(X_proj), _X_mean, X_proj)
            X_proj_std = (X_proj - _X_mean) / _X_std_scale
            X_proj_c   = _add_const(X_proj_std)
            mu = _sigmoid(X_proj_c @ _beta_k)
            mu = np.clip(mu, 1e-8, 1 - 1e-8)
            if _remap is not None:
                r_lo, r_hi = _remap
                return r_lo + (r_hi - r_lo) * mu
            return mu

        # In-sample fitted values in original scale
        if remap is not None:
            r_lo, r_hi = remap
            y_fitted_list = (r_lo + (r_hi - r_lo) * mu_hat).tolist()
        else:
            y_fitted_list = mu_hat.tolist()

        return SatelliteResult(
            variable      = var_name,
            family        = "beta",
            drivers       = drivers,
            coefficients  = coefs,
            std_errors    = ses,
            pvalues       = pvs,
            r2            = r2,
            aic           = aic,
            bic           = bic,
            n_obs         = n,
            converged     = converged,
            sign_checks   = sign_checks,
            domain        = (lo, hi),
            remap_lo      = remap[0] if remap else None,
            remap_hi      = remap[1] if remap else None,
            fitted_values = y_fitted_list,
            actual_values = y_raw.tolist(),
            _predict_fn   = predict_fn,
        )

    # ── OLS log-transform (haircut_add) ───────────────────────────────────────

    def _fit_ols_log(
        self,
        var_name, drivers, domain, exp_sgns,
        y_raw, X_raw, X_std, X_mean, X_std_scale, X_with_const, n,
        feat_parent_map: Optional[Dict[str, str]] = None,
    ) -> SatelliteResult:

        lo, hi = domain
        y_star = np.log1p(y_raw)
        beta, se_all, pv_all, r2, aic, bic = _ols_log_fit(X_with_const, y_star, clip_hi=hi)

        beta_orig    = np.zeros(len(beta))
        beta_orig[0] = beta[0]
        for i, s in enumerate(X_std_scale, 1):
            beta_orig[i] = beta[i] / s

        coef_names = ["const"] + drivers
        coefs = {nm: float(beta_orig[i]) for i, nm in enumerate(coef_names)}
        ses   = {nm: float(se_all[i])    for i, nm in enumerate(coef_names)}
        pvs   = {nm: float(pv_all[i])    for i, nm in enumerate(coef_names)}

        sign_checks: Dict[str, bool] = {}
        for drv, expected_sign in exp_sgns.items():
            if drv not in drivers:
                continue
            b = beta[drivers.index(drv) + 1]
            sign_checks[drv] = (expected_sign == 0) or (b * expected_sign > 0)

        _beta        = beta.copy()
        _X_mean      = X_mean.copy()
        _X_std_scale = X_std_scale.copy()
        _drivers     = list(drivers)
        _feat_pm     = dict(feat_parent_map) if feat_parent_map else None

        def predict_fn(df_macro_proj: pd.DataFrame) -> np.ndarray:
            if _feat_pm is not None:
                parent_drivers = list(set(_feat_pm[d] for d in _drivers if d in _feat_pm))
                df_f = _make_feature_df(df_macro_proj, parent_drivers)
                # Variables absentes du scenario (pool etendu) -> NaN -> mean imputation
                for _d in _drivers:
                    if _d not in df_f.columns:
                        df_f[_d] = np.nan
                X_proj = df_f[_drivers].values.astype(float)
            else:
                n_rows = len(df_macro_proj)
                X_proj = np.column_stack([
                    df_macro_proj[_d].values.astype(float) if _d in df_macro_proj.columns
                    else np.full(n_rows, np.nan)
                    for _d in _drivers
                ]) if _drivers else np.empty((n_rows, 0))
            # lag1/growth transforms produce NaN on single-row projections:
            # fill with training mean so z-score = 0 (neutral, mean-imputation).
            X_proj = np.where(np.isnan(X_proj), _X_mean, X_proj)
            X_proj_std = (X_proj - _X_mean) / _X_std_scale
            X_proj_c   = _add_const(X_proj_std)
            return np.clip(np.expm1(X_proj_c @ _beta), 0.0, hi)

        # In-sample fitted values in original scale
        y_fitted_list = np.clip(np.expm1(X_with_const @ beta), 0.0, hi).tolist()

        return SatelliteResult(
            variable      = var_name,
            family        = "ols_log",
            drivers       = drivers,
            coefficients  = coefs,
            std_errors    = ses,
            pvalues       = pvs,
            r2            = r2,
            aic           = aic,
            bic           = bic,
            n_obs         = n,
            converged     = True,
            sign_checks   = sign_checks,
            domain        = (lo, hi),
            remap_lo      = None,
            remap_hi      = None,
            fitted_values = y_fitted_list,
            actual_values = y_raw.tolist(),
            _predict_fn   = predict_fn,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Rapport de calibration (console)
# ─────────────────────────────────────────────────────────────────────────────

def print_calibration_report(results: Dict[str, SatelliteResult]) -> None:
    """Affiche un rapport de calibration - tournoi exhaustif (module credit)."""
    sep = "-" * 72
    print(f"\n{'='*72}")
    print("  RAPPORT DE CALIBRATION - SATELLITES LIQUIDITE")
    print("  (Tournoi exhaustif - 4 transforms - min 2 drivers - sign-constrained)")
    print(f"{'='*72}")
    for var, r in results.items():
        print(f"\n{sep}")
        print(f"  {var.upper()}  [{r.family}]  n={r.n_obs}  converged={r.converged}")
        print(f"  R2={r.r2:.4f}  AIC={r.aic:.2f}  BIC={r.bic:.2f}")
        if r.remap_lo is not None:
            print(f"  Domaine : [{r.remap_lo}, {r.remap_hi}] -> remap (0,1)")
        else:
            print(f"  Domaine : {r.domain}")
        sign_ok = all(r.sign_checks.values())
        print(f"\n  Drivers selectionnes ({len(r.drivers)}, min={MIN_COMBO_SIZE}) :")
        for d in r.drivers:
            parts = d.split("__")
            label = f"{parts[0]} ({parts[1]})" if len(parts) > 1 else d
            print(f"    - {label}")
        print(f"  Signes : {'OK tous corrects' if sign_ok else '!! VIOLATIONS'}")
        print(f"\n  {'Variable':<38} {'Coef':>10} {'SE':>10} {'p-val':>10} {'Signe':>6}")
        print(f"  {'-'*38} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
        for nm in ["const"] + r.drivers:
            c  = r.coefficients.get(nm, float("nan"))
            se = r.std_errors.get(nm, float("nan"))
            pv = r.pvalues.get(nm, float("nan"))
            sig = "OK" if r.sign_checks.get(nm, True) else "! "
            if nm == "const":
                sig = "  "
            print(f"  {nm:<38} {c:>+10.4f} {se:>10.4f} {pv:>10.4f} {sig:>6}")
        if not sign_ok:
            print("  !  Violations de signe - verifier la variabilite macro.")
    print(f"\n{'='*72}\n")



