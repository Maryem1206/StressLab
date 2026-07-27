"""
Credit Risk Wrapper — v4 (rewritten from scratch).

Pipeline rigoureux en 5 étapes, conforme à la description du moteur :
======================================================================
  Étape 1  Sélection automatique des variables macro
           • Macro 100% API (IMF WEO + World Bank via imf_weo_fetcher)
           • 4 transformations × N vars brutes → pool de features z-scorées
           • Scoring corrélation (Pearson+Spearman), top-K dédup 1/parent
           • Combinaisons adaptatives (Harrell 2015 : ≥10 obs/paramètre)
           • Garde-fou signes économiques (expected_sign de MacroIndicator)

  Étape 2  Tournoi des 3 familles satellites
           • Logit OLS (régression linéaire sur logit(PD))
           • Beta regression (MLE, Ferrari & Cribari-Neto 2004)
           • Vasicek / Probit OLS (OLS sur Φ⁻¹(PD))
           • Leaderboard : sign_violations ASC → R² DESC → RMSE ASC
           • Scalers (μ, σ) mémorisés pour la projection

  Étape 3  Projection PD point-in-time
           • Baseline = trajectoires WEO brutes (projections FMI 2025-2030)
           • Adverse / Severe = WEO baseline + chocs scénario (rampe progressive)
           • Variables non-WEO : extrapolation AR(1) sur historique
           • Re-transformation + z-score (scalers mémorisés) → modèle gagnant → PD_PIT
           • Vérification ordinale : PD_severe ≥ PD_adverse ≥ PD_baseline

  Étape 4  Frye-Jacobs LGD (Journal of Credit Risk, 2012)
           • k calibré numériquement (brentq) sur (PD_TTC, LGD_TTC)
           • ρ = corrélation d'actifs Bâle III IRB
           • LGD_PIT(t) = Φ((Φ⁻¹(PD_PIT(t)) − k) / √(1−ρ)) / PD_PIT(t)

  Étape 5  EL = PD_PIT × LGD_PIT × EAD → PlatformResult

Academic references
===================
  • Harrell (2015) "Regression Modeling Strategies" — rule of 10 EPP
  • Ferrari & Cribari-Neto (2004) "Beta regression" — J. Applied Statistics
  • Frye & Jacobs (2012) "Credit loss and systematic LGD" — J. Credit Risk
  • BCBS (2017) CRE31 — Basel III IRB asset correlation formula
  • Drehmann & Tsatsaronis (2014) — EWMA gap filtering
  • EBA (2023) Methodological Note — stress test satellite models
"""
from __future__ import annotations

import hashlib
import logging
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq, minimize
from scipy.special import gammaln
from scipy.stats import norm, pearsonr, spearmanr

from ..base import PlatformModule, PlatformResult
from ..core.imf_weo_fetcher import (
    CREDIT_MACRO_UNIVERSE,
    FetchReport,
    fetch_credit_macro,
    universe_index,
)
from ..core.historical_scenarios import (
    CRISIS_CATALOG,
    build_adverse_severe,
    build_pd_trajectories,
    compute_historical_shocks,
    extract_pd_crisis_profile,
    get_crisis,
)
from .capital_engine import CreditCapitalEngine, load_capital_history

LOG = logging.getLogger("credit_wrapper")
warnings.filterwarnings("ignore", category=Warning)

EPS = 1e-9
EWMA_SPAN = 3
TRANSFORMS = ("level", "lag1", "growth", "cycle")
HARRELL_OBS_PER_PARAM = 10

# ── Tournament memoization ──────────────────────────────────────────────────
# The satellite tournament (_run_tournament) is a pure, deterministic function
# of historical data only — it never depends on scenario shocks. Callers that
# stress-test the SAME historical dataset across many (scenario, year) pairs
# (notably ClimateOrchestrator, which re-invokes the full credit pipeline once
# per NGFS scenario × horizon year) would otherwise re-run the entire
# hundreds-of-fits tournament every single time for an identical result.
# A small content-addressed cache eliminates that redundant work with zero
# risk of changing results, since a cache hit only occurs on byte-identical
# inputs.
_TOURNAMENT_CACHE: "OrderedDict[str, Tuple[Dict, List[Dict]]]" = OrderedDict()
_TOURNAMENT_CACHE_MAX = 64


def _tournament_cache_key(
    feature_df: pd.DataFrame,
    target: pd.Series,
    specs: Dict[str, "FeatureSpec"],
    n_obs: int,
    rho: Optional[float],
) -> str:
    h = hashlib.md5()
    h.update(pd.util.hash_pandas_object(feature_df, index=True).values.tobytes())
    h.update(",".join(feature_df.columns.astype(str)).encode())
    h.update(pd.util.hash_pandas_object(target, index=True).values.tobytes())
    h.update(",".join(sorted(specs.keys())).encode())
    h.update(f"|n_obs={n_obs}|rho={rho}".encode())
    return h.hexdigest()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION A — UTILITAIRES MATHÉMATIQUES                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _probit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return norm.ppf(p)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + EPS
    return float(1 - ss_res / ss_tot)


def _aic(n: int, rss: float, k: int) -> float:
    return float(n * np.log(max(rss / n, EPS)) + 2 * k)


def _bic(n: int, rss: float, k: int) -> float:
    return float(n * np.log(max(rss / n, EPS)) + k * np.log(n))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION B — CHARGEMENT DU FICHIER PD                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Column name aliases — all lowercase, substring match ─────────────────────
_YEAR_KEYWORDS = (
    "year", "yr", "annee", "année", "an ", "date", "period", "temps",
    "fiscal_year", "fiscal year", "financial_year", "exercice",
    "reporting_year", "obs_year", "calendar_year", "sana",
)
_PD_KEYWORDS = (
    # English
    "default rate", "default_rate", "defaultrate", "default-rate",
    "default ratio", "default_ratio",
    "pd", "pd_pit", "pd_avg", "pd_thr",
    "probability of default", "probability_of_default", "prob_default",
    "loss_rate", "loss rate", "loss-rate",
    "charge_off", "charge off", "charge-off", "chargeoff",
    "write_off", "write off", "write-off", "writeoff",
    "credit_loss", "credit loss",
    "nonperforming", "non_performing", "non performing",
    "npl_ratio", "npl ratio", "npl_rate", "npl rate", "npl",
    "npe_ratio", "npe",
    "impairment_rate", "impairment rate",
    # French
    "taux de d", "taux_de_d", "taux_d", "tx_d",
    "taux défaut", "taux_défaut", "taux défaut",
    "taux de défaut", "taux de defaut", "taux defaut",
    "taux_defaut", "taux_sinistre", "taux sinistre",
    "défaut", "defaut", "sinistralite",
    "perte_sur_créances", "perte sur créances", "taux_pertes",
    # Arabic transliteration
    "maadal_al_ta3athor", "nisbat_al_qard",
)
_EAD_KEYWORDS = (
    "ead", "e.a.d", "exposure", "exposition", "encours", "portefeuille",
    "portfolio", "total_ead", "ead_total", "actif_pondere", "actif_pond",
    "risque_ponderer", "montant_exp", "montant_exposition", "exp_defaut",
    "expo_defaut", "expo_def", "gross_exp", "net_exp", "credit_exp",
    "loan_exposure", "outstanding", "outstanding_balance", "solde",
    "credit_outstand", "balance", "encours_credit", "encours_pret",
    "volume_credit", "total_actif", "actifs_ponderes", "rwa_exp",
)
_CLIMATE_KW = {"ngfs", "iiasa", "nigem", "geme3", "gem_e3", "nijem", "capital"}


def _best_parse_csv(path: Path) -> Optional[pd.DataFrame]:
    """Try multiple separators/decimals to parse a CSV robustly."""
    best_df, best_num = None, -1
    for sep in [";", ",", "\t"]:
        for dec in [",", "."]:
            try:
                df = pd.read_csv(path, sep=sep, decimal=dec, thousands=" ")
                if len(df.columns) < 2:
                    continue
                n_num = len(df.select_dtypes(include=[np.number]).columns)
                if n_num > best_num:
                    best_num, best_df = n_num, df
            except Exception:
                continue
    return best_df


def _load_credit_dataset(
    upload_paths: Dict[str, str],
) -> Tuple[pd.Series, float, int]:
    """
    Load PD and EAD time-series from the uploaded historical dataset.

    The dataset must contain at minimum three columns:
      - year  (or année/annee/date)
      - PD    (default rate, pd, taux_d …)
      - EAD   (exposure, exposition, encours, portefeuille …) in millions

    Returns
    -------
    pd_series    : pd.Series indexed by year (int), values in (0, 1)
    ead_series   : pd.Series indexed by year (int), full historical EAD in millions
    last_pd_year : int   — most recent year with observed data

    Raises
    ------
    FileNotFoundError if no valid file found
    ValueError if year, PD, or EAD column not detected
    """
    candidates = []
    _SKIP_KEYS = {"ngfs", "ngfs_ct", "capital"}

    paths_to_try: Dict[str, str] = dict(upload_paths)
    _uploads_dir = Path(__file__).resolve().parent.parent.parent.parent / "uploads"
    if _uploads_dir.exists():
        for _p in _uploads_dir.iterdir():
            if _p.is_file() and _p.suffix.lower() in (".csv", ".xlsx", ".xls"):
                paths_to_try.setdefault(_p.name, str(_p))

    for key, path_str in paths_to_try.items():
        if key in _SKIP_KEYS:
            continue
        p = Path(str(path_str))
        if not p.exists():
            continue
        if any(kw in p.stem.lower() for kw in _CLIMATE_KW):
            continue
        try:
            if p.suffix.lower() == ".csv":
                df = _best_parse_csv(p)
            elif p.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(p)
            else:
                continue
        except Exception as e:
            LOG.warning("Cannot read %s: %s", p.name, e)
            continue
        if df is None or len(df.columns) < 2 or len(df) > 2000:
            continue
        candidates.append((len(df), p.name, df))

    if not candidates:
        raise FileNotFoundError(
            "Fichier historique introuvable dans les uploads. "
            "Veuillez uploader un fichier CSV/Excel contenant les colonnes "
            "year, PD et EAD (en millions)."
        )

    def _score(df):
        col_low = {str(c).lower().strip() for c in df.columns}
        has_pd  = any(any(kw in c for kw in _PD_KEYWORDS)  for c in col_low)
        has_ead = any(any(kw in c for kw in _EAD_KEYWORDS) for c in col_low)
        return (not has_pd, not has_ead, len(df))

    candidates.sort(key=lambda x: _score(x[2]))
    _, fname, df = candidates[0]
    LOG.info("Credit dataset selected: %s (score=%s)", fname, _score(df))

    col_lower = {c: c.lower().strip() for c in df.columns}

    # ── Detect year column ────────────────────────────────────────────────
    year_col = None
    for c, low in col_lower.items():
        if any(kw in low for kw in _YEAR_KEYWORDS):
            year_col = c
            break
    if year_col is None:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                vals = df[c].dropna()
                if len(vals) > 0 and 1950 < float(vals.mean()) < 2100:
                    year_col = c
                    break
    if year_col is None:
        raise ValueError("Colonne 'year' introuvable dans le fichier historique.")

    # ── Detect PD column ──────────────────────────────────────────────────
    pd_col = None
    for c, low in col_lower.items():
        if any(kw in low for kw in _PD_KEYWORDS) and c != year_col:
            pd_col = c
            break
    if pd_col is None:
        raise ValueError(
            "Colonne PD introuvable. Nommez-la 'pd', 'default_rate' ou 'taux_d'."
        )

    # ── Detect EAD column ────────────────────────────────────────────────
    ead_col = None
    for c, low in col_lower.items():
        if any(kw in low for kw in _EAD_KEYWORDS) and c not in (year_col, pd_col):
            ead_col = c
            break
    if ead_col is None:
        raise ValueError(
            "Colonne EAD introuvable dans le fichier historique. "
            "Nommez-la 'ead', 'exposure' ou 'encours' (valeurs en millions)."
        )

    # ── Parse year index ──────────────────────────────────────────────────
    base = df[[year_col, pd_col, ead_col]].copy()
    base[year_col] = pd.to_numeric(base[year_col], errors="coerce").astype("Int64")
    base = base.dropna().set_index(year_col).sort_index()
    base.index = base.index.astype(int)

    # ── Clean PD ─────────────────────────────────────────────────────────
    base[pd_col] = pd.to_numeric(base[pd_col], errors="coerce").astype(float)
    if base[pd_col].mean() > 1:
        base[pd_col] /= 100.0
    base[pd_col] = base[pd_col].clip(EPS, 1 - EPS)

    # ── Clean EAD ────────────────────────────────────────────────────────
    base[ead_col] = pd.to_numeric(base[ead_col], errors="coerce").astype(float)
    base = base.dropna()

    if len(base) == 0:
        raise ValueError("Aucune ligne valide après nettoyage du fichier historique.")

    pd_series  = base[pd_col].rename("pd")
    ead_series = base[ead_col].rename("ead")
    last_year  = int(pd_series.index.max())

    LOG.info(
        "Credit dataset loaded: %d obs (%d–%d), PD_TTC=%.4f, EAD_last=%.2f M",
        len(pd_series), int(pd_series.index.min()), last_year,
        pd_series.mean(), float(ead_series.iloc[-1]),
    )

    return pd_series, ead_series, last_year


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION C — FEATURE ENGINEERING  (Étape 1 de la description)           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class FeatureSpec:
    """Metadata for a single engineered feature — carries everything needed
    to re-apply the exact same transform + z-score during projection."""
    feature_name: str
    parent_var: str
    transform: str
    mu: float = 0.0
    sigma: float = 1.0
    expected_sign: int = 0
    corr_score: float = 0.0


def _apply_transform(
    series: pd.Series,
    transform: str,
    span: int = EWMA_SPAN,
) -> pd.Series:
    """Apply one transform to a raw macro series (NOT z-scored)."""
    if transform == "level":
        return series.copy()
    elif transform == "lag1":
        return series.shift(1)
    elif transform == "growth":
        denom = series.shift(1).abs().clip(lower=EPS)
        return series.diff() / denom
    elif transform == "cycle":
        trend = series.ewm(span=span, min_periods=1).mean()
        return series - trend
    else:
        raise ValueError(f"Unknown transform: {transform}")


def _generate_features(
    macro_df: pd.DataFrame,
    target: pd.Series,
    uni_idx: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, FeatureSpec]]:
    """
    Generate 4 z-scored features per macro variable, score each by
    correlation with PD, and return (feature_df, specs).

    z-score parameters (μ, σ) are saved in FeatureSpec for projection.
    """
    specs: Dict[str, FeatureSpec] = {}
    features: Dict[str, pd.Series] = {}

    for col in macro_df.columns:
        raw = macro_df[col].astype(float)
        indicator = uni_idx.get(col)
        exp_sign = indicator.expected_sign if indicator else 0

        for tfm in TRANSFORMS:
            fname = f"{col}__{tfm}"
            transformed = _apply_transform(raw, tfm)

            aligned = transformed.reindex(target.index)
            valid = aligned.dropna()
            if len(valid) < 5:
                continue

            mu = float(valid.mean())
            sigma = float(valid.std())
            if sigma < EPS:
                continue

            zscored = (aligned - mu) / sigma

            overlap = pd.concat([zscored.rename("x"), target.rename("y")],
                                axis=1).dropna()
            if len(overlap) < 5:
                continue
            try:
                r_pears = abs(pearsonr(overlap["x"], overlap["y"])[0])
                r_spear = abs(spearmanr(overlap["x"], overlap["y"])[0])
                corr = 0.5 * (r_pears + r_spear)
            except Exception:
                continue
            if np.isnan(corr):
                continue

            features[fname] = zscored
            specs[fname] = FeatureSpec(
                feature_name=fname,
                parent_var=col,
                transform=tfm,
                mu=mu,
                sigma=sigma,
                expected_sign=exp_sign,
                corr_score=corr,
            )

    if not features:
        raise RuntimeError(
            "Feature engineering: aucune feature générée. "
            "Vérifier la qualité et l'alignement des données macro."
        )

    out = pd.DataFrame(features)
    LOG.info("Feature engineering: %d features from %d raw vars",
             len(features), len(macro_df.columns))
    return out, specs


def _top_k_dedup(
    feature_df: pd.DataFrame,
    specs: Dict[str, FeatureSpec],
    k: int,
) -> Tuple[pd.DataFrame, Dict[str, FeatureSpec]]:
    """
    Select top-K features by |corr| with PD, keeping at most
    1 feature per parent variable (deduplication for diversity).
    """
    ranked = sorted(specs.values(), key=lambda s: -s.corr_score)
    seen_parents: set = set()
    selected: List[FeatureSpec] = []
    for s in ranked:
        if s.parent_var in seen_parents:
            continue
        seen_parents.add(s.parent_var)
        if s.corr_score < 0.10:
            LOG.debug(
                'Top-K dedup: featuré eliminé |corr|=%.3f < 0.10 seuil min',
                s.corr_score,
            )
            continue
        selected.append(s)
        if len(selected) >= k:
            break

    sel_names = [s.feature_name for s in selected]
    sel_specs = {s.feature_name: s for s in selected}

    LOG.info("Top-K dedup: %d features retained (1/parent, K=%d)",
             len(sel_names), k)
    for s in selected:
        LOG.info("  %-40s parent=%-25s tfm=%-6s |r|=%.3f sign=%+d",
                 s.feature_name, s.parent_var, s.transform,
                 s.corr_score, s.expected_sign)

    return feature_df[sel_names].copy(), sel_specs


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION D — TOURNOI DES 3 FAMILLES SATELLITES  (Étape 2)               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _fit_logit_ols(X: np.ndarray, y: np.ndarray) -> Optional[Dict]:
    """OLS on logit(PD). X must include constant column."""
    y_t = _logit(y)
    try:
        res = sm.OLS(y_t, X).fit(disp=0)
    except Exception:
        return None
    fitted_t = res.fittedvalues
    fitted = _sigmoid(fitted_t)
    return {"params": res.params, "fitted": fitted, "pvalues": res.pvalues}


def _fit_vasicek_ols(X: np.ndarray, y: np.ndarray) -> Optional[Dict]:
    """OLS on Φ⁻¹(PD). X must include constant column."""
    y_t = _probit(y)
    try:
        res = sm.OLS(y_t, X).fit(disp=0)
    except Exception:
        return None
    fitted_t = res.fittedvalues
    fitted = norm.cdf(fitted_t)
    return {"params": res.params, "fitted": fitted, "pvalues": res.pvalues}


def _beta_negloglik(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood of Beta regression with logit link on μ."""
    n_beta = X.shape[1]
    beta = params[:n_beta]
    log_phi = params[n_beta]

    mu = _sigmoid(X @ beta)
    mu = np.clip(mu, EPS, 1 - EPS)
    phi = np.exp(np.clip(log_phi, -10, 10))

    a = mu * phi
    b = (1 - mu) * phi

    ll = np.sum(
        gammaln(phi) - gammaln(a) - gammaln(b)
        + (a - 1) * np.log(np.clip(y, EPS, 1 - EPS))
        + (b - 1) * np.log(np.clip(1 - y, EPS, 1 - EPS))
    )
    return -ll


def _fit_beta_mle(X: np.ndarray, y: np.ndarray) -> Optional[Dict]:
    """
    Beta regression via MLE (Ferrari & Cribari-Neto, 2004).
    Logit link on μ, log link on φ (dispersion).
    """
    y_logit = _logit(y)
    try:
        init_beta = np.linalg.lstsq(X, y_logit, rcond=None)[0]
    except Exception:
        init_beta = np.zeros(X.shape[1])
    x0 = np.append(init_beta, 0.0)

    try:
        result = minimize(
            _beta_negloglik, x0, args=(X, y),
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if not result.success:
            return None
    except Exception:
        return None

    beta_hat = result.x[:X.shape[1]]
    fitted = _sigmoid(X @ beta_hat)
    return {"params": beta_hat, "fitted": fitted, "pvalues": None}


def _sign_violations(
    combo: Tuple[str, ...],
    coefs: np.ndarray,
    specs: Dict[str, FeatureSpec],
) -> Tuple[int, List[str]]:
    """Count how many coefficients violate the expected economic sign."""
    violations = []
    for col, c in zip(combo, coefs):
        exp = specs[col].expected_sign
        if exp == 0:
            continue
        if (exp > 0 and c < 0) or (exp < 0 and c > 0):
            violations.append(col)
    return len(violations), violations


_VERDICT_RANK: Dict[str, int] = {
    "VALIDATED":               0,
    "VALIDATED_WITH_WARNINGS": 1,
    "REJECTED":                2,
}


def _run_tournament(
    feature_df: pd.DataFrame,
    target: pd.Series,
    specs: Dict[str, FeatureSpec],
    n_obs: int,
    rho: Optional[float] = None,
) -> Tuple[Dict, List[Dict]]:
    """
    Run the full satellite tournament (cached — see _TOURNAMENT_CACHE):
      - Enumerate all valid combinations (Harrell rule)
      - For each combo × each family (logit, beta, vasicek): fit & score
      - Run PostEstimationValidator on every candidate
      - Rank by (verdict ASC, sign_violations ASC, R² DESC, RMSE ASC)
      - Return (best_model, full_leaderboard)
    """
    key = _tournament_cache_key(feature_df, target, specs, n_obs, rho)
    cached = _TOURNAMENT_CACHE.get(key)
    if cached is not None:
        _TOURNAMENT_CACHE.move_to_end(key)
        LOG.info("Tournament: cache hit (n_obs=%d) — skipping %d fits", n_obs,
                  0)
        return cached

    result = _run_tournament_impl(feature_df, target, specs, n_obs, rho)

    _TOURNAMENT_CACHE[key] = result
    _TOURNAMENT_CACHE.move_to_end(key)
    if len(_TOURNAMENT_CACHE) > _TOURNAMENT_CACHE_MAX:
        _TOURNAMENT_CACHE.popitem(last=False)
    return result


def _run_tournament_impl(
    feature_df: pd.DataFrame,
    target: pd.Series,
    specs: Dict[str, FeatureSpec],
    n_obs: int,
    rho: Optional[float] = None,
) -> Tuple[Dict, List[Dict]]:
    # ── Harrell rule: max combination size ────────────────────────────────
    max_combo = max(2, n_obs // HARRELL_OBS_PER_PARAM - 1)
    max_combo = min(max_combo, len(feature_df.columns), 3)

    LOG.info("Tournament: n_obs=%d → Harrell max_combo=%d", n_obs, max_combo)

    # ── Prepare aligned data ──────────────────────────────────────────────
    joined = feature_df.join(target.rename("__pd__"), how="inner").dropna()
    y = joined["__pd__"].values.astype(float)
    y = np.clip(y, EPS, 1 - EPS)
    macro_cols = [c for c in joined.columns if c != "__pd__"]
    n = len(y)

    # ── Enumerate combinations ────────────────────────────────────────────
    all_combos: List[Tuple[str, ...]] = []
    for k in range(2, max_combo + 1):
        all_combos.extend(combinations(macro_cols, k))

    LOG.info("Tournament: %d combos × 3 families = %d fits",
             len(all_combos), len(all_combos) * 3)

    # ── Fit all ───────────────────────────────────────────────────────────
    leaderboard: List[Dict] = []

    for combo in all_combos:
        X_raw = joined[list(combo)].values.astype(float)
        X = sm.add_constant(X_raw, has_constant="add")

        # ── Garde VIF (max 10.0) ──────────────────────────────────────────────
        try:
            from statsmodels.stats.outliers_influence import (
                variance_inflation_factor as _vif_fn,
            )
            if X_raw.shape[0] > X_raw.shape[1] + 1:
                _cr_max_vif = max(
                    _vif_fn(X_raw, j) for j in range(X_raw.shape[1])
                )
                if _cr_max_vif > 10.0:
                    LOG.debug("Tournoi crédit: %s rejeté — VIF=%.1f > 10.0",
                              combo, _cr_max_vif)
                    continue
        except Exception:
            pass


        if n < len(combo) + 2:
            continue

        for family in ("logit", "beta", "vasicek"):
            try:
                if family == "logit":
                    fit = _fit_logit_ols(X, y)
                elif family == "beta":
                    fit = _fit_beta_mle(X, y)
                elif family == "vasicek":
                    fit = _fit_vasicek_ols(X, y)
                else:
                    continue

                if fit is None:
                    continue

                fitted    = fit["fitted"]
                # Force numpy arrays — statsmodels returns pandas Series
                # which breaks integer-position indexing after slicing
                params    = np.asarray(fit["params"], dtype=float)
                _pv_raw   = fit.get("pvalues")
                pvalues   = np.asarray(_pv_raw, dtype=float) if _pv_raw is not None else None
                intercept = float(params[0])
                coefs     = params[1:]   # guaranteed 1-D numpy array

                r2_val   = _r2(y, fitted)
                rmse_val = _rmse(y, fitted)
                rss      = float(np.sum((y - fitted) ** 2))
                n_params = len(params)
                aic_val  = _aic(n, rss, n_params)
                bic_val  = _bic(n, rss, n_params)

                n_viol, viol_vars = _sign_violations(combo, coefs, specs)

                # p-values: index 0 = intercept, 1..k = macro vars
                pv_intercept = (float(pvalues[0])
                                if pvalues is not None else None)
                pv_coefs     = (
                    {var: round(float(pvalues[j + 1]), 4)
                     for j, var in enumerate(combo)}
                    if pvalues is not None else {}
                )

                # Filtre p-value : au moins 1 coef macro doit être p < 0.20
                if pvalues is not None and len(pvalues) > 1:
                    _macro_pvs = [float(pvalues[j+1]) for j in range(len(combo))
                                  if not (pvalues[j+1] != pvalues[j+1])]
                    if _macro_pvs and all(pv > 0.20 for pv in _macro_pvs):
                        LOG.debug(
                            'Tournoi crédit: %s %s rejeté — aucun coef p<0.20',
                            family, combo,
                        )
                        continue

                # Filtre coef ≈ 0 : variable macro sans contribution au modèle
                # Un coef nul indique une collinéarité numérique ou une redondance
                # parfaite — la variable ne doit pas entrer dans le satellite crédit.
                _zero_coef_cr = [
                    var for j, var in enumerate(combo)
                    if abs(float(coefs[j])) < 1e-8
                ]
                if _zero_coef_cr:
                    LOG.debug(
                        'Tournoi crédit: %s %s rejeté — coef≈0 pour %s',
                        family, combo, _zero_coef_cr,
                    )
                    continue

                leaderboard.append({
                    "family":         family,
                    "combo":          list(combo),
                    "intercept":      round(intercept, 6),
                    "pv_intercept":   round(pv_intercept, 4) if pv_intercept is not None else None,
                    "coefs":          {var: round(float(coefs[j]), 6)
                                       for j, var in enumerate(combo)},
                    "pv_coefs":       pv_coefs,
                    "r2":             r2_val,
                    "rmse":           rmse_val,
                    "aic":            aic_val,
                    "bic":            bic_val,
                    "sign_violations": n_viol,
                    "violating_vars": viol_vars,
                    "n_obs":          n,
                    "fitted_values":  [round(float(v), 6) for v in np.asarray(fitted).tolist()],
                    "actual_values":  [round(float(v), 6) for v in y.tolist()],
                    "obs_years":      [int(yr) for yr in joined.index.tolist()],
                })

            except Exception as e:
                LOG.debug("Fit failed %s %s: %s", family, combo, e)

    if not leaderboard:
        raise RuntimeError(
            "Tournament: aucun modèle satellite n'a pu être estimé. "
            "Vérifier la qualité des données."
        )

    # ── Post-Estimation Validation on every candidate ────────────────────
    try:
        from app.modules.core.post_estimation_validator import (
            PostEstimationValidator as _PEV, RiskOutput as _RO,
        )
        for _entry in leaderboard:
            try:
                _yt = np.asarray(_entry["actual_values"], dtype=float)
                _yp = np.asarray(_entry["fitted_values"], dtype=float)
                _ro = _RO(
                    module     = "credit",
                    model_type = _entry["family"],
                    y_true     = _yt,
                    y_pred     = _yp,
                    residuals  = _yt - _yp,
                    n_obs      = int(_entry["n_obs"]),
                    k_params   = len(_entry["combo"]),
                    rho        = rho,
                )
                _vr = _PEV(_ro).run()
                _entry["verdict"]       = _vr.verdict
                _entry["n_pass"]        = _vr.n_pass
                _entry["n_warn"]        = _vr.n_warn
                _entry["n_fail"]        = _vr.n_fail
                _entry["_vreport_dict"] = _vr.to_dict()
            except Exception as _ve:
                LOG.debug(
                    "PostEstimation candidate [%s/%s]: %s",
                    _entry["family"], _entry["combo"], _ve,
                )
                _entry["verdict"]       = "VALIDATED_WITH_WARNINGS"
                _entry["n_pass"]        = 0
                _entry["n_warn"]        = 1
                _entry["n_fail"]        = 0
                _entry["_vreport_dict"] = {}
    except ImportError:
        LOG.warning("PostEstimationValidator non disponible — tri sans verdict.")
        for _entry in leaderboard:
            _entry.setdefault("verdict",        "VALIDATED_WITH_WARNINGS")
            _entry.setdefault("n_pass",         0)
            _entry.setdefault("n_warn",         0)
            _entry.setdefault("n_fail",         0)
            _entry.setdefault("_vreport_dict",  {})

    # ── Rank: verdict ASC → sign_violations ASC → R² DESC → RMSE ASC ────
    leaderboard.sort(key=lambda x: (
        _VERDICT_RANK.get(x.get("verdict", "VALIDATED_WITH_WARNINGS"), 1),
        x["sign_violations"], -x["r2"], x["rmse"],
    ))

    best = leaderboard[0]
    LOG.info(
        "★ Winner: %s | vars=%s | R²=%.3f RMSE=%.5f | sign_viol=%d | verdict=%s",
        best["family"], best["combo"], best["r2"], best["rmse"],
        best["sign_violations"], best.get("verdict", "N/A"),
    )

    return best, leaderboard


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION E — PROJECTION MACRO + CHOCS SCÉNARIO  (Étape 3)              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _extrapolate_ar1(series: pd.Series, horizon_years: List[int]) -> pd.Series:
    """
    Simple AR(1) extrapolation for variables that lack WEO projections.
    Fallback for WB-only indicators.
    """
    y = series.dropna().values.astype(float)
    if len(y) < 3:
        last_val = float(y[-1]) if len(y) > 0 else 0.0
        return pd.Series(last_val, index=horizon_years)

    y_lag = y[:-1]
    y_now = y[1:]
    phi = np.corrcoef(y_lag, y_now)[0, 1]
    phi = np.clip(phi, -0.95, 0.95)
    c = np.mean(y_now) - phi * np.mean(y_lag)

    projected = []
    last = float(y[-1])
    for _ in horizon_years:
        nxt = c + phi * last
        projected.append(nxt)
        last = nxt

    return pd.Series(projected, index=horizon_years)


def _build_scenario_macro(
    macro_full: pd.DataFrame,
    last_pd_year: int,
    horizon: List[int],
    shocks: Dict[str, float],
    uni_idx: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build a full raw macro DataFrame covering historical + projection
    for a specific scenario level.

    Baseline (shocks={}):  WEO projections where available, AR(1) otherwise.
    Adverse / Severe:      baseline + scenario shocks with progressive ramp.
    """
    hist = macro_full.loc[macro_full.index <= last_pd_year].copy()

    # ── Baseline projection : WEO pour vars couvertes, macro_projector pour les autres ──
    # Grouper les variables WB-only et les projeter en batch (AR/VAR/VECM selon n_obs)
    proj_dict: Dict[str, pd.Series] = {}
    weo_covered:  List[str] = []
    wb_only_cols: List[str] = []

    for col in hist.columns:
        future_data = macro_full.loc[macro_full.index.isin(horizon), col].dropna()
        if len(future_data) >= len(horizon) * 0.5:
            proj = future_data.reindex(horizon).ffill().bfill()
            if proj.isna().any():
                last_hist = (hist[col].dropna().iloc[-1]
                             if not hist[col].dropna().empty else 0.0)
                proj = proj.fillna(last_hist)
            proj_dict[col] = proj
            weo_covered.append(col)
        else:
            wb_only_cols.append(col)

    # Projeter les vars WB-only via macro_projector (AR/VAR/VECM selon stationnarité)
    if wb_only_cols:
        valid_wb = [c for c in wb_only_cols if not hist[c].dropna().empty]
        try:
            from ..core.macro_projector import project_macro as _proj_macro
            if valid_wb:
                _mp_res = _proj_macro(
                    historical  = hist[valid_wb],
                    variables   = valid_wb,
                    horizon_years = horizon,
                )
                for col in valid_wb:
                    proj_dict[col] = _mp_res.projected[col]
                LOG.info(
                    "macro_projector crédit : %d vars projetées (WB-only) via %s",
                    len(valid_wb), _mp_res.model_used,
                )
        except Exception as _mp_err:
            LOG.warning(
                "macro_projector crédit échoué (%s) — fallback AR(1)", _mp_err
            )
            for col in valid_wb:
                proj_dict[col] = _extrapolate_ar1(hist[col].dropna(), horizon)
        # Vars sans historique → flat 0
        for col in wb_only_cols:
            if col not in proj_dict:
                proj_dict[col] = pd.Series(0.0, index=horizon)

    proj_df = pd.DataFrame(proj_dict, index=horizon)

    # ── Apply scenario shocks (adverse / severe only) ─────────────────────
    if shocks:
        n_years = len(horizon)
        ramp = np.linspace(0.3, 1.0, n_years) ** 0.7

        directly_shocked: set = set()

        # ── Pass 1: direct shocks (variables with shock_key) ──────────
        for col in proj_df.columns:
            indicator = uni_idx.get(col)
            if indicator is None or indicator.shock_key is None:
                continue
            if indicator.shock_unit is None:
                continue

            raw_shock = shocks.get(indicator.shock_key, 0.0)
            if raw_shock == 0.0:
                continue

            base_values = proj_df[col].values.astype(float)
            delta = np.zeros(n_years)

            if indicator.shock_unit == "additive_pp":
                delta = raw_shock * 100.0 * ramp

            elif indicator.shock_unit == "additive_pct":
                delta = raw_shock * 100.0 * ramp

            elif indicator.shock_unit == "bps_to_pp":
                delta = (raw_shock / 100.0) * ramp

            elif indicator.shock_unit == "multiplicative":
                delta = raw_shock * base_values * ramp

            elif indicator.shock_unit == "multiplicative_neg":
                delta = -raw_shock * np.abs(base_values) * ramp

            proj_df[col] = base_values + delta
            directly_shocked.add(col)

            LOG.debug("Direct shock %s: %s += %.3f..%.3f (unit=%s)",
                      col, indicator.shock_key, delta[0], delta[-1],
                      indicator.shock_unit)

        # Pass 2 (elasticity-based propagation via GDP anchor) supprimée :
        # les chocs sont entièrement définis par le scénario choisi
        # (historique → profil de crise réel, paramétrique → saisie utilisateur).
        # Propager des chocs implicites violerait le caractère générique.

    # ── Concatenate historical + projected ────────────────────────────────
    full = pd.concat([hist, proj_df]).sort_index()
    full = full[~full.index.duplicated(keep="last")]

    return full


def _build_scenario_macro_from_deltas(
    macro_full: pd.DataFrame,
    last_pd_year: int,
    horizon: List[int],
    deltas: Dict[str, float],
) -> pd.DataFrame:
    """
    Build scenario macro by applying historical crisis deltas directly
    to the baseline projection. Unlike _build_scenario_macro (which uses
    shock_key routing), this function applies deltas in the raw variable
    units — because historical_scenarios.py computes them that way.

    Δvar is applied with a progressive ramp over the horizon.

    Parameters
    ----------
    macro_full : pd.DataFrame   Full macro data (historical + WEO projections)
    last_pd_year : int          Last year of observed PD
    horizon : List[int]         Projection years
    deltas : Dict[str, float]   {variable_name: Δvar} from compute_historical_shocks
    """
    # First build the baseline (no shocks)
    hist = macro_full.loc[macro_full.index <= last_pd_year].copy()

    proj_dict: Dict[str, pd.Series] = {}
    for col in hist.columns:
        future_data = macro_full.loc[macro_full.index.isin(horizon), col].dropna()

        if len(future_data) >= len(horizon) * 0.5:
            proj = future_data.reindex(horizon).ffill().bfill()
            if proj.isna().any():
                last_hist = (hist[col].dropna().iloc[-1]
                             if not hist[col].dropna().empty else 0.0)
                proj = proj.fillna(last_hist)
            proj_dict[col] = proj
        else:
            proj_dict[col] = _extrapolate_ar1(hist[col].dropna(), horizon)

    proj_df = pd.DataFrame(proj_dict, index=horizon)

    # Apply deltas with progressive ramp
    n_years = len(horizon)
    ramp = np.linspace(0.3, 1.0, n_years) ** 0.7

    for col in proj_df.columns:
        if col not in deltas:
            continue
        delta = deltas[col]
        if abs(delta) < 1e-12:
            continue

        base_values = proj_df[col].values.astype(float)
        proj_df[col] = base_values + delta * ramp

    full = pd.concat([hist, proj_df]).sort_index()
    full = full[~full.index.duplicated(keep="last")]

    return full


def _project_features(
    macro_scenario: pd.DataFrame,
    best_combo: List[str],
    specs: Dict[str, FeatureSpec],
    horizon: List[int],
) -> pd.DataFrame:
    """
    Apply the same transforms + z-score (with training μ, σ) to
    the scenario macro, producing feature values for projection.
    """
    feature_proj = pd.DataFrame(index=horizon)

    for fname in best_combo:
        spec = specs[fname]
        raw = macro_scenario[spec.parent_var].astype(float)

        # The "growth" and "lag1" transforms require the value at (t-1) to
        # compute the feature at horizon[0]. If the last historical year has
        # no data yet (e.g. WB/WEO not yet published), that denominator is
        # NaN and the entire projection collapses to NaN. Forward-fill so the
        # last *available* observation is used — the standard treatment for
        # the endpoint problem in macro stress-testing (ECB, IMF FSAP, BCBS).
        raw = raw.ffill()

        transformed = _apply_transform(raw, spec.transform, span=EWMA_SPAN)

        zscored = (transformed - spec.mu) / max(spec.sigma, EPS)

        feature_proj[fname] = zscored.reindex(horizon)

    return feature_proj


def _predict_pd(
    feature_proj: pd.DataFrame,
    best: Dict,
) -> np.ndarray:
    """Apply the winning satellite model to produce PD_PIT."""
    X_raw = feature_proj.values.astype(float)
    X_raw = np.nan_to_num(X_raw, nan=0.0)

    intercept = float(best["intercept"])
    # coefs is stored as {var: coef} — extract in feature_proj column order
    coefs_data = best["coefs"]
    if isinstance(coefs_data, dict):
        coefs = np.array([coefs_data[col] for col in feature_proj.columns],
                         dtype=float)
    else:
        coefs = np.array(coefs_data, dtype=float)
    linear = intercept + X_raw @ coefs

    family = best["family"]
    if family in ("logit", "beta"):
        pd_pit = _sigmoid(linear)
    elif family == "vasicek":
        pd_pit = norm.cdf(linear)
    else:
        raise ValueError(f"Unknown family: {family}")

    return np.clip(pd_pit, 1e-5, 0.99)


def _inject_z_shock(
    features: pd.DataFrame,
    combo: List[str],
    specs: Dict[str, "FeatureSpec"],
    shocks: Dict[str, float],
    uni_idx: Dict[str, Any],
    n_years: int,
) -> pd.DataFrame:
    """
    Inject macro shocks directly in z-score space for features whose
    transform (growth, cycle, lag1) attenuates a persistent level shock.

    Root cause: if a variable already represents a growth rate (e.g. GDP
    growth %) and we apply a persistent additive shock of -3 pp, the
    "growth" transform takes first-differences → the shock appears only in
    year 1, then cancels out. Same problem with "cycle" (EWMA gap) and
    "lag1" (one-period shift loses the persistent component after 1 year).

    Fix: for each such feature, compute the equivalent z-score delta
         Δz = shock_in_feature_units / σ_feature
    and add it with the same progressive ramp used in the raw macro layer.
    This is only applied for PATH A (parametric shocks via shock_key).
    """
    ramp = np.linspace(0.3, 1.0, n_years) ** 0.7
    out = features.copy()

    for fname in combo:
        spec = specs[fname]
        if spec.transform not in ("growth", "cycle", "lag1"):
            continue  # "level" features already transmit the shock correctly

        indicator = uni_idx.get(spec.parent_var)
        if indicator is None or indicator.shock_key is None:
            continue
        raw_shock = shocks.get(indicator.shock_key, 0.0)
        if raw_shock == 0.0:
            continue

        # Convert raw shock (in scenario units) to feature units
        if indicator.shock_unit == "additive_pp":
            shock_units = raw_shock * 100.0      # fraction → percentage points
        elif indicator.shock_unit == "additive_pct":
            shock_units = raw_shock * 100.0
        elif indicator.shock_unit == "bps_to_pp":
            shock_units = raw_shock / 100.0      # basis points → percentage points
        else:
            continue  # multiplicative shocks too complex — skip

        # z-score delta: how much the feature z-score changes per unit of persistent shock
        z_delta = shock_units / max(spec.sigma, EPS)
        out[fname] = out[fname].values + z_delta * ramp
        LOG.debug(
            "_inject_z_shock: %-38s shock=%.3f σ=%.3f Δz=%.4f (transform=%s)",
            fname, shock_units, spec.sigma, z_delta, spec.transform,
        )

    return out


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION F — FRYE-JACOBS LGD  (Étape 4 de la description)              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _asset_correlation_irb(pd_ttc: float) -> float:
    """
    Basel III IRB asset correlation ρ (BCBS CRE31 §31.28).
    Interpolates between 12% (high PD) and 24% (low PD).
    """
    pd_c = np.clip(pd_ttc, EPS, 1 - EPS)
    factor = (1 - np.exp(-50 * pd_c)) / (1 - np.exp(-50))
    return float(0.12 * factor + 0.24 * (1 - factor))


def _calibrate_fj_k(pd_ttc: float, lgd_ttc: float, rho: float) -> float:
    """
    Calibrate the Frye-Jacobs constant k by numerically solving:
        Φ( (Φ⁻¹(PD_TTC) − k) / √(1−ρ) ) / PD_TTC = LGD_TTC

    Uses scipy.optimize.brentq (bracketed root-finding).
    Exact formulation from Frye & Jacobs (2012).
    """
    pd_c = np.clip(pd_ttc, EPS, 1 - EPS)
    lgd_c = np.clip(lgd_ttc, EPS, 1 - EPS)
    sqrt_1mr = np.sqrt(max(1 - rho, EPS))
    z_pd = norm.ppf(pd_c)

    def equation(k_val):
        numerator = norm.cdf((z_pd - k_val) / sqrt_1mr)
        return numerator / pd_c - lgd_c

    try:
        k = brentq(equation, -10.0, 10.0, xtol=1e-12)
    except ValueError:
        LOG.warning(
            "Frye-Jacobs brentq failed to converge — using closed-form approximation. "
            "INPUT: pd_ttc=%.6f, lgd_ttc=%.6f, rho=%.6f. "
            "CONSEQUENCE: LGD_PIT projections may be inaccurate. "
            "Check that pd_ttc and lgd_ttc are not extreme values.",
            pd_ttc, lgd_ttc, rho,
        )
        target = lgd_c * pd_c
        target = np.clip(target, EPS, 1 - EPS)
        k = z_pd - norm.ppf(target) * sqrt_1mr

    return float(k)


def _frye_jacobs_lgd(
    pd_pit: np.ndarray,
    rho: float,
    k: float,
    floor: float = 0.05,
    cap: float = 0.99,
) -> np.ndarray:
    """
    LGD_PIT(t) = Φ( (Φ⁻¹(PD_PIT(t)) − k) / √(1−ρ) ) / PD_PIT(t)
    Clipped to [floor, cap] for stability.
    """
    pd_c = np.clip(pd_pit, EPS, 1 - EPS)
    sqrt_1mr = np.sqrt(max(1 - rho, EPS))
    z_pd = norm.ppf(pd_c)
    numerator = norm.cdf((z_pd - k) / sqrt_1mr)
    lgd = numerator / pd_c
    return np.clip(lgd, floor, cap)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION G — WRAPPER PRINCIPAL                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _build_capital_params(ead_series: pd.Series, user_params: dict) -> dict:
    """
    Build the CreditCapitalEngine params dict from the historical EAD series.

    Computes EAD CAGR and annual-std, then injects ead_growth_rate/adverse/severe
    via setdefault so user overrides are respected.

    Shared by credit/wrapper.py (conventional path) and ngfs_credit_engine.py
    (climate path) — single source of truth for EAD growth calibration.
    """
    annual_growth = ead_series.pct_change().dropna()
    n = len(ead_series)
    if n >= 2 and float(ead_series.iloc[0]) > 0:
        cagr = float(
            (ead_series.iloc[-1] / ead_series.iloc[0]) ** (1.0 / (n - 1)) - 1.0
        )
    else:
        cagr = 0.0
    std = float(annual_growth.std()) if len(annual_growth) >= 2 else 0.0

    params = dict(user_params)
    params.setdefault("ead_growth_rate",    cagr)
    params.setdefault("ead_growth_adverse", cagr - std)
    params.setdefault("ead_growth_severe",  cagr - 2.0 * std)
    return params


def _resolve_crisis_safe(crisis_name: str) -> Any:
    """
    Resolve a CrisisDefinition from CRISIS_CATALOG.

    The choice of crisis is left to the user via the ``crisis_name`` param.
    If the name is absent or unknown, a country-neutral global crisis is used
    as a fallback (no automatic country-based selection).

    Priority:
      1. Explicit crisis_name, if present in CRISIS_CATALOG
      2. Most recent global crisis (country-neutral fallback)
      3. Any crisis in CRISIS_CATALOG (absolute last resort)

    Raises RuntimeError only if CRISIS_CATALOG is completely empty.
    """
    # 1 — explicit name (valid)
    if crisis_name and crisis_name in CRISIS_CATALOG:
        LOG.info("Crisis resolved (explicit): %s", crisis_name)
        return CRISIS_CATALOG[crisis_name]

    if crisis_name:
        available = ", ".join(CRISIS_CATALOG.keys())
        LOG.warning(
            "crisis_name='%s' introuvable dans le catalogue. "
            "Disponibles : %s. Repli sur la crise globale la plus récente.",
            crisis_name, available,
        )

    # 2 — most recent global crisis (country-neutral)
    global_matches = [
        (k, c) for k, c in CRISIS_CATALOG.items() if c.country == "global"
    ]
    if global_matches:
        best_k, best_c = max(global_matches, key=lambda kc: kc[1].t_peak)
        LOG.info(
            "Crisis fallback (global, le plus récent) : '%s'. "
            "Définissez le paramètre 'crisis_name' pour choisir explicitement.",
            best_k,
        )
        return best_c

    # 3 — any crisis (absolute last resort)
    if CRISIS_CATALOG:
        best_k = max(CRISIS_CATALOG, key=lambda k: CRISIS_CATALOG[k].t_peak)
        LOG.info("Crisis fallback (dernier recours) : %s", best_k)
        return CRISIS_CATALOG[best_k]

    raise RuntimeError(
        "CRISIS_CATALOG est vide — impossible d'exécuter un scénario "
        "de type historique ou fallback."
    )


def get_projection_series(charts_data: dict, variable: str = "pd"):
    """
    Adapts already-computed credit trajectories into a ProjectionSeries, for
    the climate dashboard's Row 3 (PD) and Row 5 (CAR / Tier 1 ratio) charts.

    Additive only — does not touch run()/_run_tournament()/any existing
    method; reads charts_data a CreditModuleWrapper.run() call already
    produced (pd_trajectories or capital_trajectories).

    Parameters
    ----------
    charts_data : the "charts_data" dict of a credit PlatformResult (or the
                  climate module's own charts_data, which carries the same
                  "pd_trajectories" key).
    variable    : "pd" (from pd_trajectories) or any capital_trajectories
                  metric name — "car", "tier1_ratio", "rwa_stressed", etc.

    Returns
    -------
    ProjectionSeries | None (None if the underlying trajectory is absent).
    """
    from ..base import projection_series_from_trajectory_dict

    if variable == "pd":
        traj = (charts_data or {}).get("pd_trajectories", {})
        return projection_series_from_trajectory_dict(traj, "PD", "%")

    cap_traj = (charts_data or {}).get("capital_trajectories", {})
    traj = cap_traj.get(variable, {})
    _units = {"car": "%", "tier1_ratio": "%", "cet1_ratio": "%",
              "leverage_ratio": "%", "rwa_stressed": "M"}
    return projection_series_from_trajectory_dict(
        traj, variable, _units.get(variable, ""),
    )


class CreditModuleWrapper(PlatformModule):
    module_id = "credit"
    module_label = "Risque de Crédit"
    is_placeholder = False

    def run(self) -> Dict[str, PlatformResult]:
        p = self.params
        uni_idx = universe_index()

        # ── Parameters ───────────────────────────────────────────────────
        _COUNTRY_NAME_TO_ISO2 = {
            "egypt": "EG", "france": "FR", "germany": "DE",
            "united kingdom": "GB", "mena": "EG", "north africa": "MA",
            "europe": "DE", "gcc": "SA", "morocco": "MA", "tunisia": "TN",
            "nigeria": "NG", "south africa": "ZA", "senegal": "SN",
        }
        country = str(p.get("country_iso2", "")).strip().upper()
        if not country:
            # Fallback: derive ISO2 from full country name if provided
            _full = str(p.get("country", "")).strip().lower()
            country = _COUNTRY_NAME_TO_ISO2.get(_full, "").upper()
        if not country:
            raise ValueError(
                "Le paramètre 'country_iso2' est requis. "
                "Exemples : 'EG' (Égypte), 'MA' (Maroc), 'TN' (Tunisie), "
                "'NG' (Nigeria), 'ZA' (Afrique du Sud)."
            )
        lgd_ttc       = float(p.get("lgd_ttc", 0.45))
        history_start = int(p.get("history_start", 1990))
        horizon_start = int(p.get("horizon_start", 2025))
        horizon_end   = int(p.get("horizon_end", 2028))
        cache_dir     = str(p.get("cache_dir", "data_cache"))
        cache_ttl     = int(p.get("cache_ttl_days", 30))
        crisis_name   = str(p.get("crisis_name", ""))

        # ══════════════════════════════════════════════════════════════════
        # STEP 0 — LOAD PD + EAD + FETCH MACRO (100% API)
        # ══════════════════════════════════════════════════════════════════

        pd_series, ead_series, last_pd_year = _load_credit_dataset(self.upload_paths)
        ead_M  = float(ead_series.iloc[-1])
        pd_ttc = float(pd_series.mean())

        # Recalculate horizon to never overlap with historical data.
        # If the dataset already contains horizon_start (e.g. 2025 in both history
        # and projection), 2025 would appear twice in every chart and time_series.
        # Enforce: horizon starts strictly after last_pd_year.
        if horizon_start <= last_pd_year:
            LOG.warning(
                "horizon_start=%d ≤ last_pd_year=%d — adjusting horizon to start "
                "at %d to avoid duplicate years in projections.",
                horizon_start, last_pd_year, last_pd_year + 1,
            )
            horizon_start = last_pd_year + 1
        horizon = list(range(horizon_start, horizon_end))

        # ── Historical EAD growth rates (data-driven, replaces hardcoded 5%) ──
        _ead_annual_growth = ead_series.pct_change().dropna()
        n_ead = len(ead_series)
        if n_ead >= 2 and float(ead_series.iloc[0]) > 0:
            _ead_cagr = float(
                (ead_series.iloc[-1] / ead_series.iloc[0]) ** (1.0 / (n_ead - 1)) - 1.0
            )
        else:
            _ead_cagr = 0.0
        _ead_growth_std = float(_ead_annual_growth.std()) if len(_ead_annual_growth) >= 2 else 0.0

        LOG.info(
            "EAD historical CAGR=%.4f (%.2f%%), annual_std=%.4f — "
            "derived from %d observations (%d–%d)",
            _ead_cagr, _ead_cagr * 100, _ead_growth_std,
            n_ead, int(ead_series.index.min()), int(ead_series.index.max()),
        )

        macro_full, fetch_report = fetch_credit_macro(
            country=country, start_year=history_start,
            cache_dir=cache_dir, cache_ttl_days=cache_ttl,
        )
        if macro_full.empty:
            raise RuntimeError(
                f"Aucune donnée macro récupérée pour '{country}'.")

        macro_hist = macro_full.loc[macro_full.index <= last_pd_year].copy()

        # Align PD + macro on common years
        target = pd_series.rename("pd")
        joined = macro_hist.join(target, how="inner").dropna(subset=["pd"])
        if len(joined) < 5:
            raise ValueError(
                f"Seulement {len(joined)} années communes entre PD et macro. "
                "Minimum requis : 5."
            )

        macro_aligned = joined.drop(columns=["pd"])
        target_aligned = joined["pd"]
        n_obs = len(target_aligned)
        LOG.info("Aligned PD × macro: %d years (%d–%d)",
                 n_obs, int(joined.index.min()), int(joined.index.max()))

        # ══════════════════════════════════════════════════════════════════
        # STEP 0.5 — PRE-TESTS MACRO (filtre pool candidat)
        # ══════════════════════════════════════════════════════════════════

        from ..core.macro_pre_tests import run_macro_pre_tests

        pre_test_result = run_macro_pre_tests(  # Blocs 0-5 actifs
            macro_df=macro_aligned,
            pd_series=target_aligned,
            universe_idx=uni_idx,
        )

        if not pre_test_result.accepted:
            raise RuntimeError(
                f"Aucune variable macro n'a passé les pré-tests. "
                f"Rejetées : {pre_test_result.rejected}"
            )

        macro_aligned = macro_aligned[pre_test_result.accepted]
        LOG.info("Pre-tests: %d/%d vars accepted → pool filtré",
                 len(pre_test_result.accepted),
                 len(pre_test_result.accepted) + len(pre_test_result.rejected))

        # ══════════════════════════════════════════════════════════════════
        # STEP 1 — FEATURE ENGINEERING + VARIABLE SELECTION
        # ══════════════════════════════════════════════════════════════════

        feature_df, all_specs = _generate_features(
            macro_aligned, target_aligned, uni_idx)

        k_eff = min(15, max(n_obs - 2, 3))
        topk_df, topk_specs = _top_k_dedup(feature_df, all_specs, k=k_eff)

        # ══════════════════════════════════════════════════════════════════
        # STEP 2 — SATELLITE TOURNAMENT (3 families × all combos)
        # ══════════════════════════════════════════════════════════════════

        # rho computed here (not in STEP 3) so PostEstimationValidator can
        # use it for Vasicek B2.C checks inside the tournament loop.
        rho = _asset_correlation_irb(pd_ttc)

        best, leaderboard = _run_tournament(
            topk_df, target_aligned, topk_specs, n_obs, rho=rho)

        # ── Model selection: default = validated rank 0; override = forced rank ─
        forced_rank = p.get("forced_model_rank")
        selected_lb_idx      = 0
        _user_forced_rejected = False

        if forced_rank is None:
            # leaderboard[0] is never REJECTED unless ALL candidates are REJECTED.
            if all(e.get("verdict") == "REJECTED" for e in leaderboard):
                raise RuntimeError(
                    f"NO_VALID_MODEL: les {len(leaderboard)} candidats du tournoi "
                    "sont tous REJECTED par PostEstimationValidator. "
                    "Vérifier la qualité ou la quantité des données historiques."
                )
        else:
            try:
                r = int(forced_rank)
                if 0 <= r < len(leaderboard):
                    best = leaderboard[r]
                    selected_lb_idx = r
                    if best.get("verdict") == "REJECTED":
                        _user_forced_rejected = True
                        LOG.warning(
                            "forced_model_rank=%d pointe vers un modèle REJECTED "
                            "(%s %s) — projection lancée par choix explicite.",
                            r, best["family"], best.get("combo", []),
                        )
                    LOG.info(
                        "Modèle forcé au rang %d : %s %s (verdict=%s)",
                        r, best["family"], best.get("combo", []),
                        best.get("verdict", "N/A"),
                    )
                else:
                    LOG.warning(
                        "forced_model_rank=%d hors limites (0–%d) — rang 0 utilisé",
                        r, len(leaderboard) - 1,
                    )
            except (TypeError, ValueError):
                LOG.warning("forced_model_rank invalide (%r) — ignoré", forced_rank)

        best_combo = best["combo"]
        best_specs = {f: topk_specs[f] for f in best_combo}

        # ══════════════════════════════════════════════════════════════════
        # STEP 3 — FRYE-JACOBS CALIBRATION (once, shared across scenarios)
        # ══════════════════════════════════════════════════════════════════

        # rho already computed before tournament (STEP 2 preamble)
        fj_k = _calibrate_fj_k(pd_ttc, lgd_ttc, rho)

        LOG.info("Frye-Jacobs: PD_TTC=%.4f, LGD_TTC=%.2f, ρ=%.4f, k=%.4f",
                 pd_ttc, lgd_ttc, rho, fj_k)

        # ── Validation report for charts_data (already computed in tournament) ─
        _credit_vreport_dict: dict = best.get("_vreport_dict", {})
        LOG.info(
            "PostEstimation [credit/%s] rang %d: %d PASS %d WARN %d FAIL — %s",
            best.get("family", "?"), selected_lb_idx,
            best.get("n_pass", 0), best.get("n_warn", 0), best.get("n_fail", 0),
            best.get("verdict", "N/A"),
        )

        # ══════════════════════════════════════════════════════════════════
        # STEP 4 — RANKING TABLE
        # ══════════════════════════════════════════════════════════════════

        ranking = [
            {
                "rang":            i + 1,
                "is_selected":     i == selected_lb_idx,
                "modele":          l["family"],
                "variables":       l["combo"],
                "intercept":       l["intercept"],
                "pv_intercept":    l["pv_intercept"],
                "coefs":           l["coefs"],
                "pv_coefs":        l["pv_coefs"],
                "r2":              round(l["r2"], 4),
                "aic":             round(l["aic"], 2),
                "sign_ok":         l["sign_violations"] == 0,
                "sign_violations": l["sign_violations"],
                "violating_vars":  l["violating_vars"],
                "n_obs":           l["n_obs"],
                "fitted_values":   l.get("fitted_values", []),
                "actual_values":   l.get("actual_values", []),
                "obs_years":       l.get("obs_years", []),
                "verdict":         l.get("verdict", "VALIDATED_WITH_WARNINGS"),
                "n_fail":          l.get("n_fail", 0),
                "n_warn":          l.get("n_warn", 0),
                "n_pass":          l.get("n_pass", 0),
            }
            for i, l in enumerate(leaderboard)
        ]

        # ══════════════════════════════════════════════════════════════════
        # STEP 5 — SCENARIO-AWARE PROJECTION
        # ══════════════════════════════════════════════════════════════════
        # Three scenario types are supported:
        #   A) Historical / Parametric → shocks from scenario_manager levels
        #   B) NGFS                    → deviations from NGFS data file
        #   C) Fallback                → historical crisis decomposition
        #
        # Common to all:
        #   - Baseline = satellite model on WEO projections
        #   - Profile shape from historical PD series (monotone increasing)
        #   - Frye-Jacobs LGD on final PD trajectories

        results: Dict[str, PlatformResult] = {}
        all_pd: Dict[str, Dict] = {}
        all_lgd: Dict[str, Dict] = {}
        all_el: Dict[str, Dict] = {}

        # ── 5a. PD Baseline (always the same) ────────────────────────────
        macro_baseline = _build_scenario_macro(
            macro_full, last_pd_year, horizon, {}, uni_idx)
        feature_baseline = _project_features(
            macro_baseline, best_combo, best_specs, horizon)
        pd_baseline = _predict_pd(feature_baseline, best)

        # ── 5b. PD crisis profile + data-driven amplitude floors ─────────
        profile = extract_pd_crisis_profile(pd_series, len(horizon))
        LOG.info("Crisis profile shape: %s", profile.round(2).tolist())

        # Floors calibrated on historical PD volatility so they are always
        # visible at the scale of the actual data (not a hard-coded 0.5 pp).
        pd_hist_std = float(pd_series.std())
        _floor_adv = max(0.010, pd_hist_std * 0.50)   # ≥ 50 % de σ(PD hist)
        _floor_sev = max(0.020, pd_hist_std * 1.00)   # ≥ 100 % de σ(PD hist)
        LOG.info(
            "Amplitude floors — adv=%.4f (%.2fpp), sev=%.4f (%.2fpp) "
            "[pd_hist_std=%.4f]",
            _floor_adv, _floor_adv * 100,
            _floor_sev, _floor_sev * 100,
            pd_hist_std,
        )

        # ── 5c. Detect scenario type; compute satellite PD trajectories ──
        scenario_type = self.scenario.get("type", "")
        levels_cfg = self.scenario.get("levels", {})

        adv_cfg = levels_cfg.get("adverse", {})
        has_idio_shocks = any(
            k in adv_cfg for k in
            ("npl_add_pp", "hqla_coef", "capital_coef", "rwa_coef",
             "outflow_coef", "asf_coef", "rsf_coef")
        )
        has_macro_shocks = any(
            k in adv_cfg for k in
            ("gdp_delta", "rate_bps", "unemployment_delta",
             "inflation_delta", "oil_pct", "fx_pct")
        )
        has_ngfs_severity_proxy = "ngfs_scenario" in adv_cfg

        # Store the full satellite PD trajectory per level (not just a scalar amp)
        pd_stressed_by_level: Dict[str, np.ndarray] = {}
        # Store stressed feature DataFrames for contribution decomposition
        feature_stressed_by_level: Dict[str, pd.DataFrame] = {}
        level_deltas: Dict[str, Dict[str, float]] = {"baseline": {}}
        scenario_info: Dict[str, Any] = {}

        if has_idio_shocks:
            # ── PATH D: Idiosyncratic — NPL additive shock, no macro ──
            LOG.info("Scenario type: IDIOSYNCRATIC (npl_add_pp direct shock)")
            scenario_info["scenario_mode"] = "idiosyncratic"

            for level in ("adverse", "severe"):
                shocks = levels_cfg.get(level, {})
                npl_add = float(shocks.get("npl_add_pp", 0.0)) / 100.0  # pp → fraction
                pd_stressed_by_level[level] = np.clip(
                    pd_baseline + npl_add, 1e-5, 0.99
                )
                feature_stressed_by_level[level] = feature_baseline.copy()
                level_deltas[level] = shocks
                LOG.info(
                    "  %s npl_add=%.2f pp → PD mean: %.4f → %.4f",
                    level, npl_add * 100,
                    float(pd_baseline.mean()),
                    float(pd_stressed_by_level[level].mean()),
                )

        elif has_macro_shocks:
            # ── PATH A: Parametric / Historical shocks ────────────────
            LOG.info("Scenario type: HISTORICAL/PARAMETRIC (macro shocks)")
            scenario_info["scenario_mode"] = "historical_parametric"

            for level in ("adverse", "severe"):
                shocks = levels_cfg.get(level, {})
                macro_stressed = _build_scenario_macro(
                    macro_full, last_pd_year, horizon, shocks, uni_idx)
                feature_stressed = _project_features(
                    macro_stressed, best_combo, best_specs, horizon)
                # Inject z-score perturbation for "growth"/"cycle"/"lag1"
                # features that attenuate a persistent level shock
                feature_stressed = _inject_z_shock(
                    feature_stressed, best_combo, best_specs,
                    shocks, uni_idx, len(horizon))
                pd_stressed_by_level[level] = _predict_pd(feature_stressed, best)
                feature_stressed_by_level[level] = feature_stressed
                level_deltas[level] = shocks
                LOG.info(
                    "  %s raw satellite amp: %.4f (%.2f pp)",
                    level,
                    float(np.max(pd_stressed_by_level[level] - pd_baseline)) * 100,
                    float(np.max(pd_stressed_by_level[level] - pd_baseline)) * 100,
                )

        elif has_ngfs_severity_proxy:
            # ── PATH B: NGFS severity proxy (historical crisis anchor) ─────────
            # NOTE: this path does NOT project NGFS macro trajectories through the
            # satellite model. It maps a requested NGFS scenario to a conventional
            # historical crisis and scales that crisis's macro deltas at 50%/100%.
            # The real climate→credit TYPE B mechanism (satellite calibrated on WEO
            # history, then injected with NGFS trajectories without refit) lives in
            # app/modules/climate/ngfs_credit_engine.py and is triggered from
            # climate/wrapper.py — entirely independent of this path.
            LOG.info(
                "Scenario type: NGFS_SEVERITY_PROXY — uses crisis_name='%s' "
                "as historical severity anchor (no NGFS macro trajectories here)",
                crisis_name,
            )
            scenario_info["scenario_mode"] = "ngfs_severity_proxy"
            scenario_info["ngfs_note"] = (
                "Historical crisis profile used as NGFS severity proxy. "
                "NGFS macro trajectories are projected separately via "
                "ngfs_credit_engine.py (climate module TYPE B mechanism)."
            )
            crisis = _resolve_crisis_safe(crisis_name)
            historical_deltas = compute_historical_shocks(macro_full, crisis)
            # Re-fetch ciblé si macro_full ne couvre pas la fenêtre de crise
            # (ex. gulf_war_1990 pic=1991 avec history_start > 1991)
            if not historical_deltas or int(macro_full.index.max()) < crisis.t_peak:
                LOG.warning(
                    "Crise '%s': macro_full [%d–%d] ne couvre pas le pic %d. "
                    "Re-fetch depuis %d.",
                    crisis.name, int(macro_full.index.min()),
                    int(macro_full.index.max()), crisis.t_peak,
                    crisis.t_pre_start - 2,
                )
                _ext_mac, _ = fetch_credit_macro(
                    country=country, start_year=crisis.t_pre_start - 2,
                    cache_dir=cache_dir, cache_ttl_days=cache_ttl,
                )
                if _ext_mac is not None and not _ext_mac.empty:
                    _ext_idx = (_ext_mac.set_index("year")
                                if "year" in _ext_mac.columns else _ext_mac)
                    historical_deltas = compute_historical_shocks(_ext_idx, crisis)
                if not historical_deltas:
                    raise RuntimeError(
                        f"Crise '{crisis.name}' [fenêtre {crisis.t_pre_start}–"
                        f"{crisis.t_peak}] : aucune donnée macro disponible pour "
                        f"le pays '{country}'. Vérifiez la couverture API."
                    )
            scenario_info["crisis_name"] = crisis.name
            scenario_info["crisis_description"] = crisis.description

            for level, pct in [("adverse", 0.50), ("severe", 1.00)]:
                scaled = {v: d * pct for v, d in historical_deltas.items()}
                macro_stressed = _build_scenario_macro_from_deltas(
                    macro_full, last_pd_year, horizon, scaled)
                feature_stressed = _project_features(
                    macro_stressed, best_combo, best_specs, horizon)
                pd_stressed_by_level[level] = _predict_pd(feature_stressed, best)
                feature_stressed_by_level[level] = feature_stressed
                level_deltas[level] = scaled

        else:
            # ── PATH C: Fallback — historical crisis decomposition ────
            LOG.info("Scenario type: FALLBACK (crisis_name=%s)", crisis_name)
            scenario_info["scenario_mode"] = "historical_crisis_fallback"
            crisis = _resolve_crisis_safe(crisis_name)
            historical_deltas = compute_historical_shocks(macro_full, crisis)
            # Re-fetch ciblé si macro_full ne couvre pas la fenêtre de crise
            if not historical_deltas or int(macro_full.index.max()) < crisis.t_peak:
                LOG.warning(
                    "Crise '%s': macro_full [%d–%d] ne couvre pas le pic %d. "
                    "Re-fetch depuis %d.",
                    crisis.name, int(macro_full.index.min()),
                    int(macro_full.index.max()), crisis.t_peak,
                    crisis.t_pre_start - 2,
                )
                _ext_mac, _ = fetch_credit_macro(
                    country=country, start_year=crisis.t_pre_start - 2,
                    cache_dir=cache_dir, cache_ttl_days=cache_ttl,
                )
                if _ext_mac is not None and not _ext_mac.empty:
                    _ext_idx = (_ext_mac.set_index("year")
                                if "year" in _ext_mac.columns else _ext_mac)
                    historical_deltas = compute_historical_shocks(_ext_idx, crisis)
                if not historical_deltas:
                    raise RuntimeError(
                        f"Crise '{crisis.name}' [fenêtre {crisis.t_pre_start}–"
                        f"{crisis.t_peak}] : aucune donnée macro disponible pour "
                        f"le pays '{country}'. Vérifiez la couverture API."
                    )
            scenario_info["crisis_name"] = crisis.name
            scenario_info["crisis_description"] = crisis.description

            for level, pct in [("adverse", 0.50), ("severe", 1.00)]:
                scaled = {v: d * pct for v, d in historical_deltas.items()}
                macro_stressed = _build_scenario_macro_from_deltas(
                    macro_full, last_pd_year, horizon, scaled)
                feature_stressed = _project_features(
                    macro_stressed, best_combo, best_specs, horizon)
                pd_stressed_by_level[level] = _predict_pd(feature_stressed, best)
                feature_stressed_by_level[level] = feature_stressed
                level_deltas[level] = scaled

        # ── 5d. Build final PD trajectories ──────────────────────────────
        # Strategy:
        #   1. Use the satellite model trajectory directly when it produces
        #      meaningful amplitude (≥ 80 % of the data-driven floor).
        #   2. If satellite output is too flat, scale it up proportionally
        #      so the floor amplitude is always met.
        #   3. If satellite is entirely flat (amp ≈ 0), fall back to
        #      pd_baseline + floor × profile.
        #   4. Enforce ordinal property: severe ≥ adverse ≥ baseline ∀t.
        #   5. Enforce minimum Adverse→Severe separation (× 1.5).

        pd_arrays: Dict[str, np.ndarray] = {"baseline": pd_baseline}
        _floors = {"adverse": _floor_adv, "severe": _floor_sev}

        for level in ("adverse", "severe"):
            floor = _floors[level]
            sat = pd_stressed_by_level.get(level)

            if sat is None:
                sat = pd_baseline + floor * profile

            sat_amp = float(np.max(sat - pd_baseline))

            if sat_amp >= floor * 0.8:
                # Satellite produces meaningful separation — use directly
                traj = sat.copy()
            elif sat_amp > 1e-6:
                # Satellite too weak — scale the deviation proportionally
                scale = floor / sat_amp
                traj = pd_baseline + (sat - pd_baseline) * scale
            else:
                # Satellite flat — profile × floor fallback
                traj = pd_baseline + floor * profile

            pd_arrays[level] = np.clip(traj, 1e-5, 0.99)

        # Ordinal enforcement: adverse ≥ baseline, severe ≥ adverse
        pd_arrays["adverse"] = np.maximum(pd_arrays["adverse"], pd_arrays["baseline"])
        pd_arrays["severe"] = np.maximum(pd_arrays["severe"], pd_arrays["adverse"])

        # Minimum Adverse → Severe separation (severe at least 1.5× adverse amplitude)
        adv_amp = float(np.max(pd_arrays["adverse"] - pd_baseline))
        sev_amp = float(np.max(pd_arrays["severe"] - pd_baseline))
        if sev_amp < adv_amp * 1.5:
            deficit = adv_amp * 1.5 - sev_amp
            pd_arrays["severe"] = np.clip(
                pd_arrays["severe"] + deficit * profile, 1e-5, 0.99)

        amplitudes = {
            lv: float(np.max(pd_arrays[lv] - pd_baseline))
            for lv in ("baseline", "adverse", "severe")
        }
        LOG.info(
            "Final amplitudes — adv=%.4f (%.2fpp), sev=%.4f (%.2fpp)",
            amplitudes["adverse"], amplitudes["adverse"] * 100,
            amplitudes["severe"], amplitudes["severe"] * 100,
        )

        # ── 5d-bis. ΔPD contributions per macro variable (baseline → severe) ──
        # For each feature f in the winning model:
        #   Δz_f = mean(z_stressed_severe_f) - mean(z_baseline_f)   [z-score delta]
        #   Δlogit_f = beta_f × Δz_f                                [logit contribution]
        # Proportionally attribute total ΔPD in pp by each parent variable's share.
        pd_contributions: Dict[str, float] = {}
        sat_amp_severe = float(np.max(
            pd_stressed_by_level.get("severe", pd_baseline) - pd_baseline))
        scenario_info["satellite_amplitude_pp"] = round(sat_amp_severe * 100, 4)

        f_sev = feature_stressed_by_level.get("severe")
        if f_sev is not None:
            coefs_dict = best.get("coefs", {})
            parent_dlogit: Dict[str, float] = {}
            for fname in best_combo:
                if fname not in feature_baseline.columns:
                    continue
                parent = best_specs[fname].parent_var
                z_bl_raw = np.nanmean(feature_baseline[fname].values)
                z_sv_raw = (np.nanmean(f_sev[fname].values)
                            if fname in f_sev.columns else z_bl_raw)
                if not np.isfinite(z_bl_raw) or not np.isfinite(z_sv_raw):
                    LOG.warning(
                        "Feature '%s' produced non-finite z-scores "
                        "(z_bl=%.4g, z_sv=%.4g) — likely a NaN in the raw "
                        "macro series at the history/projection boundary. "
                        "Skipping this feature's contribution.",
                        fname, z_bl_raw, z_sv_raw,
                    )
                    continue
                z_bl = float(z_bl_raw)
                z_sv = float(z_sv_raw)
                beta = float(coefs_dict.get(fname, 0.0))
                dz = z_sv - z_bl
                LOG.debug(
                    "  ΔPD decomp: %-38s β=%.4f Δz=%.4f (z_bl=%.3f z_sv=%.3f) → Δlogit=%.6f",
                    fname, beta, dz, z_bl, z_sv, beta * dz,
                )
                parent_dlogit[parent] = parent_dlogit.get(parent, 0.0) + beta * dz

            total_dlogit = sum(parent_dlogit.values())
            total_dpd_pp = amplitudes.get("severe", 0.0) * 100.0
            LOG.info(
                "ΔPD contributions: total_dlogit=%.6f  sat_amp=%.4f pp  floor_amp=%.4f pp",
                total_dlogit, sat_amp_severe * 100, total_dpd_pp,
            )
            if abs(total_dlogit) > 1e-8:
                pd_contributions = {
                    parent: round(dlogit / total_dlogit * total_dpd_pp, 2)
                    for parent, dlogit in parent_dlogit.items()
                }
            else:
                LOG.warning(
                    "ΔPD contributions: total_dlogit≈0 despite shocks — "
                    "satellite insensitive (sat_amp=%.4f pp). Floor drove amplitude=%.2f pp. "
                    "Per-feature Δz: %s",
                    sat_amp_severe * 100, total_dpd_pp,
                    {fname: round(
                        float(np.mean(f_sev[fname].values))
                        - float(np.mean(feature_baseline[fname].values)), 4
                    ) for fname in best_combo if fname in f_sev.columns},
                )
            LOG.info("ΔPD contributions per variable (severe): %s", pd_contributions)
        scenario_info["pd_contributions"] = pd_contributions

        # ── 5e. Build PlatformResult for each level ──────────────────────

        hist_years = sorted(pd_series.index.tolist())
        hist_pd = pd_series.sort_index().values.astype(float)
        hist_lgd = _frye_jacobs_lgd(hist_pd, rho, fj_k)
        hist_el = hist_pd * hist_lgd * ead_M

        for level in ("baseline", "adverse", "severe"):
            pd_v = pd_arrays[level]

            lgd_v = _frye_jacobs_lgd(pd_v, rho, fj_k)
            el_v = pd_v * lgd_v * ead_M

            ts = pd.DataFrame({
                "year": horizon,
                "pd": pd_v,
                "lgd": lgd_v,
                "ead": ead_M,
                "el": el_v,
                "loss": el_v,
            })

            full_x = [str(y) for y in hist_years] + [str(y) for y in horizon]
            full_pd = hist_pd.tolist() + pd_v.tolist()
            full_lgd = hist_lgd.tolist() + lgd_v.tolist()
            full_el = hist_el.tolist() + el_v.tolist()

            all_pd[level]  = {"x": full_x, "y": full_pd}
            all_lgd[level] = {"x": full_x, "y": full_lgd}
            all_el[level]  = {"x": full_x, "y": full_el}

            results[level] = PlatformResult(
                module_id=self.module_id,
                module_label=self.module_label,
                scenario_id=level,
                total_loss=float(el_v.sum()),
                kpis={
                    "avg_pd":   round(float(pd_v.mean()), 6),
                    "peak_pd":  round(float(pd_v.max()), 6),
                    "avg_lgd":  round(float(lgd_v.mean()), 6),
                    "total_el": round(float(el_v.sum()), 0),
                    "peak_el":  round(float(el_v.max()), 0),
                    "ead":      ead_M,
                },
                time_series=ts,
                charts_data={
                    "pd_trajectories":            all_pd,
                    "el_trajectories":            all_el,
                    "lgd_trajectories":           all_lgd,
                    "model_table":                ranking,
                    "selected_rank_0idx":         selected_lb_idx,
                    "validation_report":          _credit_vreport_dict,
                    "user_forced_rejected_model": _user_forced_rejected,
                },
                metadata={
                    **scenario_info,
                    "shocks_applied": {
                        k: round(v, 4)
                        for k, v in level_deltas.get(level, {}).items()
                    },
                    "amplitude_pp": round(
                        amplitudes.get(level, 0) * 100, 2),
                    "profile_shape": profile.round(3).tolist(),
                    "country": country,
                    "country_iso3": fetch_report.country_iso3,
                    "best_modele": best["family"],
                    "best_combo": best["combo"],
                    "best_r2": round(best["r2"], 4),
                    "best_rmse": round(best["rmse"], 5),
                    "best_sign_violations": best["sign_violations"],
                    "pd_ttc": round(pd_ttc, 6),
                    "lgd_ttc": lgd_ttc,
                    "fj_rho": round(rho, 6),
                    "fj_k": round(fj_k, 6),
                    "n_candidates_tournament": len(leaderboard),
                    "horizon": horizon,
                    "n_obs_train": n_obs,
                    "fetch_sources": fetch_report.sources,
                    "fetch_missing": fetch_report.missing,
                    "imf_has_projections": fetch_report.has_projections,
                    "harrell_max_combo": max(
                        1, n_obs // HARRELL_OBS_PER_PARAM - 1),
                    "feature_specs": {
                        f: {
                            "parent": s.parent_var,
                            "transform": s.transform,
                            "mu": round(s.mu, 6),
                            "sigma": round(s.sigma, 6),
                            "expected_sign": s.expected_sign,
                            "corr_score": round(s.corr_score, 4),
                        }
                        for f, s in best_specs.items()
                    },
                },
            )

        # ══════════════════════════════════════════════════════════════════
        # STEP 6 — CAPITAL ENGINE (VaR ASRF / RWA / CAR / Levier)
        # ══════════════════════════════════════════════════════════════════
        # Optionnel : activé si capital_historique.csv uploadé ou rwa_base
        # dans params. Sinon : VaR + K + RWA calculés, ratios = N/A.

        _cap_path = self.upload_paths.get("capital", "")
        LOG.info("CAPITAL PATH resolved: '%s' | upload_paths keys: %s",
                 _cap_path, list(self.upload_paths.keys()))
        capital_df = load_capital_history(_cap_path)

        # Build capital engine params: start from user params, then inject the
        # data-driven EAD growth rates so the hardcoded 0.05 default is never used.
        # Adverse = CAGR − 1σ  (credit slowdown under moderate stress)
        # Severe  = CAGR − 2σ  (possible contraction for volatile portfolios)
        # User can still override all three via params if needed.
        _cap_params = _build_capital_params(ead_series, p)
        LOG.info(
            "EAD growth rates → baseline=%.4f, adverse=%.4f, severe=%.4f",
            _cap_params["ead_growth_rate"],
            _cap_params["ead_growth_adverse"],
            _cap_params["ead_growth_severe"],
        )

        cap_engine = CreditCapitalEngine(_cap_params)

        # Build lgd_arrays for capital engine
        lgd_arrays: Dict[str, np.ndarray] = {}
        for level in ("baseline", "adverse", "severe"):
            lgd_arrays[level] = _frye_jacobs_lgd(pd_arrays[level], rho, fj_k)

        # Capital CSV columns are in M (millions of local currency).
        # EAD comes from the platform in absolute units (e.g. EGP).
        # Divide EAD by 1,000,000 so ECL and cascade amounts are in the same
        # unit (millions) as the capital CSV — otherwise ECL dwarfs capital.
        ead_for_capital = ead_M

        capital_out = cap_engine.run(
            pd_hist=pd_series,
            lgd_hist=hist_lgd,
            pd_stress=pd_arrays,
            lgd_stress=lgd_arrays,
            ead=ead_for_capital,
            horizon=horizon,
            capital_df=capital_df,
        )

        # ── Idiosyncratic: apply capital_coef / rwa_coef post-engine ─────────
        # PATH D only — the ASRF engine computed a baseline from PD/LGD;
        # now we layer the direct balance-sheet shocks on top.
        if scenario_type == "idiosyncratic":
            for level in ("adverse", "severe"):
                shocks  = levels_cfg.get(level, {})
                cap_coef = float(shocks.get("capital_coef", 0.0))
                rwa_c    = float(shocks.get("rwa_coef",     0.0))
                if cap_coef == 0.0 and rwa_c == 0.0:
                    continue
                mask = capital_out.stress_df["scenario"] == level
                if cap_coef != 0.0:
                    for col in ("total_cap_proj", "tier1_proj", "cet1_proj"):
                        if col in capital_out.stress_df.columns:
                            capital_out.stress_df.loc[mask, col] = (
                                capital_out.stress_df.loc[mask, col].values
                                * (1.0 - cap_coef)
                            )
                if rwa_c != 0.0:
                    capital_out.stress_df.loc[mask, "rwa_stressed"] = (
                        capital_out.stress_df.loc[mask, "rwa_stressed"].values
                        * (1.0 + rwa_c)
                    )
                # Recompute ratios from updated capital + RWA
                sub  = capital_out.stress_df.loc[mask].copy()
                rwa_s = sub["rwa_stressed"].clip(lower=1e-6).values
                capital_out.stress_df.loc[mask, "car"] = (
                    sub["total_cap_proj"].values / rwa_s
                )
                capital_out.stress_df.loc[mask, "tier1_ratio"] = (
                    sub["tier1_proj"].values / rwa_s
                )
                if "cet1_proj" in sub.columns:
                    capital_out.stress_df.loc[mask, "cet1_ratio"] = (
                        sub["cet1_proj"].values / rwa_s
                    )
                LOG.info(
                    "Idiosyncratic capital post-adjustment [%s]: "
                    "cap_coef=%.3f, rwa_coef=%.3f",
                    level, cap_coef, rwa_c,
                )

        # Merge capital KPIs + charts_data into each PlatformResult
        for level in ("baseline", "adverse", "severe"):
            r = results[level]
            r.kpis.update({
                k: v for k, v in capital_out.kpis.items()
                if k.startswith(level)
            })
            r.charts_data["capital_trajectories"] = capital_out.capital_trajectories
            r.metadata["capital_engine_active"] = capital_df is not None
            r.metadata["min_car"] = cap_engine.min_car
            r.metadata["min_leverage"] = cap_engine.min_leverage
            r.metadata["min_cet1"] = cap_engine.min_cet1
            r.metadata["min_tier1"] = cap_engine.min_tier1

            # Add stress capital data to time_series
            stress_sub = capital_out.stress_df[
                capital_out.stress_df["scenario"] == level
            ].copy()
            if not stress_sub.empty:
                for col in ("var_99_9", "k_required", "rwa_stressed",
                            "car", "tier1_ratio", "leverage_ratio",
                            "car_flag", "leverage_flag"):
                    if col in stress_sub.columns:
                        r.time_series[col] = stress_sub[col].values

        return results

    def get_kpis(self, results: Dict) -> Dict[str, Any]:
        out = {}
        for level, r in results.items():
            for k, v in r.kpis.items():
                out[f"{level}_{k}"] = v
        return out
