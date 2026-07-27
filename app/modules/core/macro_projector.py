"""
macro_projector.py
==================
Forward projection of macro variables that feed the satellite credit model.

Why this matters
----------------
The satellite model gives PD = f(macro). To project PD over the horizon
(typically 5-6 years), we need projected paths for the macro variables that
enter the model. The choice of projection technique depends on:

  1. Number of variables in the selected satellite model (|V*|)
  2. Stationarity of each series (ADF test)
  3. Cointegration between I(1) series (Johansen test)
  4. Number of historical observations available (degrees of freedom)

Decision tree
-------------
                        |V*| = 1                    |V*| ≥ 2
                            │                           │
                     ADF: stationary?            ADF: all I(0)?
                            │                           │
               ┌────────────┴─────────┐       ┌────────┴──────────┐
             I(0)                   I(1)     YES                  NO
               │                     │     (all stationary)   (contains I(1))
             AR(p)       first-diff → AR(p)      │                  │
                         → reconstruct        VAR(p)          Johansen test
                                               on levels           │
                                                         ┌──────────┴──────────┐
                                                    Cointegrated?        Not cointegrated
                                                         │                     │
                                                      VECM(p)          VAR(p) on
                                                   (levels, ECM)       differences
                                                         │
                                            Guard: n_obs ≥ 4·k²·p ?
                                                         │
                                              ┌──────────┴──────────┐
                                             YES                    NO
                                              │                      │
                                          VECM(p)       Fallback → AR on most
                                                         correlated var

After projection (all paths):
──────────────────────────────
  → Economic Diagnostics
  → Projection Report

Order selection
---------------
For both AR and VAR, p is chosen by minimising the BIC criterion over
p ∈ {1, 2, ..., p_max} where p_max = min(4, n_obs // 4).

BIC is preferred over AIC for small samples (more parsimonious — penalises
parameters more heavily, reducing overfitting risk). For VECM, the same
BIC-based selection applies to k_ar_diff.

Cointegration
-------------
The Johansen trace test (Johansen 1988, 1991) tests H0: rank ≤ r against
H1: rank > r for r = 0, 1, ..., k-1. We use 95 % critical values.
A rank ≥ 1 indicates cointegration → VECM. Rank = 0 → VAR on differences.

Economic Diagnostics
--------------------
After projection, the platform generates observations about the projected
paths. These are purely informational — the projection is never blocked or
corrected. The user retains full judgement.

Checks performed:
  • Domain validity  : negative unemployment, PD outside [0, 1], etc.
  • Historical range : values below min or above max observed historically
  • Stability        : explosive trajectories (annual change > 3σ)
  • Macro consistency: GDP/unemployment directional coherence (if both present)

Academic references
-------------------
* Johansen (1988) "Statistical analysis of cointegration vectors" — JEBO
* Johansen (1991) "Estimation and hypothesis testing of cointegration
  vectors in Gaussian VAR models" — Econometrica
* Lütkepohl (2005) "New Introduction to Multiple Time Series Analysis"
* Hamilton (1994) "Time Series Analysis"
* Hosmer & Lemeshow (2000) "Applied Logistic Regression" — rule of 7-10
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# statsmodels imports
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

LOG = logging.getLogger("macro_projector")
warnings.filterwarnings("ignore", category=Warning)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ProjectorConfig:
    """All knobs of the macro projector."""
    p_max:              int   = 4      # max lag order tested
    p_max_div_obs:      int   = 4      # p_max = min(p_max, n_obs // p_max_div_obs)
    adf_pvalue:         float = 0.10   # ADF threshold (p > → non-stationary)
    johansen_alpha:     float = 0.05   # Johansen trace significance level
    johansen_det_order: int   = 0      # 0 = constant in cointegrating relation
    var_guard_factor:   int   = 4      # min n_obs = factor × k² × p
    min_obs_for_ar:     int   = 8      # below this → flat extrapolation
    min_obs_for_var:    int   = 12     # below this → never attempt VAR/VECM
    min_obs_for_vecm:   int   = 16     # below this → never attempt VECM
    diag_sigma_thresh:  float = 3.0    # stability: change > N×σ → warning


# ═══════════════════════════════════════════════════════════════════════════
# Results dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DiagnosticMessage:
    """One observation from the economic diagnostics engine."""
    level:   str   # "INFO" | "WARNING"
    message: str

    def __str__(self) -> str:
        icon = "OK" if self.level == "INFO" else "!"
        return f"{icon} {self.message}"


@dataclass
class ProjectionResult:
    """Complete output of project_macro()."""
    projected:          pd.DataFrame
    model_used:         str                   # "AR(p)" | "VAR(p)" | "VECM(p)" | "AR(p)-fallback" | "naive"
    p_optimal:          int
    differenced_vars:   List[str]             = field(default_factory=list)
    adf_pvalues:        Dict[str, float]      = field(default_factory=dict)
    coint_rank:         int                   = 0       # Johansen rank (0 = no cointegration)
    coint_tested:       bool                  = False   # True if Johansen was run
    johansen_details:   Dict[str, Any]        = field(default_factory=dict)
    fit_r2:             Dict[str, float]      = field(default_factory=dict)
    fallback_reason:    Optional[str]         = None
    diagnostics_msgs:   List[DiagnosticMessage] = field(default_factory=list)
    report:             Dict[str, Any]        = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# 1. STATIONARITY
# ═══════════════════════════════════════════════════════════════════════════

def _adf_test(series: pd.Series, alpha: float = 0.10) -> Tuple[bool, float]:
    """Augmented Dickey-Fuller test.

    Returns (is_stationary, p_value).
    Stationary if p < alpha (reject H0 of unit root).
    Conservative fallback on failure: assume stationary (no differencing).
    """
    s = pd.Series(series).dropna()
    if len(s) < 6:
        return True, 1.0
    if np.isclose(s.std(), 0.0):
        return True, 0.0
    try:
        max_lag = min(int(len(s) ** 0.5), len(s) // 4)
        result = adfuller(s.values, maxlag=max(1, max_lag), autolag="AIC",
                          regression="c")
        p_val = float(result[1])
        return p_val < alpha, p_val
    except Exception as e:
        LOG.debug("ADF failed for %s: %s — assume stationary", series.name, e)
        return True, 1.0


def _difference_if_needed(
    df: pd.DataFrame,
    cfg: ProjectorConfig,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """For each column, run ADF. If non-stationary, replace with first
    difference. Returns (transformed_df, differenced_columns, adf_pvalues).
    """
    df_out = df.copy()
    differenced: List[str] = []
    adf_pvals: Dict[str, float] = {}
    for col in df.columns:
        is_stat, p_val = _adf_test(df[col], alpha=cfg.adf_pvalue)
        adf_pvals[col] = round(p_val, 4)
        if not is_stat:
            df_out[col] = df[col].diff()
            differenced.append(col)
            LOG.info("ADF p=%.3f for '%s' → first-differenced", p_val, col)
    if differenced:
        df_out = df_out.dropna()
    return df_out, differenced, adf_pvals


def _reconstruct_levels(
    df_levels_hist: pd.DataFrame,
    df_diff_proj: pd.DataFrame,
    differenced_cols: List[str],
) -> pd.DataFrame:
    """Reconstruct level projections from differenced projections.

    For each year t: level[t] = level[t-1] + diff[t], anchored at last
    observed level.
    """
    out = df_diff_proj.copy()
    for col in df_diff_proj.columns:
        if col in differenced_cols:
            last_level = float(df_levels_hist[col].iloc[-1])
            out[col] = last_level + df_diff_proj[col].cumsum()
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. JOHANSEN COINTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════════════

def _johansen_test(
    df: pd.DataFrame,
    k_ar_diff: int,
    det_order: int,
    alpha: float,
) -> Tuple[int, Dict[str, Any]]:
    """Run Johansen trace test to determine cointegration rank.

    Parameters
    ----------
    df        : DataFrame of I(1) variables (levels).
    k_ar_diff : Number of lagged differences in the test VAR (= p - 1).
    det_order : Deterministic terms (−1=none, 0=const, 1=const+trend).
    alpha     : Significance level for trace statistic (0.05 typical).

    Returns
    -------
    (rank, details)
    rank = 0  → no cointegration → use VAR on differences
    rank ≥ 1  → cointegration detected → use VECM(rank)

    details keys
    ------------
    trace_stats   : list of trace statistics for H0: rank ≤ r
    critical_vals : list of 95% critical values
    eigenvalues   : list of eigenvalues
    """
    details: Dict[str, Any] = {
        "trace_stats": [], "critical_vals": [], "eigenvalues": [],
        "alpha_used": alpha, "det_order": det_order,
    }

    if len(df) < 5 * df.shape[1]:
        LOG.info("Johansen: too few obs (%d) for %d vars — rank=0", len(df), df.shape[1])
        details["note"] = "insufficient observations — rank forced to 0"
        return 0, details

    try:
        res = coint_johansen(df.values, det_order, k_ar_diff)
    except Exception as e:
        LOG.warning("Johansen test failed: %s — rank=0", e)
        details["note"] = f"test failed: {e}"
        return 0, details

    # Trace statistics and 95% critical values (col index 1 = 95%)
    trace_stats  = res.lr1.tolist()
    crit_95      = res.cvt[:, 1].tolist()  # 95% critical values
    eigenvalues  = res.eig.tolist()

    details["trace_stats"]   = [round(v, 4) for v in trace_stats]
    details["critical_vals"] = [round(v, 4) for v in crit_95]
    details["eigenvalues"]   = [round(v, 6) for v in eigenvalues]

    # Rank = number of H0 hypotheses rejected sequentially
    # H0: rank ≤ r is rejected when trace_stat[r] > crit_95[r]
    rank = 0
    for i, (ts, cv) in enumerate(zip(trace_stats, crit_95)):
        if ts > cv:
            rank = i + 1
        else:
            break

    LOG.info("Johansen trace test: rank=%d (alpha=%.2f)", rank, alpha)
    details["rank"] = rank
    return rank, details


# ═══════════════════════════════════════════════════════════════════════════
# 3. AR(p) UNIVARIATE
# ═══════════════════════════════════════════════════════════════════════════

def _fit_ar_best(series: pd.Series, p_max: int) -> Tuple[Optional[AutoReg], int, float]:
    """Fit AR(p) for p in 1..p_max, return the one with lowest BIC.

    Returns (fitted_model, optimal_p, fit_r2).
    """
    best_bic = np.inf
    best_p = 1
    best_fit = None
    best_r2 = np.nan

    y = series.dropna().astype(float).values
    if len(y) < 5:
        return None, 0, np.nan

    for p in range(1, p_max + 1):
        try:
            model = AutoReg(y, lags=p, trend="c", old_names=False).fit()
            bic = float(model.bic)
            if bic < best_bic:
                resid_var = float(np.var(model.resid))
                total_var = float(np.var(y[p:]))
                r2 = 1.0 - resid_var / max(total_var, 1e-12)
                best_bic = bic
                best_p = p
                best_fit = model
                best_r2 = float(np.clip(r2, -10.0, 1.0))
        except Exception as e:
            LOG.debug("AR(%d) failed: %s", p, e)

    return best_fit, best_p, best_r2


def _project_ar(
    series: pd.Series,
    horizon_years: List[int],
    p_max: int,
) -> Tuple[Optional[pd.Series], int, float]:
    """Project one series with AR(p), p selected by BIC.
    Returns (projection, p_optimal, fit_r2).
    """
    fit, p, r2 = _fit_ar_best(series, p_max)
    if fit is None:
        return None, 0, np.nan
    try:
        forecast = fit.forecast(steps=len(horizon_years))
        return pd.Series(forecast, index=horizon_years), p, r2
    except Exception as e:
        LOG.warning("AR forecast failed: %s", e)
        return None, p, r2


# ═══════════════════════════════════════════════════════════════════════════
# 4. VAR(p) MULTIVARIATE
# ═══════════════════════════════════════════════════════════════════════════

def _var_guard_passed(n_obs: int, k: int, p: int, factor: int) -> bool:
    """Lütkepohl (2005) degrees-of-freedom guard: n_obs ≥ factor × k² × p."""
    return n_obs >= factor * (k ** 2) * p


def _fit_var_best(
    df: pd.DataFrame,
    p_max: int,
    cfg: ProjectorConfig,
) -> Tuple[Optional[object], int, float]:
    """Fit VAR(p) for p in 1..p_max, return one with lowest BIC.

    Returns (fitted_VAR, optimal_p, mean_r2_across_vars).
    """
    n_obs, k = df.shape
    best_bic = np.inf
    best_p = 0
    best_fit = None
    best_r2 = np.nan

    if k < 2 or n_obs < cfg.min_obs_for_var:
        return None, 0, np.nan

    try:
        model = VAR(df.values)
    except Exception as e:
        LOG.warning("VAR initialisation failed: %s", e)
        return None, 0, np.nan

    for p in range(1, p_max + 1):
        if not _var_guard_passed(n_obs, k, p, cfg.var_guard_factor):
            LOG.info("VAR(%d) guard failed: n_obs=%d < %d — skip",
                     p, n_obs, cfg.var_guard_factor * k * k * p)
            continue
        try:
            fit = model.fit(p, trend="c")
            bic = float(fit.bic)
            if bic < best_bic:
                r2s = []
                for j in range(k):
                    fitted = fit.fittedvalues[:, j]
                    actual = df.values[p:, j]
                    if len(fitted) != len(actual):
                        actual = actual[-len(fitted):]
                    rv = float(np.var(actual - fitted))
                    tv = float(np.var(actual))
                    r2s.append(1.0 - rv / max(tv, 1e-12))
                best_bic = bic
                best_p = p
                best_fit = fit
                best_r2 = float(np.mean(np.clip(r2s, -10.0, 1.0)))
        except Exception as e:
            LOG.debug("VAR(%d) failed: %s", p, e)

    return best_fit, best_p, best_r2


def _project_var(
    df: pd.DataFrame,
    horizon_years: List[int],
    cfg: ProjectorConfig,
    p_max: int,
) -> Tuple[Optional[pd.DataFrame], int, float]:
    """Project all variables jointly via VAR(p).
    Returns (projection_df, p_optimal, mean_r2).
    """
    fit, p, r2 = _fit_var_best(df, p_max, cfg)
    if fit is None:
        return None, 0, np.nan
    try:
        last = df.values[-p:]
        forecast = fit.forecast(y=last, steps=len(horizon_years))
        out = pd.DataFrame(forecast, index=horizon_years, columns=df.columns)
        return out, p, r2
    except Exception as e:
        LOG.warning("VAR forecast failed: %s", e)
        return None, p, r2


# ═══════════════════════════════════════════════════════════════════════════
# 5. VECM (Vector Error Correction Model)
# ═══════════════════════════════════════════════════════════════════════════

def _fit_vecm_best(
    df: pd.DataFrame,
    coint_rank: int,
    p_max: int,
    cfg: ProjectorConfig,
) -> Tuple[Optional[object], int, float]:
    """Fit VECM for k_ar_diff in 0..p_max-1, select by BIC.

    VECM(k_ar_diff) is equivalent to a VAR(k_ar_diff + 1) in differences
    plus error correction terms. The cointegration rank is fixed from the
    Johansen test — we only search over the lag order.

    Returns (fitted_VECM, k_ar_diff_optimal, mean_r2).
    """
    n_obs, k = df.shape
    best_bic = np.inf
    best_kar = 0
    best_fit = None
    best_r2  = np.nan

    if n_obs < cfg.min_obs_for_vecm:
        LOG.info("VECM: too few obs (%d < %d) — skip", n_obs, cfg.min_obs_for_vecm)
        return None, 0, np.nan

    # k_ar_diff = number of lagged differences = p_var - 1
    for k_ar in range(0, p_max):
        if not _var_guard_passed(n_obs, k, k_ar + 1, cfg.var_guard_factor):
            LOG.info("VECM k_ar_diff=%d guard failed — skip", k_ar)
            continue
        try:
            model = VECM(
                df.values,
                k_ar_diff=k_ar,
                coint_rank=coint_rank,
                deterministic="ci",   # constant inside cointegrating relation
            )
            fit = model.fit()

            # BIC proxy: use residual sum of squares
            resids = df.values[k_ar + 1:] - fit.fittedvalues
            n_eff = resids.shape[0]
            n_params = k * (k_ar * k + coint_rank) + coint_rank * k
            rss = float(np.sum(resids ** 2))
            bic = n_eff * np.log(max(rss / n_eff, 1e-12)) + n_params * np.log(n_eff)

            if bic < best_bic:
                r2s = []
                for j in range(k):
                    fitted_j = fit.fittedvalues[:, j]
                    actual_j = df.values[k_ar + 1:, j]
                    if len(fitted_j) != len(actual_j):
                        actual_j = actual_j[-len(fitted_j):]
                    rv = float(np.var(actual_j - fitted_j))
                    tv = float(np.var(actual_j))
                    r2s.append(1.0 - rv / max(tv, 1e-12))
                best_bic = bic
                best_kar = k_ar
                best_fit = fit
                best_r2  = float(np.mean(np.clip(r2s, -10.0, 1.0)))
        except Exception as e:
            LOG.debug("VECM k_ar_diff=%d failed: %s", k_ar, e)

    return best_fit, best_kar, best_r2


def _project_vecm(
    df: pd.DataFrame,
    horizon_years: List[int],
    coint_rank: int,
    cfg: ProjectorConfig,
    p_max: int,
) -> Tuple[Optional[pd.DataFrame], int, float]:
    """Project all variables jointly via VECM.

    VECM naturally works on levels and enforces long-run equilibrium through
    the error correction mechanism. No level reconstruction is needed.

    Returns (projection_df, k_ar_diff_optimal, mean_r2).
    """
    fit, k_ar, r2 = _fit_vecm_best(df, coint_rank, p_max, cfg)
    if fit is None:
        return None, 0, np.nan
    try:
        # predict() returns an ndarray of shape (n_steps, k) in levels
        forecast = fit.predict(steps=len(horizon_years))
        out = pd.DataFrame(forecast, index=horizon_years, columns=df.columns)
        return out, k_ar, r2
    except Exception as e:
        LOG.warning("VECM forecast failed: %s", e)
        return None, k_ar, r2


# ═══════════════════════════════════════════════════════════════════════════
# 6. ECONOMIC DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

# Known domain bounds per variable-name keyword (lower, upper).
# None means unbounded on that side.
_DOMAIN_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "unemployment":     (0.0,  100.0),
    "chômage":          (0.0,  100.0),
    "inflation":        (-50.0, 200.0),
    "interest":         (-20.0, 100.0),
    "rate":             (-20.0, 100.0),
    "gdp_growth":       (-50.0,  50.0),
    "growth":           (-50.0,  50.0),
    "default":          (0.0,    1.0),   # PD must be in [0,1]
    "pd":               (0.0,    1.0),
    "lgd":              (0.0,    1.0),
}


def _domain_bounds(col: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (lower, upper) domain bounds for a variable, or (None, None)."""
    col_lower = col.lower()
    for keyword, bounds in _DOMAIN_BOUNDS.items():
        if keyword in col_lower:
            return bounds
    return None, None


def _economic_diagnostics(
    historical: pd.DataFrame,
    projected: pd.DataFrame,
    sigma_thresh: float = 3.0,
) -> List[DiagnosticMessage]:
    """Generate economic observations on projected trajectories.

    This function NEVER modifies the projection. It only observes and reports.

    Checks:
      1. Domain validity    — values outside known physical/economic bounds
      2. Historical range   — values below historical min or above historical max
      3. Stability          — annual changes > N × historical σ (explosive path)
      4. Macro consistency  — GDP ↓ should correspond to unemployment ↑

    Parameters
    ----------
    historical : DataFrame with historical observations (level values).
    projected  : DataFrame with projected values (future horizon only).
    sigma_thresh : multiplier for stability check.

    Returns
    -------
    List of DiagnosticMessage (level = "INFO" or "WARNING").
    """
    msgs: List[DiagnosticMessage] = []
    cols = [c for c in projected.columns if c in historical.columns]

    # ── Per-variable checks ──────────────────────────────────────────────
    for col in projected.columns:
        proj_vals = projected[col].dropna().values
        if len(proj_vals) == 0:
            continue

        # 1. Domain validity
        lo, hi = _domain_bounds(col)
        if lo is not None and np.any(proj_vals < lo):
            msgs.append(DiagnosticMessage(
                "WARNING",
                f"{col}: projected values below domain minimum ({lo}). "
                f"Min projected = {proj_vals.min():.4f}."
            ))
        if hi is not None and np.any(proj_vals > hi):
            msgs.append(DiagnosticMessage(
                "WARNING",
                f"{col}: projected values above domain maximum ({hi}). "
                f"Max projected = {proj_vals.max():.4f}."
            ))

        # 2. Historical range
        if col in historical.columns:
            hist_vals = historical[col].dropna().values
            if len(hist_vals) > 0:
                hist_min = float(hist_vals.min())
                hist_max = float(hist_vals.max())
                proj_min = float(proj_vals.min())
                proj_max = float(proj_vals.max())
                if proj_min < hist_min:
                    msgs.append(DiagnosticMessage(
                        "WARNING",
                        f"{col}: projected minimum ({proj_min:.4f}) is below "
                        f"historical minimum ({hist_min:.4f})."
                    ))
                if proj_max > hist_max:
                    msgs.append(DiagnosticMessage(
                        "WARNING",
                        f"{col}: projected maximum ({proj_max:.4f}) exceeds "
                        f"historical maximum ({hist_max:.4f})."
                    ))

                # 3. Stability — annual changes
                hist_diff = np.diff(hist_vals)
                if len(hist_diff) > 1:
                    hist_sigma = float(np.std(hist_diff, ddof=1))
                    proj_diff  = np.diff(proj_vals)
                    explosive  = np.abs(proj_diff) > sigma_thresh * hist_sigma
                    if np.any(explosive) and hist_sigma > 1e-9:
                        worst = float(np.abs(proj_diff[explosive]).max())
                        msgs.append(DiagnosticMessage(
                            "WARNING",
                            f"{col}: potentially explosive trajectory. "
                            f"Largest annual change = {worst:.4f} "
                            f"({worst / hist_sigma:.1f}× historical σ = {hist_sigma:.4f})."
                        ))

    # ── No domain issues — add global confirmation if no warnings emitted ──
    domain_warnings = [m for m in msgs if "domain" in m.message or
                       "below domain" in m.message or "above domain" in m.message]
    if not domain_warnings:
        msgs.append(DiagnosticMessage(
            "INFO", "No invalid values detected (all projections within domain bounds)."
        ))

    # ── 4. Macro consistency: GDP ↓ should correspond to unemployment ↑ ──
    gdp_col  = next((c for c in projected.columns if "gdp_growth" in c.lower()
                     or ("gdp" in c.lower() and "growth" in c.lower())), None)
    unem_col = next((c for c in projected.columns if "unemployment" in c.lower()
                     or "chômage" in c.lower()), None)

    if gdp_col and unem_col:
        gdp_trend  = float(projected[gdp_col].iloc[-1] - projected[gdp_col].iloc[0])
        unem_trend = float(projected[unem_col].iloc[-1] - projected[unem_col].iloc[0])
        # Expected: GDP ↓ → unemployment ↑  (negative correlation)
        consistent = (gdp_trend < 0 and unem_trend > 0) or \
                     (gdp_trend > 0 and unem_trend < 0) or \
                     (abs(gdp_trend) < 0.5 and abs(unem_trend) < 0.5)
        if consistent:
            msgs.append(DiagnosticMessage(
                "INFO",
                f"GDP / unemployment directional coherence verified "
                f"(ΔGDP={gdp_trend:+.2f}, ΔUnemployment={unem_trend:+.2f})."
            ))
        else:
            msgs.append(DiagnosticMessage(
                "WARNING",
                f"Unusual GDP / unemployment co-movement: "
                f"ΔGDP={gdp_trend:+.2f} and ΔUnemployment={unem_trend:+.2f} "
                f"both move in the same direction."
            ))

    # ── Stability — global confirmation if no explosive warnings ──
    stab_warnings = [m for m in msgs if "explosive" in m.message]
    if not stab_warnings:
        msgs.append(DiagnosticMessage(
            "INFO", "Projection trajectory appears stable over the horizon."
        ))

    return msgs


# ═══════════════════════════════════════════════════════════════════════════
# 7. PROJECTION REPORT BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def _build_report(
    model_used:      str,
    p_optimal:       int,
    adf_pvalues:     Dict[str, float],
    differenced_vars: List[str],
    coint_tested:    bool,
    coint_rank:      int,
    johansen_details: Dict[str, Any],
    horizon_years:   List[int],
    projected:       pd.DataFrame,
    diagnostics_msgs: List[DiagnosticMessage],
    fallback_reason: Optional[str],
) -> Dict[str, Any]:
    """Assemble the structured Projection Report.

    Keys
    ----
    methodology     : human-readable description of the path taken
    steps_completed : ordered list of processing steps
    adf_results     : per-variable ADF summary
    cointegration   : Johansen details (or None if not tested)
    projection_table: list of {year: int, <var>: float, ...}
    diagnostics     : list of {level: str, message: str}
    """
    # ── Methodology string ────────────────────────────────────────────────
    method_parts = [f"Model: {model_used}"]
    if differenced_vars:
        method_parts.append(
            f"First-differenced: {', '.join(differenced_vars)}"
        )
    if fallback_reason:
        method_parts.append(f"Note: {fallback_reason}")
    methodology = " · ".join(method_parts)

    # ── Steps ────────────────────────────────────────────────────────────
    steps: List[str] = []
    steps.append(f"ADF stationarity test completed ({len(adf_pvalues)} variable(s)).")
    if differenced_vars:
        steps.append(f"First-differencing applied to: {', '.join(differenced_vars)}.")
    else:
        steps.append("All variables stationary at level — no differencing applied.")
    if coint_tested:
        steps.append(
            f"Johansen cointegration test completed. "
            f"Rank = {coint_rank} "
            f"({'VECM used' if 'VECM' in model_used else 'VAR on differences used'})."
        )
    steps.append(
        f"Projection method: {model_used} · "
        f"lag order p = {p_optimal} · "
        f"horizon = {len(horizon_years)} year(s)."
    )
    if fallback_reason:
        steps.append(f"Fallback triggered: {fallback_reason}.")

    # ── ADF results table ────────────────────────────────────────────────
    adf_rows = [
        {"variable": v, "adf_pvalue": p,
         "stationary": p < 0.10, "differenced": v in differenced_vars}
        for v, p in adf_pvalues.items()
    ]

    # ── Projection table ─────────────────────────────────────────────────
    proj_rows = []
    for yr in projected.index:
        row: Dict[str, Any] = {"year": int(yr)}
        for col in projected.columns:
            val = projected.loc[yr, col]
            row[col] = round(float(val), 6) if not np.isnan(val) else None
        proj_rows.append(row)

    # ── Diagnostics list ─────────────────────────────────────────────────
    diag_rows = [
        {"level": m.level, "message": m.message, "display": str(m)}
        for m in diagnostics_msgs
    ]

    return {
        "methodology":      methodology,
        "steps_completed":  steps,
        "adf_results":      adf_rows,
        "cointegration":    johansen_details if coint_tested else None,
        "projection_table": proj_rows,
        "diagnostics":      diag_rows,
        "model_used":       model_used,
        "p_optimal":        p_optimal,
        "coint_rank":       coint_rank,
        "horizon_years":    horizon_years,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def project_macro(
    historical: pd.DataFrame,
    variables: List[str],
    horizon_years: List[int],
    correlation_with_target: Optional[Dict[str, float]] = None,
    cfg: Optional[ProjectorConfig] = None,
) -> ProjectionResult:
    """Project the selected satellite variables over the horizon.

    Parameters
    ----------
    historical              : DataFrame indexed by year with macro vars as cols.
    variables               : Variable names to project (the satellite V*).
    horizon_years           : Integer years (e.g. [2025, 2026, 2027, 2028]).
    correlation_with_target : Optional {var: |corr| with PD} for fallback.
    cfg                     : ProjectorConfig (defaults if None).

    Returns
    -------
    ProjectionResult with projected DataFrame, diagnostics, and report.

    Guarantees
    ----------
    * Works for any country (no country-specific logic).
    * Works for any variable set (no hardcoded variable names).
    * Adapts to n_obs: AR for short samples, VAR/VECM for longer ones.
    * Never blocks the projection — diagnostics are observational only.
    """
    cfg = cfg or ProjectorConfig()
    n_steps = len(horizon_years)

    # ── Validate inputs ──────────────────────────────────────────────────
    missing = [v for v in variables if v not in historical.columns]
    if missing:
        raise ValueError(f"Variables not in historical DataFrame: {missing}")

    df = historical[variables].copy().sort_index().dropna(how="all")
    df = df.ffill().bfill().dropna()
    n_obs = len(df)
    k = len(variables)

    LOG.info("project_macro: n_obs=%d, k=%d, horizon=%d", n_obs, k, n_steps)

    # ── Degenerate case — insufficient data ──────────────────────────────
    if n_obs < cfg.min_obs_for_ar or n_steps == 0:
        LOG.warning("Insufficient obs (%d < %d) — naive flat projection",
                    n_obs, cfg.min_obs_for_ar)
        last = df.iloc[-1] if len(df) else pd.Series({v: 0.0 for v in variables})
        flat = pd.DataFrame(
            {v: [float(last[v])] * n_steps for v in variables},
            index=horizon_years,
        )
        diag_msgs = _economic_diagnostics(historical[variables], flat,
                                          cfg.diag_sigma_thresh)
        report = _build_report(
            "naive", 0, {}, [], False, 0, {}, horizon_years, flat, diag_msgs,
            fallback_reason=f"n_obs={n_obs} < min_obs_for_ar={cfg.min_obs_for_ar}",
        )
        return ProjectionResult(
            projected=flat, model_used="naive", p_optimal=0,
            fallback_reason=f"n_obs={n_obs} < min_obs_for_ar={cfg.min_obs_for_ar}",
            diagnostics_msgs=diag_msgs, report=report,
        )

    p_max = max(1, min(cfg.p_max, n_obs // cfg.p_max_div_obs))

    # ────────────────────────────────────────────────────────────────────
    # SINGLE VARIABLE PATH — always AR(p)
    # ────────────────────────────────────────────────────────────────────
    if k == 1:
        df_fit, differenced, adf_pvals = _difference_if_needed(df, cfg)
        proj, p_opt, r2 = _project_ar(df_fit[variables[0]], horizon_years, p_max)
        if proj is None:
            last_v = float(df[variables[0]].iloc[-1])
            proj = pd.Series([last_v] * n_steps, index=horizon_years)
            p_opt, r2 = 0, np.nan
        proj_df = pd.DataFrame({variables[0]: proj}, index=horizon_years)
        if differenced:
            proj_df = _reconstruct_levels(df, proj_df, differenced)

        model_used = f"AR({p_opt})"
        diag_msgs  = _economic_diagnostics(historical[variables], proj_df,
                                            cfg.diag_sigma_thresh)
        report = _build_report(
            model_used, p_opt, adf_pvals, differenced,
            False, 0, {}, horizon_years, proj_df, diag_msgs, None,
        )
        return ProjectionResult(
            projected=proj_df.astype(float), model_used=model_used,
            p_optimal=p_opt, differenced_vars=differenced,
            adf_pvalues=adf_pvals,
            fit_r2={variables[0]: round(float(r2), 4) if not np.isnan(r2) else np.nan},
            diagnostics_msgs=diag_msgs, report=report,
        )

    # ────────────────────────────────────────────────────────────────────
    # MULTIPLE VARIABLES PATH
    # ────────────────────────────────────────────────────────────────────

    # ── Step 1: ADF stationarity check ──────────────────────────────────
    _, differenced, adf_pvals = _difference_if_needed(df, cfg)
    has_i1 = len(differenced) > 0
    n_obs_eff = len(df.dropna()) - (1 if has_i1 else 0)

    # ── Step 2: choose path ──────────────────────────────────────────────

    # ── PATH A: All I(0) → VAR on levels ─────────────────────────────────
    if not has_i1:
        LOG.info("All vars stationary — attempting VAR on levels")
        if (n_obs_eff >= cfg.min_obs_for_var and
                _var_guard_passed(n_obs_eff, k, 1, cfg.var_guard_factor)):
            var_proj, p_opt, mean_r2 = _project_var(df, horizon_years, cfg, p_max)
            if var_proj is not None:
                model_used = f"VAR({p_opt})"
                diag_msgs  = _economic_diagnostics(historical[variables], var_proj,
                                                    cfg.diag_sigma_thresh)
                report = _build_report(
                    model_used, p_opt, adf_pvals, [], False, 0, {},
                    horizon_years, var_proj, diag_msgs, None,
                )
                return ProjectionResult(
                    projected=var_proj.astype(float), model_used=model_used,
                    p_optimal=p_opt, adf_pvalues=adf_pvals,
                    fit_r2={"mean_across_vars": round(mean_r2, 4)},
                    diagnostics_msgs=diag_msgs, report=report,
                )
        fallback_reason = (
            f"VAR guard failed (n_obs_eff={n_obs_eff}) — fallback to AR"
        )
        LOG.info(fallback_reason)

    # ── PATH B: Contains I(1) → Johansen test ────────────────────────────
    else:
        k_ar_diff_johansen = max(0, p_max - 1)
        coint_rank, johansen_details = _johansen_test(
            df, k_ar_diff_johansen,
            cfg.johansen_det_order, cfg.johansen_alpha,
        )
        coint_tested = True
        LOG.info("Johansen rank = %d", coint_rank)

        # ── PATH B1: Cointegrated → VECM ─────────────────────────────────
        if coint_rank >= 1:
            LOG.info("Cointegration detected — attempting VECM(rank=%d)", coint_rank)
            vecm_proj, k_ar_opt, mean_r2 = _project_vecm(
                df, horizon_years, coint_rank, cfg, p_max,
            )
            if vecm_proj is not None:
                model_used = f"VECM({k_ar_opt + 1})"
                diag_msgs  = _economic_diagnostics(historical[variables], vecm_proj,
                                                    cfg.diag_sigma_thresh)
                report = _build_report(
                    model_used, k_ar_opt + 1, adf_pvals, differenced,
                    True, coint_rank, johansen_details,
                    horizon_years, vecm_proj, diag_msgs, None,
                )
                return ProjectionResult(
                    projected=vecm_proj.astype(float), model_used=model_used,
                    p_optimal=k_ar_opt + 1, differenced_vars=differenced,
                    adf_pvalues=adf_pvals, coint_rank=coint_rank,
                    coint_tested=True, johansen_details=johansen_details,
                    fit_r2={"mean_across_vars": round(mean_r2, 4)},
                    diagnostics_msgs=diag_msgs, report=report,
                )
            fallback_reason = "VECM fit failed — fallback to VAR on differences"
            LOG.warning(fallback_reason)

        # ── PATH B2: Not cointegrated → VAR on differences ────────────────
        else:
            fallback_reason = (
                "Johansen rank=0 (no cointegration) — VAR on differences"
            )
            johansen_details["note"] = fallback_reason
            coint_tested = True

        # ── VAR on differenced data (B2 or fallback from B1) ─────────────
        df_diff = df.diff().dropna()
        n_diff  = len(df_diff)
        if (n_diff >= cfg.min_obs_for_var and
                _var_guard_passed(n_diff, k, 1, cfg.var_guard_factor)):
            var_proj, p_opt, mean_r2 = _project_var(df_diff, horizon_years, cfg, p_max)
            if var_proj is not None:
                var_proj_lev = _reconstruct_levels(df, var_proj, list(df.columns))
                model_used = f"VAR({p_opt})-diff"
                diag_msgs  = _economic_diagnostics(historical[variables], var_proj_lev,
                                                    cfg.diag_sigma_thresh)
                report = _build_report(
                    model_used, p_opt, adf_pvals, list(df.columns),
                    coint_tested, coint_rank,
                    johansen_details if coint_tested else {},
                    horizon_years, var_proj_lev, diag_msgs, fallback_reason,
                )
                return ProjectionResult(
                    projected=var_proj_lev.astype(float), model_used=model_used,
                    p_optimal=p_opt, differenced_vars=list(df.columns),
                    adf_pvalues=adf_pvals, coint_rank=coint_rank,
                    coint_tested=coint_tested,
                    johansen_details=johansen_details if coint_tested else {},
                    fit_r2={"mean_across_vars": round(mean_r2, 4)},
                    fallback_reason=fallback_reason,
                    diagnostics_msgs=diag_msgs, report=report,
                )
        fallback_reason = (
            f"VAR on differences guard failed "
            f"(n_diff={n_diff}) — fallback to AR"
        )
        LOG.info(fallback_reason)

    # ────────────────────────────────────────────────────────────────────
    # FINAL FALLBACK: AR on most correlated variable, others held flat
    # ────────────────────────────────────────────────────────────────────
    df_fit, differenced_fb, _ = _difference_if_needed(df, cfg)

    if correlation_with_target:
        primary = max(variables,
                      key=lambda v: abs(correlation_with_target.get(v, 0.0)))
        LOG.info("Fallback AR on most correlated var: '%s' (|r|=%.3f)",
                 primary, abs(correlation_with_target.get(primary, 0.0)))
    else:
        primary = variables[0]
        LOG.info("Fallback AR on first var: '%s'", primary)

    projections: Dict[str, pd.Series] = {}
    fit_r2: Dict[str, float] = {}
    p_used = 0

    for v in variables:
        if v != primary:
            last_v = float(df[v].iloc[-1])
            projections[v] = pd.Series([last_v] * n_steps, index=horizon_years)
            fit_r2[v] = np.nan
        else:
            proj, p, r2 = _project_ar(df_fit[v], horizon_years, p_max)
            if proj is None:
                last_v = float(df[v].iloc[-1])
                projections[v] = pd.Series([last_v] * n_steps, index=horizon_years)
                fit_r2[v] = np.nan
            else:
                projections[v] = proj
                fit_r2[v] = round(float(r2), 4) if not np.isnan(r2) else np.nan
                p_used = max(p_used, p)

    proj_df = pd.DataFrame(projections, index=horizon_years)
    if differenced_fb:
        proj_df = _reconstruct_levels(df, proj_df, differenced_fb)

    model_used = f"AR({p_used})-fallback"

    # Variables from outer scope: coint_tested / coint_rank may not be defined
    # if the I(0) path was taken
    _coint_tested = locals().get("coint_tested", False)
    _coint_rank   = locals().get("coint_rank", 0)
    _johansen_det = locals().get("johansen_details", {})

    diag_msgs = _economic_diagnostics(historical[variables], proj_df,
                                       cfg.diag_sigma_thresh)
    report = _build_report(
        model_used, p_used, adf_pvals,
        differenced_fb, _coint_tested, _coint_rank, _johansen_det,
        horizon_years, proj_df, diag_msgs,
        fallback_reason=fallback_reason or "AR fallback",
    )

    return ProjectionResult(
        projected=proj_df.astype(float),
        model_used=model_used,
        p_optimal=p_used,
        differenced_vars=differenced_fb,
        adf_pvalues=adf_pvals,
        coint_rank=_coint_rank,
        coint_tested=_coint_tested,
        johansen_details=_johansen_det,
        fit_r2=fit_r2,
        fallback_reason=fallback_reason or "AR fallback",
        diagnostics_msgs=diag_msgs,
        report=report,
    )
