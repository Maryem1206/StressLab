"""
model_estimation.py
===================
Estimate satellite models: OLS, Logit, Beta MLE, Vasicek-OLS.
Mise à jour : ajout du RMSE dans FittedModel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln
import statsmodels.api as sm

from .utils import EngineConfig, get_logger

log = get_logger(__name__)


@dataclass
class FittedModel:
    name: str
    family: str
    variables: List[str] = field(default_factory=list)
    coefficients: Dict[str, float] = field(default_factory=dict)
    std_errors: Dict[str, float] = field(default_factory=dict)
    pvalues: Dict[str, float] = field(default_factory=dict)
    n_obs: int = 0
    aic: float = np.nan
    bic: float = np.nan
    r2: float = np.nan
    rmse: float = np.nan
    converged: bool = True
    fail_reason: Optional[str] = None
    variable_set: str = "orig"          # "orig" | "std"
    scaler: Optional[Dict[str, Tuple[float, float]]] = None  # {col: (mean, std)}
    fitted_values: List[float] = field(default_factory=list)
    actual_values: List[float] = field(default_factory=list)
    obs_years:     List[int]   = field(default_factory=list)


# ── OLS ──────────────────────────────────────────────────────────────
def fit_ols(X: pd.DataFrame, y: pd.Series, name: str = "OLS") -> FittedModel:
    try:
        Xc = sm.add_constant(X.astype(float), has_constant="add")
        res = sm.OLS(y.astype(float), Xc).fit()
        rmse = float(np.sqrt(np.mean(res.resid ** 2)))
        return FittedModel(
            name=name, family="OLS",
            variables=list(X.columns),
            coefficients=res.params.to_dict(),
            std_errors=res.bse.to_dict(),
            pvalues=res.pvalues.to_dict(),
            n_obs=int(res.nobs),
            aic=float(res.aic), bic=float(res.bic),
            r2=float(res.rsquared),
            rmse=rmse,
            converged=True,
            fitted_values=[round(float(v), 6) for v in res.fittedvalues.tolist()],
            actual_values=[round(float(v), 6) for v in y.tolist()],
            obs_years=[int(idx) for idx in y.index.tolist()],
        )
    except Exception as e:
        return FittedModel(name=name, family="OLS",
                           variables=list(X.columns),
                           converged=False, fail_reason=str(e))


# ── Logit ────────────────────────────────────────────────────────────
def fit_logit(X: pd.DataFrame, y: pd.Series, name: str = "Logit") -> FittedModel:
    try:
        if not ((y > 0) & (y < 1)).all():
            return FittedModel(name=name, family="Logit",
                               variables=list(X.columns),
                               converged=False,
                               fail_reason="target not in (0,1)")
        Xc = sm.add_constant(X.astype(float), has_constant="add")
        res = sm.GLM(y.astype(float), Xc,
                     family=sm.families.Binomial()).fit()
        ll0 = sm.GLM(y.astype(float), np.ones((len(y), 1)),
                     family=sm.families.Binomial()).fit().llf
        pseudo_r2 = 1.0 - (res.llf / ll0) if ll0 != 0 else np.nan
        rmse = float(np.sqrt(np.mean((y.values - res.mu) ** 2)))
        return FittedModel(
            name=name, family="Logit",
            variables=list(X.columns),
            coefficients=res.params.to_dict(),
            std_errors=res.bse.to_dict(),
            pvalues=res.pvalues.to_dict(),
            n_obs=int(res.nobs),
            aic=float(res.aic), bic=float(res.bic),
            r2=float(pseudo_r2),
            rmse=rmse,
            converged=bool(res.converged) if hasattr(res, "converged") else True,
            fitted_values=[round(float(v), 6) for v in res.mu.tolist()],
            actual_values=[round(float(v), 6) for v in y.tolist()],
            obs_years=[int(idx) for idx in y.index.tolist()],
        )
    except Exception as e:
        return FittedModel(name=name, family="Logit",
                           variables=list(X.columns),
                           converged=False, fail_reason=str(e))


# ── Beta MLE ─────────────────────────────────────────────────────────
def _beta_neg_loglik(params, X, y):
    k = X.shape[1]
    beta = params[:k]
    phi = np.exp(params[k])
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
    a = mu * phi
    b = (1.0 - mu) * phi
    ll = (gammaln(phi) - gammaln(a) - gammaln(b)
          + (a - 1.0) * np.log(y) + (b - 1.0) * np.log(1.0 - y))
    return -float(np.sum(ll))


def _numerical_hessian(f, x, args, eps=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = (f(xpp, *args) - f(xpm, *args)
                       - f(xmp, *args) + f(xmm, *args)) / (4 * eps * eps)
            H[j, i] = H[i, j]
    return H


def fit_beta(X: pd.DataFrame, y: pd.Series, name: str = "Beta") -> FittedModel:
    try:
        if not ((y > 0) & (y < 1)).all():
            return FittedModel(name=name, family="Beta",
                               variables=list(X.columns),
                               converged=False,
                               fail_reason="target not in (0,1)")
        Xc = sm.add_constant(X.astype(float), has_constant="add")
        col_names = list(Xc.columns)
        Xa = Xc.values
        ya = y.astype(float).values
        n, k = Xa.shape

        logit_y = np.log(ya / (1.0 - ya))
        beta0, *_ = np.linalg.lstsq(Xa, logit_y, rcond=None)
        x0 = np.concatenate([beta0, [np.log(10.0)]])

        result = optimize.minimize(
            _beta_neg_loglik, x0, args=(Xa, ya),
            method="L-BFGS-B", options={"maxiter": 500, "disp": False},
        )
        if not result.success:
            return FittedModel(name=name, family="Beta",
                               variables=list(X.columns),
                               converged=False,
                               fail_reason=f"optimizer: {result.message}")

        params = result.x
        ll = -result.fun
        try:
            H = _numerical_hessian(_beta_neg_loglik, params, (Xa, ya))
            cov = np.linalg.inv(H)
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except Exception:
            se = np.full_like(params, np.nan)

        beta_hat = params[:k]
        beta_se = se[:k]
        z = np.where(beta_se > 0, beta_hat / beta_se, np.nan)
        pvals = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))

        n_params = k + 1
        aic = -2.0 * ll + 2.0 * n_params
        bic = -2.0 * ll + n_params * np.log(n)

        eta_hat = Xa @ beta_hat
        mu_hat = 1.0 / (1.0 + np.exp(-np.clip(eta_hat, -50, 50)))
        pseudo_r2 = float(np.corrcoef(mu_hat, ya)[0, 1] ** 2) \
            if np.std(mu_hat) > 0 and np.std(ya) > 0 else np.nan
        rmse = float(np.sqrt(np.mean((ya - mu_hat) ** 2)))

        return FittedModel(
            name=name, family="Beta",
            variables=list(X.columns),
            coefficients=dict(zip(col_names, beta_hat.tolist())),
            std_errors=dict(zip(col_names, beta_se.tolist())),
            pvalues=dict(zip(col_names, pvals.tolist())),
            n_obs=n, aic=float(aic), bic=float(bic),
            r2=float(pseudo_r2), rmse=rmse,
            converged=True,
            fitted_values=[round(float(v), 6) for v in mu_hat.tolist()],
            actual_values=[round(float(v), 6) for v in ya.tolist()],
            obs_years=[int(idx) for idx in y.index.tolist()],
        )
    except Exception as e:
        return FittedModel(name=name, family="Beta",
                           variables=list(X.columns),
                           converged=False, fail_reason=str(e))


# ── Vasicek-OLS ──────────────────────────────────────────────────────
def fit_vasicek_ols(X: pd.DataFrame, y: pd.Series,
                    name: str = "Vasicek-OLS") -> FittedModel:
    try:
        if not ((y > 0) & (y < 1)).all():
            return FittedModel(name=name, family="Vasicek-OLS",
                               variables=list(X.columns),
                               converged=False,
                               fail_reason="target not in (0,1)")
        z = pd.Series(stats.norm.ppf(y.astype(float).values), index=y.index)
        fm = fit_ols(X, z, name=name)
        fm.family = "Vasicek-OLS"
        # Convert fitted values back to PD scale via norm.cdf
        fm.fitted_values = [round(float(stats.norm.cdf(v)), 6) for v in fm.fitted_values]
        fm.actual_values = [round(float(v), 6) for v in y.tolist()]
        fm.obs_years     = [int(idx) for idx in y.index.tolist()]
        return fm
    except Exception as e:
        return FittedModel(name=name, family="Vasicek-OLS",
                           variables=list(X.columns),
                           converged=False, fail_reason=str(e))


# ── Driver ───────────────────────────────────────────────────────────
def estimate_all(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: EngineConfig,
    model_tag: str = "model",
    variable_set: str = "orig",
    scaler: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[FittedModel]:
    """
    Estime Logit, Beta et Vasicek-OLS sur le subset X, y.
    OLS exclu — uniquement des familles adaptées aux cibles bornées (0,1).

    Parameters
    ----------
    variable_set : "orig" ou "std" — indique si X est standardisé
    scaler       : {col: (mean, std)} utilisé lors de la projection pour
                   ré-appliquer la standardisation sur les données futures
    """
    out: List[FittedModel] = []

    # Values > 1 indicate percentage format — genuinely incompatible with these families.
    if (y > 1).any():
        log.warning(
            "estimate_all [%s]: target contient des valeurs > 1 (format pourcentage ?) — "
            "Logit/Beta/Vasicek impossibles. Subset ignoré.", model_tag
        )
        return []

    # Clip exact boundary values (y=0 is common in PD data with zero-default years).
    # Standard Smithson-Verkuilen transformation for bounded models.
    _eps = 1e-7
    if not ((y > 0) & (y < 1)).all():
        log.info(
            "estimate_all [%s]: target contient des valeurs à 0 ou 1 — "
            "clipping vers (%.0e, 1-%.0e) pour permettre l'estimation.", model_tag, _eps, _eps
        )
        y = y.clip(_eps, 1 - _eps)

    for fit_fn, family in [
        (fit_logit,      "Logit"),
        (fit_beta,       "Beta"),
        (fit_vasicek_ols, "Vasicek-OLS"),
    ]:
        m = fit_fn(X, y, name=f"{model_tag}|{family}")
        m.variable_set = variable_set
        m.scaler = scaler
        out.append(m)

    converged = [m for m in out if m.converged]
    for m in out:
        if not m.converged:
            log.info("Skipping %s (failed: %s).", m.name, m.fail_reason)
    return converged
