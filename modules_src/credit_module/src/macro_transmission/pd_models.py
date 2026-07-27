"""PD satellite models — three families with a unified interface.

Universal, country-agnostic. All three models share the API:

    model.fit(X, y) -> self
    model.predict(X) -> pd.Series in (0, 1)
    model.params_                      # dict feature -> coefficient
    model.intercept_                   # float
    model.diagnostics_                 # dict with r2, rmse, aic, n_obs, pvalues, ...

Implemented families:

* Beta     : Beta regression à la Ferrari & Cribari-Neto (2004).
             Logit link on the mean. Direct MLE via scipy.
             Reference: Ferrari, S. and Cribari-Neto, F. (2004).
             "Beta Regression for Modelling Rates and Proportions",
             Journal of Applied Statistics 31(7), pp. 799-815.
* Logit    : OLS on logit(y), inverted at predict time.
* Vasicek  : OLS on probit(y) = Phi^{-1}(y), inverted at predict time.
             Naming follows Vasicek (2002) loan portfolio framework
             where the systematic factor enters via Phi^{-1}.

All families accept a numerical pandas DataFrame X and a pandas Series y in (0, 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

EPS = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1.0 - p))


def _inv_logit(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def _probit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return norm.ppf(p)


def _inv_probit(z: np.ndarray) -> np.ndarray:
    return norm.cdf(z)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class _BasePDModel:
    family: str = "base"
    link: str = "identity"

    def __init__(self) -> None:
        self.features_: List[str] = []
        self.intercept_: float = 0.0
        self.params_: Dict[str, float] = {}
        self.diagnostics_: Dict[str, Any] = {}
        self.fitted_: bool = False

    # public API to be implemented by subclasses
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_BasePDModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    # Diagnostics computed from in-sample y_hat
    def _store_diagnostics(self, y: np.ndarray, y_hat: np.ndarray,
                           n_params: int, extra: Dict[str, Any] | None = None) -> None:
        n = len(y)
        resid = y - y_hat
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + EPS
        r2 = 1.0 - ss_res / ss_tot
        rmse = float(np.sqrt(ss_res / max(n, 1)))
        # Gaussian-like AIC for ranking (consistent across families)
        sigma2 = max(ss_res / max(n, 1), EPS)
        loglik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)
        aic = 2 * n_params - 2 * loglik
        bic = n_params * np.log(max(n, 1)) - 2 * loglik
        diag = {
            "n_obs": int(n),
            "n_params": int(n_params),
            "r2": float(r2),
            "rmse": float(rmse),
            "aic": float(aic),
            "bic": float(bic),
            "loglik": float(loglik),
        }
        if extra:
            diag.update(extra)
        self.diagnostics_ = diag

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "link": self.link,
            "features": list(self.features_),
            "intercept": float(self.intercept_),
            "params": {k: float(v) for k, v in self.params_.items()},
            "diagnostics": self.diagnostics_,
        }


# ---------------------------------------------------------------------------
# Logit family — OLS on logit(y)
# ---------------------------------------------------------------------------

class LogitPDModel(_BasePDModel):
    family = "logit"
    link = "logit"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogitPDModel":
        self.features_ = list(X.columns)
        X_const = sm.add_constant(X.astype(float).values, has_constant="add")
        y_t = _logit(y.astype(float).values)
        ols = sm.OLS(y_t, X_const).fit()
        coefs = ols.params
        self.intercept_ = float(coefs[0])
        self.params_ = {f: float(c) for f, c in zip(self.features_, coefs[1:])}
        # Predicted in original (0,1) space
        y_hat = _inv_logit(ols.predict(X_const))
        pvals = list(ols.pvalues)
        extra = {
            "pvalues": {f: float(p) for f, p in zip(self.features_, pvals[1:])},
            "intercept_pvalue": float(pvals[0]),
            "f_pvalue": float(ols.f_pvalue) if ols.f_pvalue is not None else float("nan"),
            "ols_r2": float(ols.rsquared),  # R^2 in transformed space
            "ols_aic": float(ols.aic),
        }
        self._store_diagnostics(y.astype(float).values, y_hat,
                                n_params=len(self.features_) + 1, extra=extra)
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        z = self.intercept_ + np.zeros(len(X))
        for f in self.features_:
            z = z + X[f].astype(float).values * self.params_[f]
        return pd.Series(_inv_logit(z), index=X.index, name="pd_pit")


# ---------------------------------------------------------------------------
# Vasicek family — OLS on probit(y)
# ---------------------------------------------------------------------------

class VasicekPDModel(_BasePDModel):
    family = "vasicek"
    link = "probit"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "VasicekPDModel":
        self.features_ = list(X.columns)
        X_const = sm.add_constant(X.astype(float).values, has_constant="add")
        y_t = _probit(y.astype(float).values)
        ols = sm.OLS(y_t, X_const).fit()
        coefs = ols.params
        self.intercept_ = float(coefs[0])
        self.params_ = {f: float(c) for f, c in zip(self.features_, coefs[1:])}
        y_hat = _inv_probit(ols.predict(X_const))
        pvals = list(ols.pvalues)
        extra = {
            "pvalues": {f: float(p) for f, p in zip(self.features_, pvals[1:])},
            "intercept_pvalue": float(pvals[0]),
            "f_pvalue": float(ols.f_pvalue) if ols.f_pvalue is not None else float("nan"),
            "ols_r2": float(ols.rsquared),
            "ols_aic": float(ols.aic),
        }
        self._store_diagnostics(y.astype(float).values, y_hat,
                                n_params=len(self.features_) + 1, extra=extra)
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        z = self.intercept_ + np.zeros(len(X))
        for f in self.features_:
            z = z + X[f].astype(float).values * self.params_[f]
        return pd.Series(_inv_probit(z), index=X.index, name="pd_pit")


# ---------------------------------------------------------------------------
# Beta family — Beta regression (Ferrari & Cribari-Neto, 2004) by direct MLE
# ---------------------------------------------------------------------------

def _beta_neg_loglik(theta: np.ndarray, X_const: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood of Beta regression with logit link.

    Parametrization: y ~ Beta(mu * phi, (1 - mu) * phi)
    with mu = inv_logit(X @ beta), phi > 0.
    """
    p = X_const.shape[1]
    beta = theta[:p]
    log_phi = theta[p]
    phi = np.exp(log_phi)  # ensures phi > 0

    eta = X_const @ beta
    mu = _inv_logit(eta)
    mu = np.clip(mu, EPS, 1 - EPS)

    a = mu * phi
    b = (1.0 - mu) * phi
    y_clipped = np.clip(y, EPS, 1 - EPS)

    ll = (gammaln(a + b) - gammaln(a) - gammaln(b)
          + (a - 1.0) * np.log(y_clipped)
          + (b - 1.0) * np.log(1.0 - y_clipped))
    return -float(np.sum(ll))


class BetaPDModel(_BasePDModel):
    family = "beta"
    link = "logit"

    def __init__(self) -> None:
        super().__init__()
        self.phi_: float = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BetaPDModel":
        self.features_ = list(X.columns)
        X_const = sm.add_constant(X.astype(float).values, has_constant="add")
        y_arr = y.astype(float).values
        p = X_const.shape[1]

        # Warm-start from logit OLS
        warm = sm.OLS(_logit(y_arr), X_const).fit()
        beta0 = list(warm.params)
        # Method-of-moments style start for log(phi)
        mu0 = _inv_logit(X_const @ np.array(beta0))
        var0 = max(np.var(y_arr - mu0), EPS)
        phi0 = max(1.0, (np.mean(mu0 * (1.0 - mu0)) / var0) - 1.0)
        theta0 = np.array(beta0 + [np.log(phi0)])

        res = minimize(_beta_neg_loglik, theta0, args=(X_const, y_arr),
                       method="L-BFGS-B")
        if not res.success:
            # Fallback: use warm start
            theta = theta0
        else:
            theta = res.x
        beta = theta[:p]
        self.phi_ = float(np.exp(theta[p]))
        self.intercept_ = float(beta[0])
        self.params_ = {f: float(c) for f, c in zip(self.features_, beta[1:])}

        # In-sample fit + Wald-style p-values (approximate, via numerical Hessian if possible)
        eta = X_const @ beta
        y_hat = _inv_logit(eta)

        # Numerical standard errors via Hessian inverse — fallback to NaN
        pvals = self._approx_pvalues(theta, X_const, y_arr)
        extra = {
            "phi": float(self.phi_),
            "pvalues": {f: float(p_) for f, p_ in zip(self.features_, pvals[1:p])},
            "intercept_pvalue": float(pvals[0]),
            "loglik_beta": float(-res.fun) if res.success else float("nan"),
            "converged": bool(res.success),
        }
        self._store_diagnostics(y_arr, y_hat,
                                n_params=p + 1, extra=extra)
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        z = self.intercept_ + np.zeros(len(X))
        for f in self.features_:
            z = z + X[f].astype(float).values * self.params_[f]
        return pd.Series(_inv_logit(z), index=X.index, name="pd_pit")

    @staticmethod
    def _approx_pvalues(theta: np.ndarray, X_const: np.ndarray, y: np.ndarray
                        ) -> List[float]:
        """Numerical Hessian -> approximate Wald p-values for the Beta MLE."""
        p_total = len(theta)
        eps = 1e-4
        H = np.zeros((p_total, p_total))
        f0 = _beta_neg_loglik(theta, X_const, y)
        for i in range(p_total):
            for j in range(i, p_total):
                t_pp = theta.copy(); t_pp[i] += eps; t_pp[j] += eps
                t_pm = theta.copy(); t_pm[i] += eps; t_pm[j] -= eps
                t_mp = theta.copy(); t_mp[i] -= eps; t_mp[j] += eps
                t_mm = theta.copy(); t_mm[i] -= eps; t_mm[j] -= eps
                f_pp = _beta_neg_loglik(t_pp, X_const, y)
                f_pm = _beta_neg_loglik(t_pm, X_const, y)
                f_mp = _beta_neg_loglik(t_mp, X_const, y)
                f_mm = _beta_neg_loglik(t_mm, X_const, y)
                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
                H[j, i] = H[i, j]
        try:
            cov = np.linalg.inv(H)
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            se = np.full(p_total, np.nan)
        # Beta coefficients = first p (excluding the log_phi term)
        from scipy.stats import norm as _n
        z = theta[:-1] / np.where(se[:-1] > 0, se[:-1], np.nan)
        pvals = 2 * (1 - _n.cdf(np.abs(z)))
        return list(pvals)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PD_MODEL_FAMILIES: Dict[str, type] = {
    "beta": BetaPDModel,
    "logit": LogitPDModel,
    "vasicek": VasicekPDModel,
}


def build_pd_model(family: str) -> _BasePDModel:
    if family not in PD_MODEL_FAMILIES:
        raise ValueError(f"Unknown PD model family: {family}. "
                         f"Available: {list(PD_MODEL_FAMILIES.keys())}")
    return PD_MODEL_FAMILIES[family]()
