"""
stepwise.py
===========
Backward elimination, mixing three criteria:
  1) variables whose coefficient sign violates the economic prior are
     removed first (highest priority);
  2) then the variable with the largest p-value above threshold;
  3) we stop dropping once the new BIC stops improving AND no p-value
     violation remains.

The procedure operates on an OLS fit; it is a *screening* step before the
multi-model estimation in `model_estimation.py`.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .utils import EngineConfig, expected_sign, get_logger

log = get_logger(__name__)


def _fit_ols(X: pd.DataFrame, y: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(y.astype(float),
                  sm.add_constant(X.astype(float), has_constant="add")).fit()


def backward_elimination(X: pd.DataFrame, y: pd.Series,
                         cfg: EngineConfig
                         ) -> Tuple[List[str], pd.DataFrame]:
    """Run backward elimination. Returns (kept_variables, trace_table)."""
    work = X.copy()
    trace_rows = []
    step = 0
    while work.shape[1] >= 1:
        step += 1
        fit = _fit_ols(work, y)
        pvals = fit.pvalues.drop("const", errors="ignore")
        coefs = fit.params.drop("const", errors="ignore")
        bic = float(fit.bic)

        # 1) economic sign violations
        sign_violators = []
        for v in work.columns:
            exp = expected_sign(v, cfg.risk_type)
            if exp == 0:
                continue
            # NB: target convention for liquidity is inverted in priors.
            if np.sign(coefs.get(v, 0.0)) != 0 and \
               np.sign(coefs.get(v, 0.0)) != exp:
                sign_violators.append(v)

        # 2) p-value violators
        pval_violators = pvals[pvals > cfg.pvalue_threshold].index.tolist()

        # 3) near-zero standardized coefficient
        # β_std_i = |β_i| × σ(x_i) / σ(y) — mesure la contribution réelle
        # en unités de l'écart-type de la cible, indépendamment de l'échelle.
        # Si β_std < seuil OU contribution relative < seuil → coefficient nul.
        sigma_y = float(y.std()) if float(y.std()) > 1e-12 else 1.0
        std_thresh   = getattr(cfg, "zero_coef_std_threshold",   0.01)
        contrib_thresh = getattr(cfg, "zero_coef_contrib_threshold", 0.02)

        beta_std = {
            v: abs(float(coefs.get(v, 0.0))) * float(work[v].std(ddof=1)) / sigma_y
            for v in work.columns
        }
        total_contrib = sum(beta_std.values()) or 1.0
        contrib_rel   = {v: beta_std[v] / total_contrib for v in work.columns}

        zero_coef_violators = [
            v for v in work.columns
            if beta_std[v] < std_thresh or contrib_rel[v] < contrib_thresh
        ]

        trace_rows.append({
            "step": step,
            "n_vars": work.shape[1],
            "bic": bic,
            "rsquared_adj": float(fit.rsquared_adj),
            "sign_violators": list(sign_violators),
            "pval_violators": list(pval_violators),
            "zero_coef_violators": list(zero_coef_violators),
        })

        # decide what to drop — priorité : signe > coef nul > p-value
        drop = None
        if sign_violators:
            # drop the violator with the largest p-value (least supported)
            drop = max(sign_violators, key=lambda v: float(pvals.get(v, 1.0)))
            log.info("Backward step %d: drop '%s' (sign violation).",
                     step, drop)
        elif zero_coef_violators:
            # drop the variable with the lowest standardized contribution
            drop = min(zero_coef_violators, key=lambda v: contrib_rel[v])
            log.info(
                "Backward step %d: drop '%s' (β_std=%.4f, contrib=%.1f%%).",
                step, drop, beta_std[drop], contrib_rel[drop] * 100,
            )
        elif pval_violators:
            drop = max(pval_violators, key=lambda v: float(pvals[v]))
            log.info("Backward step %d: drop '%s' (p=%.3f > %.2f).",
                     step, drop, float(pvals[drop]), cfg.pvalue_threshold)
        else:
            log.info("Backward step %d: no violators — stop.", step)
            break

        # safety: if dropping would empty the model, stop instead
        candidate = work.drop(columns=[drop])
        if candidate.shape[1] == 0:
            log.info("Backward: would drop the last variable; stopping.")
            break

        # accept the drop only if BIC improves OR a sign violation existed
        new_fit = _fit_ols(candidate, y)
        if sign_violators or new_fit.bic < bic:
            work = candidate
        else:
            log.info("Backward: dropping '%s' would worsen BIC (%.2f -> %.2f) "
                     "and no sign violation — stopping.",
                     drop, bic, float(new_fit.bic))
            break

    return work.columns.tolist(), pd.DataFrame(trace_rows)
