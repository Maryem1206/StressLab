"""
cross_risk_selector.py
=======================
Selects, from a pool of candidate historical macro variables, only those
independently accepted by ALL THREE risk modules: PD (crédit), AND at least
one liquidity satellite target (LCR/NSFR), AND at least one market beta
factor (beta1/2/3).

Rationale: without this filter, credit/liquidity/market each pick whatever
macro variable best fits their own target in isolation — the same NGFS
scenario could end up driving credit stress through "unemployment_rate"
while market stress is driven through "fiscal_balance_gdp", with no shared
macro narrative tying the three together. This module enforces a single,
cross-validated driver set instead.

Used only by climate/wrapper.py before calling the credit PD tournament
(macro_selection_engine.historical_validation.run_stage2, via
EngineConfig.restrict_candidate_vars) — it does not touch the standalone
credit/liquidity/market dashboards' own variable selection.

The acceptance test here is intentionally a lightweight, single-variable
screening gate (sign + p-value), not a full model fit — the real model
fitting (transforms, VIF, backward elimination, tournament) still happens
downstream in each module's own pipeline, just restricted to this
pre-agreed pool.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .utils import expected_sign

log = logging.getLogger("climate.cross_risk_selector")

_PVALUE_THRESHOLD = 0.20  # lenient — screening gate, not the final model
_MIN_OBS = 6


def _univariate_accept(x: pd.Series, y: pd.Series, sign_prior: int) -> bool:
    """Single-variable OLS screen: sign consistent with prior + p-value below threshold."""
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < _MIN_OBS or df["x"].std() == 0 or df["y"].std() == 0:
        return False
    X = sm.add_constant(df["x"].astype(float), has_constant="add")
    try:
        res = sm.OLS(df["y"].astype(float), X).fit()
    except Exception:
        return False
    coef = float(res.params.get("x", 0.0))
    pval = float(res.pvalues.get("x", 1.0))
    if pval >= _PVALUE_THRESHOLD:
        return False
    if sign_prior != 0 and np.sign(coef) != np.sign(sign_prior):
        return False
    return True


def select_unanimous_variables(
    macro_df: pd.DataFrame,
    pd_series: pd.Series,
    liquidity_targets: Dict[str, pd.Series],
    market_targets: Dict[str, pd.Series],
) -> List[str]:
    """
    Parameters
    ----------
    macro_df          : candidate historical macro pool (index=year, columns=variable)
    pd_series         : historical PD / Default rate series (index=year)
    liquidity_targets : {name: series}, e.g. {"lcr": ..., "nsfr": ...} — pass {}
                        to skip the liquidity gate (treated as always-pass;
                        see module docstring on graceful degradation)
    market_targets    : {name: series}, e.g. {"beta1": ..., "beta2": ..., "beta3": ...}
                        — pass {} to skip the market gate

    Returns
    -------
    Column names accepted by PD AND (liquidity_targets empty OR at least one
    liquidity target) AND (market_targets empty OR at least one market target).
    """
    accepted: List[str] = []
    for col in macro_df.columns:
        x = macro_df[col]

        if not _univariate_accept(x, pd_series, expected_sign(col, "credit")):
            continue

        if liquidity_targets:
            liq_ok = any(
                _univariate_accept(x, series, expected_sign(col, "liquidity"))
                for series in liquidity_targets.values()
            )
            if not liq_ok:
                continue

        if market_targets:
            mkt_ok = False
            for name, series in market_targets.items():
                risk_key = f"market_{name}" if f"market_{name}" else "market"
                sign_prior = expected_sign(col, risk_key)
                if sign_prior == 0:
                    sign_prior = expected_sign(col, "market")
                if _univariate_accept(x, series, sign_prior):
                    mkt_ok = True
                    break
            if not mkt_ok:
                continue

        accepted.append(col)

    log.info(
        "Cross-risk unanimous selection: %d/%d variables accepted by PD%s%s: %s",
        len(accepted), len(macro_df.columns),
        " + liquidité" if liquidity_targets else "",
        " + marché" if market_targets else "",
        accepted,
    )
    return accepted
