"""
vif.py
======
Variance Inflation Factor: iterative removal of the worst offender until
all remaining variables are below threshold.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from .utils import EngineConfig, get_logger

log = get_logger(__name__)


def _vif_table(X: pd.DataFrame) -> pd.DataFrame:
    if X.shape[1] < 2:
        return pd.DataFrame({"variable": X.columns, "vif": [1.0] * X.shape[1]})
    X_const = add_constant(X, has_constant="add").values
    vifs = []
    # column 0 in X_const is the constant; skip it
    for i, col in enumerate(X.columns, start=1):
        try:
            v = variance_inflation_factor(X_const, i)
        except Exception:
            v = np.inf
        vifs.append({"variable": col, "vif": v})
    return pd.DataFrame(vifs).sort_values("vif", ascending=False).reset_index(drop=True)


def remove_high_vif(X: pd.DataFrame,
                    cfg: EngineConfig
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Iteratively drop the variable with the largest VIF until all <= threshold.

    Returns (reduced_X, final_vif_table, removed_variables).
    """
    work = X.copy()
    removed: List[str] = []
    while work.shape[1] >= 2:
        tbl = _vif_table(work)
        worst = tbl.iloc[0]
        if worst["vif"] <= cfg.vif_threshold:
            break
        log.info("VIF: dropping '%s' (VIF=%.2f > %.2f).",
                 worst["variable"], worst["vif"], cfg.vif_threshold)
        removed.append(str(worst["variable"]))
        work = work.drop(columns=[worst["variable"]])
    final = _vif_table(work) if work.shape[1] >= 1 else pd.DataFrame()
    log.info("VIF: %d variable(s) remain after multicollinearity filter.",
             work.shape[1])
    return work, final, removed
