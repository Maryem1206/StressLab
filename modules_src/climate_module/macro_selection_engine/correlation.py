"""
correlation.py
==============
Pearson correlation between target and HISTORICAL macro variables (Stage 2).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .utils import EngineConfig, get_logger

log = get_logger(__name__)


def correlation_with_target(X: pd.DataFrame, y: pd.Series,
                            cfg: EngineConfig
                            ) -> Tuple[pd.DataFrame, List[str]]:
    """Compute Pearson correlations of each X column with y; rank by |corr|.

    Returns (correlation_table, kept_variables).
    """
    rows = []
    for c in X.columns:
        x = X[c].astype(float)
        if x.std() == 0:
            r = np.nan
        else:
            r = float(np.corrcoef(x, y.astype(float))[0, 1])
        rows.append({"variable": c, "pearson_r": r,
                     "abs_r": abs(r) if pd.notna(r) else np.nan})

    table = (pd.DataFrame(rows)
             .dropna(subset=["pearson_r"])
             .sort_values("abs_r", ascending=False)
             .reset_index(drop=True))

    kept = table.loc[table["abs_r"] >= cfg.min_abs_correlation,
                     "variable"].tolist()
    log.info("Correlation filter: %d/%d variables pass |r|>=%.2f.",
             len(kept), len(table), cfg.min_abs_correlation)
    return table, kept


def correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    return X.corr(method="pearson")
