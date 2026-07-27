"""
preprocessing.py
================
Time-series alignment and missing-value handling for the HISTORICAL frame
(Stage 2 input). NGFS data is NOT aligned with historical data — they are
processed independently per the engine's two-stage design.

Frequency: annual (per project convention).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .utils import EngineConfig, get_logger

log = get_logger(__name__)


def align_and_clean(hist_df: pd.DataFrame,
                    candidate_vars: List[str],
                    cfg: EngineConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """Build the design matrix X and target y from the historical frame.

    Steps
    -----
    1. Subset to year + candidate_vars + target.
    2. Sort by year, set as index.
    3. Linear interpolation of internal NaNs (annual data),
       no extrapolation of leading/trailing gaps -> drop those rows.
    4. Drop columns that are still all-NaN or constant.
    """
    needed = [cfg.hist_year_col, cfg.target_variable] + list(candidate_vars)
    missing = [c for c in needed if c not in hist_df.columns]
    if missing:
        raise ValueError(f"Historical frame missing required columns: {missing}")

    df = hist_df[needed].copy()
    df = df.sort_values(cfg.hist_year_col).set_index(cfg.hist_year_col)

    # Coerce any object/str columns to numeric before interpolation.
    # pandas 2.x raises TypeError on interpolate() for non-numeric dtypes.
    # Also handles European number format (comma decimal) and % signs.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        cleaned = (
            df[col].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(r"\s", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(cleaned, errors="coerce")

    # Interpolate internal NaNs (annual). Do NOT extrapolate.
    df = df.interpolate(method="linear", limit_direction="both",
                        limit_area="inside")

    # Drop rows with any remaining NaNs (typically leading/trailing).
    n_before = len(df)
    df = df.dropna(how="any")
    n_after = len(df)
    if n_after < n_before:
        log.info("Dropped %d rows with leading/trailing NaNs.", n_before - n_after)

    if n_after < 5:
        raise ValueError(
            f"Too few aligned observations after cleaning: {n_after}. "
            f"Need >=5 for any reasonable regression."
        )

    # Drop columns with zero variance (uninformative).
    nunique = df.nunique()
    constants = nunique[nunique <= 1].index.tolist()
    if constants:
        log.warning("Dropping constant columns: %s", constants)
        df = df.drop(columns=constants)

    y = df[cfg.target_variable]
    X = df.drop(columns=[cfg.target_variable])
    log.info("Stage 2 design matrix: %d obs, %d candidate variables.",
             X.shape[0], X.shape[1])
    return X, y


def standardize(X: pd.DataFrame) -> pd.DataFrame:
    """Z-score each column. Useful for VIF / coefficient comparability."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0).replace(0, np.nan)
    return (X - mu) / sd
