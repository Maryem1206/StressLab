"""
core/utils.py
=============
Shared platform utilities reusable across all risk modules.

Contents
--------
- FrequencyAligner : aligns mixed-frequency time series to a monthly DatetimeIndex
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

LOG = logging.getLogger("core.utils")


class FrequencyAligner:
    """Align mixed-frequency time series to a common monthly DatetimeIndex.

    All series are reindexed to the provided monthly index via forward-fill
    (ffill), then optionally backward-filled for leading NaN values (bfill).
    Series already at monthly frequency are matched by date; annual or
    quarterly series are broadcast to every month of their period.

    Parameters
    ----------
    monthly_index : pd.DatetimeIndex
        Target monthly index (e.g. from pd.date_range(..., freq='MS')).
    bfill_leading : bool
        Back-fill leading NaN values produced by reindex (default True).
    ffill_limit : int, optional
        Maximum number of consecutive months a forward-fill may bridge
        (default None = unlimited, appropriate for annual/quarterly series
        that are intentionally broadcast across every month of their
        period). Pass a small integer (e.g. 2) for series that are
        already observed at monthly frequency, so a multi-year reporting
        gap is left as NaN instead of being silently frozen at the last
        known value.

    Usage
    -----
    aligner = FrequencyAligner(monthly_index)
    df = aligner.align({"inflation": annual_series, "policy_rate": monthly_s})

    Notes
    -----
    Standard practice in central bank macro models (IMF WEO, BIS, EBA).
    Shared across credit, liquidity, climate, and market risk modules.
    """

    def __init__(
        self,
        monthly_index: "pd.DatetimeIndex",
        bfill_leading: bool = True,
        ffill_limit: "Optional[int]" = None,
    ) -> None:
        if not isinstance(monthly_index, pd.DatetimeIndex):
            raise TypeError("monthly_index must be a pd.DatetimeIndex")
        self.monthly_index = monthly_index.normalize()
        self.bfill_leading = bfill_leading
        self.ffill_limit = ffill_limit

    def align_series(self, series: "pd.Series", name: str = "") -> "pd.Series":
        """Align a single series to the monthly index via forward-fill.

        Parameters
        ----------
        series : pd.Series
            Input series with any DatetimeIndex or integer-year index.
        name : str
            Column name for the returned series.

        Returns
        -------
        pd.Series indexed by self.monthly_index.
        """
        s = series.copy()
        label = name or (str(series.name) if series.name is not None else "")

        if not isinstance(s.index, pd.DatetimeIndex):
            try:
                s.index = pd.to_datetime(s.index.astype(str))
            except Exception as exc:
                LOG.warning(
                    "FrequencyAligner: cannot convert index for '%s': %s -- "
                    "returning all-NaN series",
                    label, exc,
                )
                return pd.Series(index=self.monthly_index, dtype=float, name=label)
        s.index = s.index.normalize()

        combined_idx = s.index.union(self.monthly_index).sort_values()
        s = s.reindex(combined_idx)
        s = s.ffill(limit=self.ffill_limit)
        if self.bfill_leading:
            # Only back-fill NaNs that precede the first observation (true
            # "leading" gap). An unbounded s.bfill() here would also fill
            # interior gaps left NaN on purpose by a finite ffill_limit,
            # bridging them from the wrong side instead of leaving them NaN.
            first_valid = s.first_valid_index()
            if first_valid is not None:
                leading = s.index < first_valid
                s.loc[leading] = s.bfill().loc[leading]
        s = s.reindex(self.monthly_index)
        s.name = label
        return s

    def align(self, series_dict: "Dict[str, pd.Series]") -> "pd.DataFrame":
        """Align multiple series and return a single monthly DataFrame.

        Parameters
        ----------
        series_dict : dict
            {column_name: pd.Series} -- series may have different frequencies.

        Returns
        -------
        pd.DataFrame indexed by self.monthly_index, one column per input series.
        """
        aligned = {n: self.align_series(s, name=n) for n, s in series_dict.items()}
        df = pd.DataFrame(aligned, index=self.monthly_index)
        all_nan_cols = [c for c in df.columns if df[c].isna().all()]
        if all_nan_cols:
            LOG.warning(
                "FrequencyAligner: %d column(s) entirely NaN after alignment: %s",
                len(all_nan_cols), all_nan_cols,
            )
        return df