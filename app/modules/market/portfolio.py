"""
portfolio.py
============
Synthetic fixed-income portfolio construction and repricing.

SyntheticPortfolioBuilder
    Builds the CIB Egypt portfolio snapshot (31/12/2023) from two
    configurable weight tables (T-Bills and T-Bonds).  All parameters
    are generic -- no hard-coded country or bank.

PortfolioRepricer
    Computes DP for each instrument under a yield shock vector Dy(tau).
    Method "auto": full revaluation if |Dy_max| > 200 bp, else duration.

References
----------
Fabozzi, F.J. (2007). Fixed Income Mathematics. McGraw-Hill.
BCBS 238 (2013) -- LCR methodology.
BCBS Basel II.5 (2009) -- stressed VaR.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("market.portfolio")


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------

@dataclass
class Instrument:
    """Single sovereign fixed-income instrument."""
    label:           str
    maturity_years:  float
    notional_egp_bn: float      # face value in EGP billions
    coupon_rate:     float      # annual coupon in % (0 for T-bills)
    is_tbill:        bool       # True -> discount instrument (single CF at maturity)

    # ---- computed properties ----

    @property
    def modified_duration(self) -> float:
        """Modified duration in years (exact for bullet bonds, exact for T-bills)."""
        n = self.maturity_years
        c = self.coupon_rate / 100.0   # decimal
        if self.is_tbill or c < 1e-8:
            return n                    # T-bill: single CF => D_mod = maturity
        # Macaulay duration of a par-priced annual-coupon bond
        # D_mac = (1+y)/y - n / ((1+y)^n - 1)   where y ~ c (par assumption)
        y = max(c, 1e-6)
        try:
            mac = (1.0 + y) / y - n / ((1.0 + y) ** n - 1.0)
        except (ZeroDivisionError, OverflowError):
            mac = n
        return mac / (1.0 + y)

    @property
    def bpv_egp_m(self) -> float:
        """BPV in EGP millions = D_mod x Notional_EGP_M x 0.0001."""
        return self.modified_duration * self.notional_egp_bn * 1_000.0 * 0.0001


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

@dataclass
class Portfolio:
    """Collection of sovereign fixed-income instruments."""
    instruments: List[Instrument] = field(default_factory=list)
    country:     str              = ""
    as_of_date:  str              = ""

    @property
    def total_notional_egp_bn(self) -> float:
        return sum(i.notional_egp_bn for i in self.instruments)

    @property
    def portfolio_duration(self) -> float:
        """Notional-weighted average modified duration."""
        total = self.total_notional_egp_bn
        if total < 1e-12:
            return 0.0
        return sum(i.modified_duration * i.notional_egp_bn
                   for i in self.instruments) / total

    @property
    def portfolio_bpv_egp_m(self) -> float:
        """Total BPV of the portfolio in EGP millions."""
        return sum(i.bpv_egp_m for i in self.instruments)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class SyntheticPortfolioBuilder:
    """Build the synthetic sovereign bond portfolio from exposure totals.

    Parameters
    ----------
    tbills_egp_bn : float
        Total T-Bill exposure in EGP billions (from annual report Note 5).
    tbonds_egp_bn : float
        Total T-Bond exposure in EGP billions (from annual report Note 5).
    yield_df : pd.DataFrame
        Historical yield curve (for coupon estimation per T-Bond maturity).
    tbill_weights : dict {days: weight}
        Proportional split across T-Bill maturities.  Must sum to ~1.
        Default: CBE Outstanding Balance T-Bills Q2 FY2023/2024 (Dec 2023).
    tbond_weights : dict {years: weight}
        Proportional split across T-Bond maturities.  Must sum to ~1.
        Default: CBE T-Bond auctions 2023 accepted amounts (T7Y & T10Y = 0).
    country : str
    as_of_date : str

    Notes
    -----
    duration_years and bpv_egp_m are computed outputs, never inputs.
    T10Y is excluded from the portfolio (weight = 0 in 2023 CBE data).
    """

    # Default weights (configurable)
    DEFAULT_TBILL_WEIGHTS: Dict[int, float] = {
        91: 0.34, 182: 0.15, 273: 0.08, 364: 0.43
    }
    # Source: CBE Outstanding Balance T-Bills Q2 FY2023/2024 (December 2023)

    DEFAULT_TBOND_WEIGHTS: Dict[int, float] = {3: 0.9835, 5: 0.0165}
    # Source: EGP_T-Bonds_Coupon_Auction_Historical.xlsx
    # col. Accepted Amount, filtered Issue Date 2023, grouped by Tenor
    # T7Y = 0.000 EGP Mds, T10Y = 0.000 EGP Mds -> excluded

    _TBILL_COLS: Dict[int, str]  = {91: "T91j", 182: "T182j", 273: "T273j", 364: "T364j"}
    _TBOND_COLS: Dict[int, str]  = {3: "T3Y",  5: "T5Y"}

    def __init__(
        self,
        tbills_egp_bn: float,
        tbonds_egp_bn: float,
        yield_df: pd.DataFrame,
        tbill_weights: Optional[Dict[int, float]] = None,
        tbond_weights: Optional[Dict[int, float]] = None,
        country:       str = "EGY",
        as_of_date:    str = "",
    ) -> None:
        self.tbills_egp_bn = float(tbills_egp_bn)
        self.tbonds_egp_bn = float(tbonds_egp_bn)
        self.yield_df      = yield_df
        self.tbill_weights = tbill_weights or self.DEFAULT_TBILL_WEIGHTS
        self.tbond_weights = tbond_weights or self.DEFAULT_TBOND_WEIGHTS
        self.country       = country
        self.as_of_date    = as_of_date

    def build(self) -> Portfolio:
        instruments: List[Instrument] = []

        # T-Bills (discount instruments, coupon = 0)
        for days, w in self.tbill_weights.items():
            instruments.append(Instrument(
                label           = f"T-Bill {days}j",
                maturity_years  = days / 365.0,
                notional_egp_bn = self.tbills_egp_bn * float(w),
                coupon_rate     = 0.0,
                is_tbill        = True,
            ))

        # T-Bonds (coupon = median WAY for that maturity in the reference year)
        for years, w in self.tbond_weights.items():
            col    = self._TBOND_COLS.get(int(years))
            coupon = self._estimate_coupon(col)
            instruments.append(Instrument(
                label           = f"T-Bond {years}Y",
                maturity_years  = float(years),
                notional_egp_bn = self.tbonds_egp_bn * float(w),
                coupon_rate     = coupon,
                is_tbill        = False,
            ))

        port = Portfolio(
            instruments = instruments,
            country     = self.country,
            as_of_date  = self.as_of_date,
        )
        LOG.info(
            "Portfolio built: %d instruments | notional=%.2f Bn EGP | "
            "D_mod=%.4f yr | BPV=%.2f M EGP",
            len(instruments),
            port.total_notional_egp_bn,
            port.portfolio_duration,
            port.portfolio_bpv_egp_m,
        )
        return port

    def _estimate_coupon(self, col: Optional[str]) -> float:
        """Estimate coupon as 2023 median WAY for the given maturity column."""
        if not col or col not in self.yield_df.columns:
            return 0.0
        s = self.yield_df[col].dropna()
        if len(s) == 0:
            return 0.0
        try:
            s_yr = s[s.index.year == 2023]
            if len(s_yr) > 0:
                return float(s_yr.median())
        except Exception:
            pass
        return float(s.iloc[-1])


# ---------------------------------------------------------------------------
# Repricer
# ---------------------------------------------------------------------------

# Maturity label -> years (for nearest-neighbour matching)
_MAT_YEARS: Dict[str, float] = {
    "T91j": 91/365, "T182j": 182/365, "T273j": 273/365, "T364j": 364/365,
    "T3Y": 3.0, "T5Y": 5.0, "T10Y": 10.0,
}

# Full revaluation threshold: if |Dy_max| > 2 pp (= 200 bp) -> use full
_FULL_REVL_THRESHOLD_PP = 2.0


class PortfolioRepricer:
    """Reprice a portfolio under a yield shock vector Dy(tau).

    Methods
    -------
    "duration" : DP/P ~ -D_mod * Dy + 0.5 * C * Dy^2
    "full"     : full cash-flow discounting
    "auto"     : full if |Dy_max| > 200 bp, else duration

    Parameters
    ----------
    portfolio : Portfolio
    yield_curve_T : pd.Series
        Last observed yield curve indexed by maturity label, in %.
    delta_y : pd.Series
        Yield shock vector, indexed by maturity label, in percentage points.
        T10Y entry (if present) is handled by the caller -- instruments that
        have no T10Y counterpart will receive Dy = 0 for that maturity.

    References
    ----------
    Fabozzi (2007). Fixed Income Mathematics. McGraw-Hill.
    """

    def __init__(
        self,
        portfolio:      Portfolio,
        yield_curve_T:  pd.Series,
        delta_y:        pd.Series,
    ) -> None:
        self.portfolio     = portfolio
        self.yield_curve_T = yield_curve_T
        self.delta_y       = delta_y

    def reprice(self, method: str = "auto") -> Tuple[float, Dict[str, float]]:
        """Compute DP for the full portfolio.

        Returns
        -------
        total_dp : float
            Total P&L in EGP millions (negative = loss).
        breakdown : dict {instrument_label: dp_m_egp}
        """
        dy_max   = float(self.delta_y.abs().max()) if len(self.delta_y) else 0.0
        use_full = (dy_max > _FULL_REVL_THRESHOLD_PP) if method == "auto" else (method == "full")
        method_eff = "full" if use_full else "duration"

        LOG.debug(
            "PortfolioRepricer: Dy_max=%.2f pp -> method=%s", dy_max, method_eff
        )

        breakdown: Dict[str, float] = {}
        total_dp = 0.0

        for inst in self.portfolio.instruments:
            mat_col = _closest_col(inst.maturity_years)
            dy_pp   = float(self.delta_y.get(mat_col, 0.0))
            dy_dec  = dy_pp / 100.0          # pp -> decimal

            if method_eff == "duration":
                dp = _duration_dp(inst, dy_dec)
            else:
                y_base = float(self.yield_curve_T.get(mat_col, 0.0)) / 100.0
                dp     = _full_dp(inst, y_base, dy_dec)

            breakdown[inst.label] = dp
            total_dp += dp

        return total_dp, breakdown


# ---- private helpers -------------------------------------------------------

def _closest_col(maturity_years: float) -> str:
    return min(_MAT_YEARS, key=lambda c: abs(_MAT_YEARS[c] - maturity_years))


def _duration_dp(inst: Instrument, dy_dec: float) -> float:
    """DP (EGP millions) using duration + convexity approximation."""
    d_mod     = inst.modified_duration
    convexity = d_mod ** 2      # rough approximation
    dp_frac   = -d_mod * dy_dec + 0.5 * convexity * dy_dec ** 2
    return dp_frac * inst.notional_egp_bn * 1_000.0   # Bn EGP -> M EGP


def _full_dp(inst: Instrument, y_base: float, dy_dec: float) -> float:
    """DP (EGP millions) from full cash-flow revaluation."""
    y_stress  = y_base + dy_dec
    notional_m = inst.notional_egp_bn * 1_000.0   # M EGP
    n          = inst.maturity_years
    c          = inst.coupon_rate / 100.0

    if inst.is_tbill or c < 1e-8:
        # Discount instrument: PV = N / (1 + y * n)
        pv_base   = notional_m / max(1.0 + y_base   * n, 1e-6)
        pv_stress = notional_m / max(1.0 + y_stress * n, 1e-6)
        return pv_stress - pv_base

    # Annual coupon bond
    years  = np.arange(1, int(round(n)) + 1, dtype=float)
    cfs    = np.full_like(years, c * notional_m)
    cfs[-1] += notional_m
    df_b   = (1.0 + max(y_base,   1e-6)) ** years
    df_s   = (1.0 + max(y_stress, 1e-6)) ** years
    return float(np.sum(cfs / df_s) - np.sum(cfs / df_b))


# ---------------------------------------------------------------------------
# Vectorized batch repricing — numerically identical to calling
# PortfolioRepricer.reprice() once per row of delta_y_batch, but replaces the
# O(n_simulations) Python-level loop (one PortfolioRepricer + reprice() call
# per Monte Carlo draw) with numpy broadcasting across the simulation axis,
# looping only over the (small, fixed) instrument list. Introduced because
# FHSSampler.run() calls reprice() inside a 10,000-iteration bootstrap loop,
# and the climate orchestrator calls run() once per (scenario, horizon year)
# — 50-75x per run — making the un-vectorized path the dominant cost of a
# full climate run (~2h observed). Additive: PortfolioRepricer.reprice()
# itself is untouched and still used by every other (non-batch) call site.
# ---------------------------------------------------------------------------

def _duration_dp_batch(inst: Instrument, dy_dec: np.ndarray) -> np.ndarray:
    """Vectorized _duration_dp(): same formula, dy_dec is now an array."""
    d_mod     = inst.modified_duration
    convexity = d_mod ** 2
    dp_frac   = -d_mod * dy_dec + 0.5 * convexity * dy_dec ** 2
    return dp_frac * inst.notional_egp_bn * 1_000.0


def _full_dp_batch(inst: Instrument, y_base: float, dy_dec: np.ndarray) -> np.ndarray:
    """Vectorized _full_dp(): same formula, dy_dec is now an array."""
    y_stress   = y_base + dy_dec
    notional_m = inst.notional_egp_bn * 1_000.0
    n          = inst.maturity_years
    c          = inst.coupon_rate / 100.0

    if inst.is_tbill or c < 1e-8:
        pv_base   = notional_m / max(1.0 + y_base * n, 1e-6)          # scalar
        pv_stress = notional_m / np.maximum(1.0 + y_stress * n, 1e-6)  # (n_sim,)
        return pv_stress - pv_base

    years = np.arange(1, int(round(n)) + 1, dtype=float)   # (n_years,)
    cfs   = np.full_like(years, c * notional_m)
    cfs[-1] += notional_m
    df_b  = (1.0 + max(y_base, 1e-6)) ** years                          # (n_years,)
    y_stress_safe = np.maximum(y_stress, 1e-6)                          # (n_sim,)
    df_s  = (1.0 + y_stress_safe[:, None]) ** years[None, :]            # (n_sim, n_years)
    pv_stress = np.sum(cfs[None, :] / df_s, axis=1)                     # (n_sim,)
    pv_base   = float(np.sum(cfs / df_b))
    return pv_stress - pv_base


def reprice_batch(
    portfolio:      "Portfolio",
    yield_curve_T:  pd.Series,
    delta_y_batch:  pd.DataFrame,
    method:         str = "auto",
) -> np.ndarray:
    """Batch equivalent of PortfolioRepricer(...).reprice(method) — one call
    per Monte Carlo draw becomes one vectorized pass over instruments.

    Parameters
    ----------
    delta_y_batch : pd.DataFrame, shape (n_simulations, n_maturities)
        Row i = the delta_y Series that would be passed to PortfolioRepricer
        for simulation i (same maturity-label columns as the scalar path;
        a T10Y column, if present, is ignored exactly as reprice() drops it
        via the caller).

    Returns
    -------
    np.ndarray, shape (n_simulations,) — total_dp per simulation, identical
    (up to floating-point rounding) to looping reprice() n_simulations times.
    """
    n_sim = len(delta_y_batch)
    dy_max_per_sim = delta_y_batch.abs().max(axis=1).to_numpy()
    use_full_per_sim = (
        dy_max_per_sim > _FULL_REVL_THRESHOLD_PP if method == "auto"
        else np.full(n_sim, method == "full")
    )

    total_dp = np.zeros(n_sim)
    for inst in portfolio.instruments:
        mat_col = _closest_col(inst.maturity_years)
        dy_pp  = (delta_y_batch[mat_col].to_numpy()
                  if mat_col in delta_y_batch.columns else np.zeros(n_sim))
        dy_dec = dy_pp / 100.0
        y_base = float(yield_curve_T.get(mat_col, 0.0)) / 100.0

        dp_duration = _duration_dp_batch(inst, dy_dec)
        dp_full     = _full_dp_batch(inst, y_base, dy_dec)
        total_dp += np.where(use_full_per_sim, dp_full, dp_duration)

    return total_dp