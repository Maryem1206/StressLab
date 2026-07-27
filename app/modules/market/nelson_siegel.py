"""
nelson_siegel.py
================
Nelson-Siegel yield curve fitting and reconstruction.

For each observation date, beta1/beta2/beta3 are estimated by OLS
on the available maturities (NaN rows are skipped automatically).
Lambda is fixed a priori following Diebold & Li (2006).

References
----------
Nelson, C.R. & Siegel, A.F. (1987). Parsimonious modeling of yield curves.
    Journal of Business, 60(4), 473-489.
Diebold, F.X. & Li, C. (2006). Forecasting the term structure of government
    bond yields. Journal of Econometrics, 130(2), 337-364.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("market.nelson_siegel")

# Fixed decay parameter -- Diebold & Li (2006) calibrated to US Treasury data
LAMBDA_DEFAULT: float = 0.0609

# Maturities in years for each standard column label
MATURITY_YEARS: Dict[str, float] = {
    "T91j":  91  / 365,
    "T182j": 182 / 365,
    "T273j": 273 / 365,
    "T364j": 364 / 365,
    "T3Y":   3.0,
    "T5Y":   5.0,
    "T10Y":  10.0,
}


def _ns_design_matrix(tau: np.ndarray, lam: float) -> np.ndarray:
    """Compute the 3-column Nelson-Siegel design matrix.

    Columns: [1,  f(lambda,tau),  g(lambda,tau)]
    where
        f(l,t) = (1 - exp(-l*t)) / (l*t)
        g(l,t) = f(l,t) - exp(-l*t)
    """
    lt = lam * tau
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(lt > 1e-12, (1.0 - np.exp(-lt)) / lt, 1.0)
    g = f - np.exp(-lt)
    return np.column_stack([np.ones_like(tau), f, g])


class NelsonSiegelFitter:
    """Fit Nelson-Siegel parameters per calendar date via Ridge regression.

    Parameters
    ----------
    lambda_ : float
        Decay parameter lambda.  Use search_lambda() to calibrate automatically.
    min_maturities : int
        Minimum number of non-NaN maturities required to fit a date.
        Dates below this threshold receive NaN betas.
    ridge_alpha : float
        L2 regularisation parameter added to X'X diagonal before inversion.
        Prevents beta explosion when the design matrix is near-singular (which
        happens with any lambda where tau* = 1/lambda falls outside the data's
        maturity range).  Default 0.5 is appropriate for yields expressed in %.

    Usage
    -----
    lambda_, _ = NelsonSiegelFitter.search_lambda(yield_df)
    fitter    = NelsonSiegelFitter(lambda_=lambda_)
    betas_df  = fitter.fit(yield_df)

    Interpretation
    --------------
    beta1 = long-run level   (lim tau->inf y(tau))
    beta2 = slope (negative of spread short - long)
    beta3 = curvature (medium-term hump/trough)
    """

    def __init__(
        self,
        lambda_: float = LAMBDA_DEFAULT,
        min_maturities: int = 3,
        ridge_alpha: float = 0.5,
    ) -> None:
        self.lambda_        = float(lambda_)
        self.min_maturities = int(min_maturities)
        self.ridge_alpha    = float(ridge_alpha)

    # ---- public ----------------------------------------------------------

    def fit(self, yield_df: pd.DataFrame) -> pd.DataFrame:
        """Fit Nelson-Siegel for each row (date) of yield_df.

        Parameters
        ----------
        yield_df : pd.DataFrame
            Monthly DataFrame with columns that are a subset of MATURITY_YEARS.
            Values in % (e.g. 18.50 for 18.5 %). NaN = no auction that month.

        Returns
        -------
        pd.DataFrame
            Same index as yield_df, columns {beta1, beta2, beta3}.
            Dates with insufficient data have NaN in all three columns.
        """
        rows = []
        for date, row in yield_df.iterrows():
            b1, b2, b3 = self._fit_row(row)
            rows.append({"date": date, "beta1": b1, "beta2": b2, "beta3": b3})

        betas_df = pd.DataFrame(rows).set_index("date")
        n_valid = int(betas_df.notna().all(axis=1).sum())
        LOG.info(
            "Nelson-Siegel: %d/%d dates fitted (lambda=%.4f, min_mat=%d)",
            n_valid, len(betas_df), self.lambda_, self.min_maturities,
        )
        if n_valid == 0:
            raise RuntimeError(
                "NelsonSiegelFitter: no date had enough maturities to fit. "
                f"Required: {self.min_maturities}, lambda={self.lambda_}"
            )

        # ── Post-fit sanity check on all 3 betas ─────────────────────────
        # β₁ = long-run yield level; for any sovereign curve β₁ should stay
        # within a physically plausible range.  Values >> 100 indicate that
        # the design matrix X is ill-conditioned (λ too small → τ* = 1/λ
        # far outside the data's maturity range → columns of X nearly
        # collinear → OLS produces huge compensating coefficients).
        valid_betas = betas_df.dropna()
        for col, lo, hi, label in [
            ("beta1", -50.0,  200.0, "niveau long terme"),
            ("beta2", -500.0, 300.0, "pente"),
            ("beta3", -700.0, 400.0, "courbure"),
        ]:
            if col not in valid_betas.columns:
                continue
            col_max = float(valid_betas[col].abs().max())
            n_extreme = int((valid_betas[col].abs() > hi).sum())
            if n_extreme > 0:
                LOG.warning(
                    "NS post-fit: %s (%s) has %d/%d dates with |value| > %.0f "
                    "(max=%.1f). Root cause: lambda=%.4f → τ*=%.1f ans hors plage "
                    "données (max %.1f ans). Utiliser search_lambda() pour recalibrer.",
                    col, label, n_extreme, len(valid_betas), hi, col_max,
                    self.lambda_, 1.0 / self.lambda_,
                    max(MATURITY_YEARS.values()),
                )
        # ─────────────────────────────────────────────────────────────────

        return betas_df

    def reconstruct(
        self,
        beta1: float,
        beta2: float,
        beta3: float,
        maturities: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """Reconstruct a yield curve from NS parameters.

        Parameters
        ----------
        beta1, beta2, beta3 : float
        maturities : dict, optional
            {label: years} -- defaults to MATURITY_YEARS.

        Returns
        -------
        pd.Series  indexed by maturity label, values in %.
        """
        mats = maturities or MATURITY_YEARS
        taus  = np.array(list(mats.values()))
        X     = _ns_design_matrix(taus, self.lambda_)
        yields = X @ np.array([beta1, beta2, beta3])
        return pd.Series(yields, index=list(mats.keys()))

    def reconstruct_batch(
        self,
        betas: np.ndarray,
        maturities: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Vectorized reconstruct(): many (beta1, beta2, beta3) rows at once.

        Numerically identical to calling reconstruct() once per row (same
        fixed design matrix X, reused across rows) — introduced so Monte
        Carlo callers (FHSSampler) avoid one Python call + Series allocation
        per simulation.

        Parameters
        ----------
        betas : (n, 3) array — columns [beta1, beta2, beta3].

        Returns
        -------
        pd.DataFrame, shape (n, n_maturities), columns = maturity labels.
        """
        mats = maturities or MATURITY_YEARS
        taus = np.array(list(mats.values()))
        X    = _ns_design_matrix(taus, self.lambda_)   # (n_mat, 3)
        yields = betas @ X.T                             # (n, n_mat)
        return pd.DataFrame(yields, columns=list(mats.keys()))

    # ---- public (class-level) -------------------------------------------

    @staticmethod
    def search_lambda(
        yield_df: pd.DataFrame,
        lambda_grid: "np.ndarray | None" = None,
        min_maturities: int = 3,
        ridge_alpha: float = 0.5,
    ) -> Tuple[float, float]:
        """Grid-search for the optimal lambda minimising average Ridge MSE.

        IMPORTANT: uses Ridge regression (not OLS) for each candidate lambda.
        With pure OLS, MSE decreases monotonically as lambda→0 because a
        near-singular X matrix allows arbitrarily large compensating betas
        that annihilate each other in y_hat.  Ridge penalises ||beta||² so
        the MSE objective is non-monotone and the grid search finds the lambda
        where the NS factors genuinely explain the yield curve shape.

        The grid starts at 0.10 (tau* = 10 Y, the edge of Egyptian data) to
        exclude lambda values whose curvature peak is outside the observed
        maturity range.

        Parameters
        ----------
        yield_df : pd.DataFrame
            Same format as NelsonSiegelFitter.fit() expects.
        lambda_grid : array-like, optional
            Candidate lambda values.
            Default: 200 values in [0.10, 3.0] (tau* in [0.33 Y, 10 Y]).
        min_maturities : int
            Minimum non-NaN maturities to include a date.
        ridge_alpha : float
            Same regularisation parameter used in _fit_row.

        Returns
        -------
        (best_lambda, best_avg_ridge_mse) : Tuple[float, float]
        """
        tau_max = max(MATURITY_YEARS.values())   # 10 Y for Egypt
        lam_min = 1.0 / tau_max                  # 0.10 -- tau* at data boundary
        if lambda_grid is None:
            lambda_grid = np.linspace(lam_min, 3.0, 200)

        best_lam: float = float(lambda_grid[0])
        best_mse: float = np.inf

        # Extract each date's (taus, y) ONCE, outside the lambda loop —
        # yield_df.iterrows() builds a fresh pandas Series per call, and this
        # used to run inside the lambda loop (200 grid points x ~192 monthly
        # dates = ~38 400 iterrows() calls), which dominated search_lambda()'s
        # runtime (measured ~73s of an ~89s market pre-fit) even though the
        # actual per-date math (a 3x3 Ridge solve) is essentially free. Same
        # dropna/min_maturities filtering as before, computed once.
        row_data: List[Tuple[np.ndarray, np.ndarray]] = []
        for _, row in yield_df.iterrows():
            valid = row.dropna()
            valid = valid[[c for c in valid.index if c in MATURITY_YEARS]]
            if len(valid) < min_maturities:
                continue
            taus = np.array([MATURITY_YEARS[c] for c in valid.index])
            y = valid.values.astype(float)
            row_data.append((taus, y))

        ridge_eye = ridge_alpha * np.eye(3)
        for lam in lambda_grid:
            lam = float(lam)
            mse_total, n_dates = 0.0, 0
            for taus, y in row_data:
                X = _ns_design_matrix(taus, lam)
                try:
                    # Ridge: beta = (X'X + alpha*I)^{-1} X'y
                    XtX = X.T @ X + ridge_eye
                    beta = np.linalg.solve(XtX, X.T @ y)
                    y_hat = X @ beta
                    mse_total += float(np.mean((y - y_hat) ** 2))
                    n_dates += 1
                except Exception:
                    pass
            if n_dates > 0:
                avg = mse_total / n_dates
                if avg < best_mse:
                    best_mse = avg
                    best_lam = lam

        LOG.info(
            "NS lambda grid-search (Ridge alpha=%.2f): "
            "best_lambda=%.4f  tau*=%.2fY  avg_MSE=%.6f  (grid=%d pts)",
            ridge_alpha, best_lam, 1.0 / best_lam, best_mse, len(lambda_grid),
        )
        return best_lam, best_mse

    # ---- private ---------------------------------------------------------

    def _fit_row(self, row: pd.Series) -> Tuple[float, float, float]:
        """Ridge regression fit for one date. Returns (beta1, beta2, beta3) or (nan, nan, nan).

        Uses Ridge instead of OLS to prevent beta explosion when X is near-singular.
        With OLS, small lambda → near-collinear X → huge compensating betas that
        cancel in y_hat (so MSE ≈ 0 but betas are meaningless).  Ridge penalises
        ||beta||² and keeps coefficients in a physically plausible range.
        """
        nan3 = (float("nan"), float("nan"), float("nan"))

        valid = row.dropna()
        valid = valid[[c for c in valid.index if c in MATURITY_YEARS]]
        if len(valid) < self.min_maturities:
            return nan3

        taus = np.array([MATURITY_YEARS[c] for c in valid.index])
        y    = valid.values.astype(float)
        X    = _ns_design_matrix(taus, self.lambda_)

        try:
            # Ridge: beta = (X'X + alpha*I)^{-1} X'y
            XtX  = X.T @ X + self.ridge_alpha * np.eye(3)
            Xty  = X.T @ y
            beta = np.linalg.solve(XtX, Xty)
            return float(beta[0]), float(beta[1]), float(beta[2])
        except np.linalg.LinAlgError:
            return nan3


class NelsonSiegelReconstructor:
    """Reconstruct stressed yield curves and compute Dy(tau) = y*(tau) - y_T(tau).

    This object is the unique interface between the macro stress pipeline
    (which delivers stressed betas) and the portfolio repricing engine.

    Parameters
    ----------
    fitter : NelsonSiegelFitter
        Carries lambda and reconstruct() method.
    baseline_curve : pd.Series
        Observed (or NS-fitted) yield curve at the last date T, in %.
        Indexed by maturity label (e.g. "T91j", "T3Y" ...).
    maturities : dict, optional
        {label: years} -- defaults to MATURITY_YEARS.
    """

    def __init__(
        self,
        fitter: NelsonSiegelFitter,
        baseline_curve: pd.Series,
        maturities: Optional[Dict[str, float]] = None,
    ) -> None:
        self.fitter         = fitter
        self.baseline_curve = baseline_curve
        self.maturities     = maturities or MATURITY_YEARS

    def stressed_curve(self, beta1: float, beta2: float, beta3: float) -> pd.Series:
        """Return y*(tau) for given stressed betas (in %)."""
        return self.fitter.reconstruct(beta1, beta2, beta3, self.maturities)

    def delta_curve(self, beta1: float, beta2: float, beta3: float) -> pd.Series:
        """Return Dy(tau) = y*(tau) - y_T(tau) in percentage points.

        The T10Y entry is included here; the PortfolioRepricer will drop it
        because no 10Y instrument is in the synthetic CIB portfolio.
        """
        y_stress = self.stressed_curve(beta1, beta2, beta3)
        y_base   = self.baseline_curve.reindex(y_stress.index)
        return y_stress - y_base

    def delta_curve_batch(self, betas: np.ndarray) -> pd.DataFrame:
        """Vectorized delta_curve(): many (beta1, beta2, beta3) rows at once.

        betas : (n, 3) array — columns [beta1, beta2, beta3].
        Returns pd.DataFrame, shape (n, n_maturities).
        """
        y_stress_batch = self.fitter.reconstruct_batch(betas, self.maturities)
        y_base = self.baseline_curve.reindex(y_stress_batch.columns)
        return y_stress_batch - y_base