"""
fhs_sampler.py
==============
Filtered Historical Simulation (FHS) for sVaR and sES computation.

Pipeline
--------
1.  Compute Dbeta_k,t = beta_k,t - beta_k,t-1  (k = 1, 2, 3).
2.  Fit GARCH(1,1) on each Dbeta_k series via the arch library.
    Fallback to unconditional historical volatility if GARCH fails.
3.  Extract standardised residuals z_k,t = eps_k,t / sigma_k,t.
4.  Identify the BCBS stress window: 12 consecutive months maximising
    sum_k Var(Dbeta_k) over a rolling window (from 2005 onwards).
5.  Bootstrap N = 10 000 vecteurs (z1,t, z2,t, z3,t) with replacement
    from the stress window (draw whole rows to preserve co-dependence).
6.  Reconstruct stressed betas: beta*_k,s = beta_k,T + sigma_k,T * z_k,s.
7.  NS reconstruction + PortfolioRepricer for each scenario.
8.  Derive sVaR(99%) and sES(97.5%) from the {DP_1,...,DP_N} distribution.

References
----------
Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). VaR without
    correlations for portfolios of derivative securities.
    Journal of Futures Markets, 19(5), 583-602.
BCBS (2009). Revisions to the Basel II market risk framework (Basel II.5).
    Paragraph 718(Lxxvi)(b) -- stress period since 2005.
Artzner, P., Delbaen, F., Eber, J-M. & Heath, D. (1999). Coherent measures
    of risk. Mathematical Finance, 9(3), 203-228.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .nelson_siegel import NelsonSiegelReconstructor
from .portfolio import Portfolio, reprice_batch

LOG = logging.getLogger("market.fhs_sampler")

_STRESS_WINDOW_MONTHS = 12
_STRESS_HISTORY_FROM  = "2005-01-01"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MarketRiskMeasure:
    """Market risk measures produced by FHSSampler.

    All monetary figures in EGP millions.
    """
    svar_99:               float           # sVaR 99%  (positive = loss)
    ses_975:               float           # sES  97.5% (positive = loss)
    loss_distribution:     np.ndarray      # N simulated DP values (negative = loss)
    stress_window_start:   pd.Timestamp
    stress_window_end:     pd.Timestamp
    n_simulations:         int
    portfolio_duration:    float           # years
    portfolio_bpv:         float           # EGP millions
    beta1_stressed:        float           # beta1 at worst scenario
    beta2_stressed:        float           # beta2 at worst scenario
    beta3_stressed:        float           # beta3 at worst scenario
    validation_gate_verdict: str = "VALIDATED"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "svar_99":             round(self.svar_99, 4),
            "ses_975":             round(self.ses_975, 4),
            "n_simulations":       self.n_simulations,
            "stress_window_start": str(self.stress_window_start.date()),
            "stress_window_end":   str(self.stress_window_end.date()),
            "portfolio_duration":  round(self.portfolio_duration, 4),
            "portfolio_bpv":       round(self.portfolio_bpv, 4),
            "beta1_stressed":      round(self.beta1_stressed, 4),
            "beta2_stressed":      round(self.beta2_stressed, 4),
            "beta3_stressed":      round(self.beta3_stressed, 4),
            "validation_gate_verdict": self.validation_gate_verdict,
            "warnings":            self.warnings,
        }


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class FHSSampler:
    """Filtered Historical Simulation for market risk measurement.

    Parameters
    ----------
    betas_df : pd.DataFrame
        Monthly NS factors {beta1, beta2, beta3}.
    reconstructor : NelsonSiegelReconstructor
        Configured with the baseline curve at T.
    portfolio : Portfolio
        Synthetic sovereign bond portfolio.
    yield_curve_T : pd.Series
        Last observed yield curve (indexed by maturity label, in %).
    n_simulations : int
        Bootstrap sample size (default 10 000).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        betas_df:       pd.DataFrame,
        reconstructor:  NelsonSiegelReconstructor,
        portfolio:      Portfolio,
        yield_curve_T:  pd.Series,
        n_simulations:  int = 100_000,
        random_state:   int = 42,
    ) -> None:
        self.betas_df      = betas_df.dropna()
        self.reconstructor = reconstructor
        self.portfolio     = portfolio
        self.yield_curve_T = yield_curve_T
        self.n_simulations = int(n_simulations)
        self.rng           = np.random.default_rng(int(random_state))
        self._warns: List[str] = []

        # Populated once by _prepare() — GARCH fit, BCBS stress window and
        # bootstrap draws are scenario-independent (historical-data-only), so
        # they must not be redone on every run() call across scenarios.
        self._prepared      = False
        self._sigma_T_dict:  Dict[str, float] = {}
        self._z_boot:        Optional[np.ndarray] = None
        self._sw_start:      Optional[pd.Timestamp] = None
        self._sw_end:        Optional[pd.Timestamp] = None

    def _prepare(self) -> None:
        """
        One-time (memoized) GARCH fit + BCBS stress-window search + bootstrap
        draw. Scenario-independent: depends only on historical betas_df, so
        run() calls with different stressed_betas all reuse this exact same
        bootstrap sample — only the simulation's centre point changes,
        keeping baseline-vs-stressed comparisons apples-to-apples (same
        random draws, only the deterministic mean shifts).
        """
        if self._prepared:
            return

        # Step 1: first differences
        delta_b = self.betas_df.diff().dropna()

        # Step 2 & 3: GARCH / hist-vol -> standardised residuals
        z_cols:       Dict[str, pd.Series] = {}
        sigma_T_dict: Dict[str, float]     = {}
        for col in ("beta1", "beta2", "beta3"):
            z_s, sig_T = self._fit_garch(delta_b[col])
            z_cols[col]      = z_s
            sigma_T_dict[col] = sig_T

        z_df = pd.DataFrame(z_cols).dropna()

        # Step 4: BCBS stress window
        sw_start, sw_end = self._find_stress_window(delta_b)
        LOG.info("BCBS stress window: %s -> %s", sw_start.date(), sw_end.date())

        z_stress = z_df.loc[
            (z_df.index >= sw_start) & (z_df.index <= sw_end)
        ]
        if len(z_stress) < 6:
            self._warns.append(
                f"Stress window too short ({len(z_stress)} obs) -- using full history"
            )
            LOG.warning("Stress window too short -- using full residual history")
            z_stress = z_df

        # Step 5: bootstrap N row-vectors from stress window with kernel smoothing.
        # Drawing from only 12 stress-window observations produces at most 12
        # unique ΔP values → p95 = p99 (discrete distribution artefact).
        # Fix: add Silverman-bandwidth Gaussian jitter to each bootstrapped
        # z-vector so the distribution is continuous while remaining centred on
        # the historical stress residuals (Barone-Adesi et al. 1999).
        n_st   = len(z_stress)
        idx_b  = self.rng.integers(0, n_st, size=self.n_simulations)
        z_boot = z_stress.values[idx_b].copy()    # (N, 3)
        z_arr  = z_stress.values
        for j in range(z_arr.shape[1]):
            sigma_j = float(np.std(z_arr[:, j], ddof=1)) if n_st > 1 else 1.0
            h_j     = 1.06 * sigma_j * (n_st ** -0.2)   # Silverman's rule
            z_boot[:, j] += self.rng.normal(0.0, h_j, size=self.n_simulations)

        self._sigma_T_dict = sigma_T_dict
        self._z_boot       = z_boot
        self._sw_start     = sw_start
        self._sw_end       = sw_end
        self._prepared      = True

    def run(self, stressed_betas: Optional[Dict[str, float]] = None) -> MarketRiskMeasure:
        """
        Parameters
        ----------
        stressed_betas : optional {"beta1", "beta2", "beta3"}
            When provided, the bootstrap simulation is re-centred on these
            climate-stressed NS factors instead of the last historical betas
            — producing a genuinely scenario-conditional sVaR/sES (historical
            tail-risk volatility applied around the stressed curve, not the
            unconditional historical one). GARCH fitting and the BCBS stress
            window (both scenario-independent, historical-data-only) are
            fit once via _prepare() and reused across every call — only the
            reprice loop below re-runs per scenario.
        """
        self._prepare()
        sigma_T_dict = self._sigma_T_dict
        z_boot       = self._z_boot
        sw_start     = self._sw_start
        sw_end       = self._sw_end

        last_betas = self.betas_df.iloc[-1]
        # Centre of the simulation: last historical betas (unconditional risk
        # measure) unless a climate-stressed centre is supplied, in which
        # case the historical bootstrap tail risk is applied AROUND the
        # scenario-stressed curve instead — this is what makes sVaR/sES
        # scenario-conditional.
        center = stressed_betas if stressed_betas is not None else {
            "beta1": float(last_betas["beta1"]),
            "beta2": float(last_betas["beta2"]),
            "beta3": float(last_betas["beta3"]),
        }

        # Steps 6-7: simulate and reprice — vectorized across all
        # n_simulations at once (numerically identical to constructing one
        # PortfolioRepricer + calling reprice() per simulation, see
        # portfolio.reprice_batch() docstring for why this replaced the
        # former per-simulation Python loop: with n_simulations=10_000 and
        # this run() called once per (scenario, horizon year) by the climate
        # orchestrator — 50-75x per run — the scalar loop was the dominant
        # cost of a full climate run, observed at ~2h).
        betas_batch = np.column_stack([
            float(center["beta1"]) + sigma_T_dict["beta1"] * z_boot[:, 0],
            float(center["beta2"]) + sigma_T_dict["beta2"] * z_boot[:, 1],
            float(center["beta3"]) + sigma_T_dict["beta3"] * z_boot[:, 2],
        ])
        delta_y_batch = self.reconstructor.delta_curve_batch(betas_batch)
        delta_y_batch = delta_y_batch.drop(columns="T10Y", errors="ignore")

        dp_dist = reprice_batch(
            portfolio     = self.portfolio,
            yield_curve_T = self.yield_curve_T,
            delta_y_batch = delta_y_batch,
            method        = "auto",
        )

        # Step 8: risk measures (losses = -DP)
        loss_dist = -dp_dist
        svar_99   = float(np.percentile(loss_dist, 99))
        var_975   = float(np.percentile(loss_dist, 97.5))
        tail      = loss_dist[loss_dist >= var_975]
        ses_975   = float(np.mean(tail)) if len(tail) > 0 else var_975

        # Stressed betas at worst DP
        worst_i = int(np.argmin(dp_dist))
        z_w     = z_boot[worst_i]
        b1_w    = float(center["beta1"]) + sigma_T_dict["beta1"] * z_w[0]
        b2_w    = float(center["beta2"]) + sigma_T_dict["beta2"] * z_w[1]
        b3_w    = float(center["beta3"]) + sigma_T_dict["beta3"] * z_w[2]

        verdict = self._validate(svar_99, ses_975, dp_dist)

        LOG.info(
            "FHS done: sVaR(99%%)=%.2f M EGP  sES(97.5%%)=%.2f M EGP  N=%d",
            svar_99, ses_975, self.n_simulations,
        )

        return MarketRiskMeasure(
            svar_99              = svar_99,
            ses_975              = ses_975,
            loss_distribution    = dp_dist,
            stress_window_start  = sw_start,
            stress_window_end    = sw_end,
            n_simulations        = self.n_simulations,
            portfolio_duration   = self.portfolio.portfolio_duration,
            portfolio_bpv        = self.portfolio.portfolio_bpv_egp_m,
            beta1_stressed       = b1_w,
            beta2_stressed       = b2_w,
            beta3_stressed       = b3_w,
            validation_gate_verdict = verdict,
            warnings             = list(self._warns),
        )

    # ---- GARCH(1,1) --------------------------------------------------------

    def _fit_garch(self, series: pd.Series) -> Tuple[pd.Series, float]:
        """Fit GARCH(1,1). Returns (standardised_residuals, sigma_T).

        Falls back to unconditional historical volatility + WARNING if the
        arch library is absent or GARCH does not converge.
        """
        try:
            from arch import arch_model
            s_scaled = series.dropna() * 100.0   # scale for numerical stability
            am = arch_model(s_scaled, vol="Garch", p=1, q=1,
                            dist="normal", rescale=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(disp="off", options={"maxiter": 300, "ftol": 1e-8})
            # arch: convergence_flag == 0 means SUCCESS
            if res.convergence_flag == 0:
                cond_vol = res.conditional_volatility / 100.0
                common   = series.dropna().index.intersection(cond_vol.index)
                z_series = (series.dropna().reindex(common)
                            / cond_vol.reindex(common)).dropna()
                sigma_T  = float(cond_vol.iloc[-1])
                LOG.debug(
                    "GARCH converged for %s: sigma_T=%.6f", series.name, sigma_T
                )
                return z_series, sigma_T
            else:
                msg = (f"GARCH(1,1) non convergé pour {series.name} "
                       "(flag=%d) -- fallback volatilité historique" % res.convergence_flag)
                LOG.warning(msg)
                self._warns.append(msg)
        except ImportError:
            msg = ("arch library absent -- fallback volatilité historique. "
                   "Installer via: pip install arch>=5.0")
            if msg not in self._warns:
                LOG.warning(msg)
                self._warns.append(msg)
        except Exception as exc:
            msg = (f"GARCH échoué pour {series.name}: {exc} "
                   "-- fallback volatilité historique")
            LOG.warning(msg)
            self._warns.append(msg)

        # Fallback: historical vol
        hist_vol = float(series.dropna().std(ddof=1))
        if hist_vol < 1e-12:
            hist_vol = 1e-6
        z_series = (series.dropna() / hist_vol).dropna()
        return z_series, hist_vol

    # ---- Stress window -----------------------------------------------------

    def _find_stress_window(
        self, delta_betas: pd.DataFrame
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """12-month window maximising sum_k Var(Dbeta_k) -- BCBS II.5 (2009)."""
        db = delta_betas.loc[
            delta_betas.index >= _STRESS_HISTORY_FROM
        ].dropna()

        if len(db) < _STRESS_WINDOW_MONTHS:
            LOG.warning(
                "Insufficient data for stress window search "
                "(%d < %d months) -- using full history",
                len(db), _STRESS_WINDOW_MONTHS,
            )
            return db.index[0], db.index[-1]

        best_var    = -1.0
        best_start  = 0
        W           = _STRESS_WINDOW_MONTHS

        for i in range(len(db) - W + 1):
            window   = db.iloc[i: i + W]
            tot_var  = float(window.var(ddof=1).sum())
            if tot_var > best_var:
                best_var   = tot_var
                best_start = i

        return db.index[best_start], db.index[best_start + W - 1]

    # ---- Validation gate ---------------------------------------------------

    def _validate(
        self,
        svar_99: float,
        ses_975: float,
        dp_dist: np.ndarray,
    ) -> str:
        issues: List[str] = []
        if svar_99 < 0:
            issues.append(
                f"sVaR={svar_99:.2f} < 0 -- les gains dominent: "
                "vérifier les signes des shocks ou la composition du portefeuille"
            )
        if ses_975 < svar_99 * 0.85:
            issues.append(
                f"sES={ses_975:.2f} << sVaR={svar_99:.2f} "
                "(ratio < 0.85) -- vérifier la distribution"
            )
        n_zero = int(np.sum(np.abs(dp_dist) < 1e-6))
        if n_zero > self.n_simulations * 0.5:
            issues.append(
                f"{n_zero}/{self.n_simulations} simulations ont DP~0 "
                "-- vérifier les poids du portefeuille ou les shocks NS"
            )

        for w in issues:
            LOG.warning("ValidationGate[market]: %s", w)
            self._warns.append(w)

        if any("sVaR" in w or "sES" in w for w in issues):
            return "REJECTED"
        return "VALIDATED_WITH_WARNINGS" if issues else "VALIDATED"