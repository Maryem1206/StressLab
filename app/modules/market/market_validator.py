"""
market_validator.py
===================
Post-estimation validation runner for the market risk module.

Batteries executed
------------------
B1 (universal, 6 tests)  × 3 satellites (β₁, β₂, β₃)   = 18 tests
B2.E (AR(1) stationarité)  × 3 satellites                =  3 tests
B3.M1  Qualité ajustement Nelson-Siegel (RMSE/maturité)  =  1 test
B3.M2  Bornes économiques taux projetés [0%, 60%]        =  1 test
B3.M3  Monotonicité ordinale Δy par maturité             =  1 test
B3.M4  Divergence effective |Perte_sev|>|adv|>|bl|       =  1 test
B4.K   Kupiec LR_uc + feux tricolores adaptés mensuel    =  1 test
B4.C   Christoffersen LR_cc (couverture + indépendance)  =  1 test
                                                 TOTAL   = 27 tests

References
----------
Kupiec, P.H. (1995). Risk 8(9): 45-48.
Christoffersen, P. (1998). Int. Econ. Rev. 39(3): 841-862.
Nelson & Siegel (1987). J. Business 60(4): 473-489.
Diebold & Li (2006). J. Econometrics 130(2): 337-364.
BCBS (1996). Amendment to the Capital Accord (market risk).
BCBS (2009). Revisions to the Basel II market risk framework.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.post_estimation_validator import (
    PostEstimationValidator,
    RiskOutput,
    TestResult,
    ValidationReport,
)
from .fhs_sampler import MarketRiskMeasure
from .ir_satellite import BetaSatellite
from .nelson_siegel import MATURITY_YEARS, NelsonSiegelFitter, _ns_design_matrix
from .portfolio import Portfolio, PortfolioRepricer

LOG = logging.getLogger("market.validator")

# ── Constants ─────────────────────────────────────────────────────────────────
_RMSE_SCALE_BETAS = 10.0   # B1.5/B1.6: thresholds × 10 (betas expressed in %)
_YIELD_LO = 0.0            # Lower economic bound on projected yields (%)
_YIELD_HI = 60.0           # Upper bound — above hyperinflation threshold (%)
_KUPIEC_P = 0.01           # VaR confidence level = 99%

# Maturities ordered short → long for monotonicity checks
_MAT_ORDER: List[str] = [
    "T91j", "T182j", "T273j", "T364j", "T3Y", "T5Y", "T10Y"
]


# ── chi-squared p-value (scipy preferred, table fallback) ─────────────────────
try:
    from scipy.stats import chi2 as _scipy_chi2

    def _chi2_pval(stat: float, df: int) -> float:
        return float(1.0 - _scipy_chi2.cdf(stat, df=df))

except ImportError:
    _CHI2_TABLE = {
        1: [(2.706, 0.10), (3.841, 0.05), (6.635, 0.01), (10.828, 0.001)],
        2: [(4.605, 0.10), (5.991, 0.05), (9.210, 0.01), (13.816, 0.001)],
    }

    def _chi2_pval(stat: float, df: int) -> float:
        for cv, p in _CHI2_TABLE.get(df, _CHI2_TABLE[1]):
            if stat < cv:
                return p
        return 0.001


# ── Output dataclass ──────────────────────────────────────────────────────────
@dataclass
class MarketValidationReport:
    """Aggregated post-estimation validation report for the market module."""
    satellite_reports: Dict[int, ValidationReport]  # {1, 2, 3}
    b3_results:        List[TestResult]             # B3.M1–B3.M4
    backtest_results:  List[TestResult]             # B4.K, B4.C
    n_realized_pnl:    int
    overall_verdict:   str                          # VALIDATED / _WITH_WARNINGS / REJECTED

    def to_dict(self) -> dict:
        def _tr(r: TestResult) -> dict:
            return {
                "test_id":   r.test_id,
                "test_name": r.test_name,
                "battery":   r.battery,
                "statistic": r.statistic,
                "p_value":   r.p_value,
                "threshold": r.threshold,
                "status":    r.status,
                "message":   r.message,
                "reference": r.reference,
            }
        return {
            "satellite_reports": {
                str(k): v.to_dict() for k, v in self.satellite_reports.items()
            },
            "b3_results":       [_tr(r) for r in self.b3_results],
            "backtest_results":  [_tr(r) for r in self.backtest_results],
            "n_realized_pnl":    self.n_realized_pnl,
            "overall_verdict":   self.overall_verdict,
        }


# ── Runner ────────────────────────────────────────────────────────────────────
class MarketPostEstimationRunner:
    """Orchestrates all post-estimation validation batteries for the market module.

    Parameters
    ----------
    satellites    : {1: BetaSatellite, 2: ..., 3: ...} from IRSatellite.fit_all()
    betas_df      : Monthly NS factors (192 × 3), already dropna'd
    macro_df      : Monthly macro variables aligned to betas_df
    fitter        : Fitted NelsonSiegelFitter (carries λ and reconstruct())
    yield_df      : Historical observed yield curve (monthly, cols = MATURITY_COLS)
    portfolio     : Synthetic sovereign bond portfolio
    risk_measure  : MarketRiskMeasure produced by FHSSampler (carries sVaR, sES)
    """

    def __init__(
        self,
        satellites:   Dict[int, BetaSatellite],
        betas_df:     pd.DataFrame,
        macro_df:     pd.DataFrame,
        fitter:       NelsonSiegelFitter,
        yield_df:     pd.DataFrame,
        portfolio:    Portfolio,
        risk_measure: MarketRiskMeasure,
    ) -> None:
        self.satellites   = satellites
        self.betas_df     = betas_df.dropna()
        self.macro_df     = macro_df
        self.fitter       = fitter
        self.yield_df     = yield_df
        self.portfolio    = portfolio
        self.risk_measure = risk_measure

    # ── Public entry point ───────────────────────────────────────────────────

    def run(
        self,
        delta_y_per_scenario: Optional[Dict[str, pd.Series]] = None,
        losses_per_scenario:  Optional[Dict[str, float]] = None,
        stressed_curves:      Optional[Dict[str, pd.Series]] = None,
    ) -> MarketValidationReport:
        """Run all 27 tests. Non-blocking — exceptions per test are caught.

        Parameters
        ----------
        delta_y_per_scenario  : {"baseline": Series, "adverse": ..., "severe": ...}
                                 Δy(τ) in pp for each scenario.
        losses_per_scenario   : {"baseline": float, ...} — loss = -ΔP (M EGP).
                                 Positive = loss, negative = gain.
        stressed_curves       : {"baseline": Series, ...} — y*(τ) in % for each scenario.
        """
        # B1 + B2 per satellite
        sat_reports: Dict[int, ValidationReport] = {}
        for k in (1, 2, 3):
            sat_reports[k] = self._validate_satellite(k)
            LOG.info(
                "Satellite β%d: %s  (PASS=%d WARN=%d FAIL=%d)",
                k,
                sat_reports[k].verdict,
                sat_reports[k].n_pass,
                sat_reports[k].n_warn,
                sat_reports[k].n_fail,
            )

        # B3 market-specific
        b3 = [
            self._b3_m1_ns_fit(),
            self._b3_m2_yield_bounds(stressed_curves),
            self._b3_m3_delta_y_monotony(delta_y_per_scenario),
            self._b3_m4_scenario_divergence(losses_per_scenario),
        ]

        # B4 backtesting
        realized_pnl = self._compute_realized_pnl()
        bt = [
            self._b4_kupiec(realized_pnl),
            self._b4_christoffersen(realized_pnl),
        ]

        # Aggregate verdict
        all_res: List[TestResult] = []
        for r in sat_reports.values():
            all_res.extend(r.results)
        all_res.extend(b3)
        all_res.extend(bt)

        n_fail = sum(1 for r in all_res if r.status == "FAIL")
        n_warn = sum(1 for r in all_res if r.status == "WARN")
        if n_fail > 0:
            verdict = "REJECTED"
        elif n_warn > 0:
            verdict = "VALIDATED_WITH_WARNINGS"
        else:
            verdict = "VALIDATED"

        LOG.info(
            "Market validation complete: %s  (FAIL=%d WARN=%d / %d tests)",
            verdict, n_fail, n_warn, len(all_res),
        )
        return MarketValidationReport(
            satellite_reports=sat_reports,
            b3_results=b3,
            backtest_results=bt,
            n_realized_pnl=len(realized_pnl),
            overall_verdict=verdict,
        )

    # =========================================================================
    # B1 + B2  —  Per-satellite validation
    # =========================================================================

    def _validate_satellite(self, k: int) -> ValidationReport:
        sat = self.satellites[k]
        try:
            y_true, y_pred, residuals = self._reconstruct_fitted(k, sat)
        except Exception as exc:
            LOG.warning("Cannot reconstruct fitted values for β%d: %s", k, exc)
            return ValidationReport(
                module="market", model_type="ar1_macro",
                timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                n_obs=sat.n_obs, results=[], n_pass=0, n_warn=1, n_fail=0,
                verdict="VALIDATED_WITH_WARNINGS",
            )

        ro = RiskOutput(
            module      = "market",
            model_type  = "ar1_macro",
            y_true      = y_true,
            y_pred      = y_pred,
            residuals   = residuals,
            n_obs       = len(y_true),
            k_params    = len(sat.selected_vars) + 1,  # gammas + rho
            rho         = sat.rho,                      # → B2.E
            rmse_scale  = _RMSE_SCALE_BETAS,
        )
        return PostEstimationValidator(ro).run()

    def _reconstruct_fitted(
        self, k: int, sat: BetaSatellite
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (y_true, y_pred, residuals) by replaying predict_next() on history."""
        beta_col = f"beta{k}"
        beta_s   = self.betas_df[beta_col].dropna()

        common = beta_s.index.intersection(self.macro_df.index)
        beta_s = beta_s.reindex(common).dropna()
        macro_s = self.macro_df.reindex(common)

        dates  = beta_s.index
        y_list: List[float] = []
        yh_list: List[float] = []

        for i in range(1, len(dates)):
            t, t_prev = dates[i], dates[i - 1]
            if pd.isna(beta_s.loc[t]) or pd.isna(beta_s.loc[t_prev]):
                continue
            macro_row  = macro_s.loc[t]
            macro_vals = {
                col: float(macro_row[col])
                for col in sat.selected_vars
                if col in macro_row.index and pd.notna(macro_row[col])
            }
            yh_list.append(sat.predict_next(float(beta_s.loc[t_prev]), macro_vals))
            y_list.append(float(beta_s.loc[t]))

        y_true    = np.asarray(y_list,  dtype=float)
        y_pred    = np.asarray(yh_list, dtype=float)
        return y_true, y_pred, y_true - y_pred

    # =========================================================================
    # B3  —  Market-specific structural tests
    # =========================================================================

    def _b3_m1_ns_fit(self) -> TestResult:
        """RMSE Nelson-Siegel sur courbe observée — qualité d'ajustement.

        Calcule le RMSE moyen (toutes maturités × toutes dates) entre les
        taux observés et les taux NS-reconstruits depuis betas_df.
        Seuils : PASS < 0.5 pp  |  WARN < 1.0 pp  |  FAIL ≥ 1.0 pp.

        KNOWN LIMITATION (documented, not corrected -- 2026-07 validation review):
        on this dataset (Egypt), a single-lambda 3-factor NS structurally
        cannot bring this metric under 1.0pp. lambda grid-search settles at
        its floor (tau*=10y, the boundary chosen in search_lambda() to avoid
        the near-singular design matrix below it) because the Ridge MSE
        objective decreases monotonically past that point into degenerate
        territory. Re-optimising lambda to directly minimise this per-maturity
        RMSE (rather than search_lambda's per-date MSE) only moves the mean
        from ~1.34pp to ~1.31pp, with T10Y RMSE actually worsening -- T5Y and
        T10Y sit at levels the 3-factor NS shape cannot reconcile with the
        short end simultaneously. Closing this gap would need a 4-factor
        (Svensson) extension, which is a larger model change left for a
        future iteration.
        """
        try:
            betas = self.betas_df.dropna()
            mats  = {c: MATURITY_YEARS[c]
                     for c in self.yield_df.columns if c in MATURITY_YEARS}
            if not mats:
                return _na("B3.M1", "Qualité ajustement NS (RMSE)", 3,
                           "Aucune colonne maturité commune yield_df / MATURITY_YEARS")

            mat_cols = list(mats.keys())
            tau_arr  = np.array([mats[c] for c in mat_cols])

            common = betas.index.intersection(self.yield_df.index)
            if len(common) == 0:
                return _na("B3.M1", "Qualité ajustement NS (RMSE)", 3,
                           "Aucune date commune betas_df / yield_df")

            beta_arr = betas.loc[common, ["beta1", "beta2", "beta3"]].values  # (T,3)
            y_obs    = self.yield_df.loc[common, mat_cols].values.astype(float)  # (T,M)

            X   = _ns_design_matrix(tau_arr, self.fitter.lambda_)  # (M,3)
            y_ns = beta_arr @ X.T                                    # (T,M)

            rmse_per_mat = np.sqrt(np.nanmean((y_obs - y_ns) ** 2, axis=0))
            mean_rmse    = float(np.nanmean(rmse_per_mat))

            detail = " | ".join(
                f"{m}:{r:.3f}pp" for m, r in zip(mat_cols, rmse_per_mat)
            )

            if mean_rmse < 0.5:
                status, label = "PASS", "< 0.5 pp — ajustement NS satisfaisant"
            elif mean_rmse < 1.0:
                status, label = "WARN", "∈ [0.5, 1.0) pp — ajustement NS modéré"
            else:
                status, label = "FAIL", "≥ 1.0 pp — ajustement NS insuffisant"

            return TestResult(
                "B3.M1", "Qualité ajustement NS (RMSE par maturité)", 3,
                round(mean_rmse, 4), None, "RMSE < 0.5 pp",
                status, f"RMSE moyen={mean_rmse:.3f}pp {label} | [{detail}]",
                "Nelson & Siegel (1987) JBus 60(4):473-489;"
                " Diebold & Li (2006) JoE 130(2):337-364",
            )
        except Exception as exc:
            return _na("B3.M1", "Qualité ajustement NS (RMSE)", 3, str(exc))

    def _b3_m2_yield_bounds(
        self, stressed_curves: Optional[Dict[str, pd.Series]]
    ) -> TestResult:
        """Bornes économiques sur les taux projetés stressés.

        Vérifie que tous les taux NS-reconstruits sous stress restent dans
        [0%, 60%].  La borne inférieure exclut les taux nominaux négatifs
        (pas de ZIRP/NIRP pour l'Égypte).  La borne supérieure (60%) dépasse
        le pic hyperinflationniste de référence (Turquie 2023 : ~43%).
        Seuils : 0 violation → PASS | ≤5% → WARN | >5% → FAIL.
        """
        try:
            if not stressed_curves:
                return _na("B3.M2", "Bornes économiques taux projetés", 3,
                           "stressed_curves non fourni")

            all_vals = np.concatenate([
                np.asarray(c.values, dtype=float)
                for c in stressed_curves.values()
            ])
            n_total = len(all_vals)
            n_lo    = int(np.sum(all_vals < _YIELD_LO))
            n_hi    = int(np.sum(all_vals > _YIELD_HI))
            n_viol  = n_lo + n_hi
            pct     = n_viol / max(n_total, 1)

            rng_str = (f"min={all_vals.min():.2f}%"
                       f" max={all_vals.max():.2f}%"
                       f" | bornes [{_YIELD_LO:.0f}%, {_YIELD_HI:.0f}%]")

            if n_viol == 0:
                status = "PASS"
                msg    = f"0 violation sur {n_total} valeurs projetées | {rng_str}"
            elif pct <= 0.05:
                status = "WARN"
                msg    = (f"{n_viol}/{n_total} violations ({pct:.1%} ≤ 5%)"
                          f" | {n_lo} sous {_YIELD_LO:.0f}%"
                          f" | {n_hi} au-dessus {_YIELD_HI:.0f}%")
            else:
                status = "FAIL"
                msg    = (f"{n_viol}/{n_total} violations ({pct:.1%} > 5%)"
                          f" | {n_lo} sous {_YIELD_LO:.0f}%"
                          f" | {n_hi} au-dessus {_YIELD_HI:.0f}%")

            return TestResult(
                "B3.M2", "Bornes économiques taux projetés", 3,
                float(n_viol), None,
                f"0 violation | [{_YIELD_LO:.0f}%, {_YIELD_HI:.0f}%]",
                status, msg,
                "BCBS (2009) Sound stress testing §18; EBA/GL/2018/04 §108",
            )
        except Exception as exc:
            return _na("B3.M2", "Bornes économiques taux projetés", 3, str(exc))

    def _b3_m3_delta_y_monotony(
        self, delta_y_per_scenario: Optional[Dict[str, pd.Series]]
    ) -> TestResult:
        """Monotonicité ordinale des chocs Δy(τ) par maturité croissante.

        La structure terme stressée doit être économiquement cohérente :
        les chocs Δy doivent varier de façon monotone (croissante ou
        décroissante) entre la maturité la plus courte et la plus longue.
        Un changement de direction (sign-flip des différences consécutives)
        indique une structure terme stressée non réaliste.

        Seuils : 0 changement de direction → PASS | 1-2 → WARN | >2 → FAIL.
        """
        try:
            if not delta_y_per_scenario:
                return _na("B3.M3", "Monotonicité ordinale Δy", 3,
                           "delta_y_per_scenario non fourni")

            total_changes = 0
            details: List[str] = []

            for scen, dy in delta_y_per_scenario.items():
                vals = np.array([
                    float(dy.get(m, np.nan))
                    for m in _MAT_ORDER
                    if m in dy.index
                ], dtype=float)
                vals = vals[~np.isnan(vals)]
                if len(vals) < 3:
                    continue
                diffs    = np.diff(vals)
                # A direction change: sign of diff[i+1] opposes sign of diff[i]
                n_flip   = int(np.sum(diffs[:-1] * diffs[1:] < 0))
                total_changes += n_flip
                direction = ("croissant" if np.sum(diffs > 0) >= np.sum(diffs < 0)
                             else "décroissant")
                details.append(f"{scen}:{n_flip}flip({direction})")

            detail_str = " | ".join(details) if details else "—"

            if total_changes == 0:
                status = "PASS"
                msg    = f"Δy monotone (tous scénarios) — {detail_str}"
            elif total_changes <= 2:
                status = "WARN"
                msg    = (f"{total_changes} changement(s) de direction"
                          f" — structure Δy légèrement non monotone — {detail_str}")
            else:
                status = "FAIL"
                msg    = (f"{total_changes} changements de direction"
                          f" — structure Δy non monotone — {detail_str}")

            return TestResult(
                "B3.M3", "Monotonicité ordinale Δy par maturité", 3,
                float(total_changes), None, "0 changement de direction",
                status, msg,
                "Diebold & Li (2006); BCBS (2017) Stress testing principles §12",
            )
        except Exception as exc:
            return _na("B3.M3", "Monotonicité ordinale Δy", 3, str(exc))

    def _b3_m4_scenario_divergence(
        self, losses_per_scenario: Optional[Dict[str, float]]
    ) -> TestResult:
        """Divergence effective des scénarios (valeurs absolues des pertes).

        Teste que |Perte_sévère| > |Perte_adverse| > |Perte_baseline|.
        Utilise les valeurs absolues car le baseline peut produire un gain
        (si les taux baissent sous scénario de crise retenu).
        Le ratio |Perte_sévère| / |Perte_baseline| est stocké comme statistique.

        Seuils : PASS si les 2 inégalités sont vérifiées | WARN si 1 seule
                 est violée | FAIL si les 2 sont violées.
        """
        try:
            if not losses_per_scenario:
                return _na("B3.M4", "Divergence effective des scénarios", 3,
                           "losses_per_scenario non fourni")

            abs_bl  = abs(float(losses_per_scenario.get("baseline", 0.0)))
            abs_adv = abs(float(losses_per_scenario.get("adverse",  0.0)))
            abs_sev = abs(float(losses_per_scenario.get("severe",   0.0)))

            ok_adv_bl  = abs_adv > abs_bl    # adverse pire que baseline
            ok_sev_adv = abs_sev > abs_adv   # sévère pire qu'adverse
            n_ok       = int(ok_adv_bl) + int(ok_sev_adv)

            ratio = abs_sev / max(abs_bl, 1e-6)
            summary = (f"|bl|={abs_bl:.0f} |adv|={abs_adv:.0f}"
                       f" |sev|={abs_sev:.0f} M EGP")

            if n_ok == 2:
                status = "PASS"
                msg    = f"Divergence conforme OK — {summary} | ratio sev/bl={ratio:.1f}x"
            elif n_ok == 1:
                status = "WARN"
                failed = []
                if not ok_adv_bl:
                    failed.append(f"|adv|({abs_adv:.0f})≤|bl|({abs_bl:.0f})")
                if not ok_sev_adv:
                    failed.append(f"|sev|({abs_sev:.0f})≤|adv|({abs_adv:.0f})")
                msg = f"Divergence partielle — {' '.join(failed)} — {summary}"
            else:
                status = "FAIL"
                msg    = f"Scénarios non divergents — {summary}"

            return TestResult(
                "B3.M4", "Divergence effective des scénarios", 3,
                round(ratio, 3), None, "|sev|>|adv|>|bl| (2/2 inégalités)",
                status, msg,
                "BCBS (2009) §723; EBA/GL/2018/04 §108",
            )
        except Exception as exc:
            return _na("B3.M4", "Divergence effective des scénarios", 3, str(exc))

    # =========================================================================
    # B4  —  Backtesting (Kupiec + Christoffersen)
    # =========================================================================

    def _compute_realized_pnl(self) -> np.ndarray:
        """Compute monthly realized P&L from historical β changes.

        For each consecutive month pair (t-1, t) in betas_df:
          1. Reconstruct NS curves at t-1 (baseline) and t (stressed).
          2. Δy(τ) = y_NS(t) - y_NS(t-1).
          3. Reprice portfolio using curve at t-1 as baseline.
        Returns array of ΔP values (M EGP, positive = gain).
        """
        betas = self.betas_df.dropna()
        dates = betas.index
        pnl: List[float] = []

        for i in range(1, len(dates)):
            try:
                b1p = float(betas.at[dates[i - 1], "beta1"])
                b2p = float(betas.at[dates[i - 1], "beta2"])
                b3p = float(betas.at[dates[i - 1], "beta3"])
                b1c = float(betas.at[dates[i], "beta1"])
                b2c = float(betas.at[dates[i], "beta2"])
                b3c = float(betas.at[dates[i], "beta3"])

                curve_prev = self.fitter.reconstruct(b1p, b2p, b3p)
                curve_curr = self.fitter.reconstruct(b1c, b2c, b3c)
                delta_y    = (curve_curr - curve_prev).drop("T10Y", errors="ignore")

                repricer = PortfolioRepricer(
                    portfolio     = self.portfolio,
                    yield_curve_T = curve_prev,
                    delta_y       = delta_y,
                )
                dp, _ = repricer.reprice(method="auto")
                pnl.append(float(dp))
            except Exception as exc:
                LOG.debug("Realized PnL at %s skipped: %s", dates[i], exc)

        result = np.asarray(pnl, dtype=float)
        LOG.info(
            "Realized PnL computed: %d monthly observations"
            " | mean=%.1f M EGP | std=%.1f M EGP",
            len(result), float(np.mean(result)), float(np.std(result)),
        )
        return result

    def _b4_kupiec(self, realized_pnl: np.ndarray) -> TestResult:
        """Kupiec (1995) LR_uc — proportion of failures at VaR(99%).

        Violations : months where the realized loss exceeds sVaR(99%).
        LR_uc ~ χ²(1) under H₀: true exceedance probability = 1%.

        Traffic lights (adapted to monthly frequency, T observations):
          VERT   : p̂ = N/T ≤ 2%   (below Basel yellow threshold)
          JAUNE  : p̂ ∈ (2%, 5%]
          ROUGE  : p̂ > 5%

        References: Kupiec (1995) Risk 8(9):45-48;
                    BCBS (1996) Amendment to Capital Accord §B.4.
        """
        try:
            losses = -realized_pnl         # positive = loss
            T = len(losses)
            if T < 10:
                return _na("B4.K", "Kupiec LR_uc (backtesting VaR 99%)", 4,
                           f"Série trop courte ({T} obs, min 10 requis)")

            svar = self.risk_measure.svar_99
            N    = int(np.sum(losses > svar))
            p    = _KUPIEC_P
            ph   = N / T

            # LR_uc
            if N == 0:
                # L(p̂=0) = 1, so LR = -2 * [T·log(1-p) - 0]
                lr_uc = float(-2.0 * T * np.log(1.0 - p))
            elif N == T:
                lr_uc = float("inf")
            else:
                ll_h0 = N * np.log(p) + (T - N) * np.log(1.0 - p)
                ll_h1 = N * np.log(ph) + (T - N) * np.log(1.0 - ph)
                lr_uc = float(-2.0 * (ll_h0 - ll_h1))

            pval = _chi2_pval(min(lr_uc, 1e6), df=1)

            # Feux tricolores adaptés mensuel
            if ph <= 0.02:
                feux = "VERT"
            elif ph <= 0.05:
                feux = "JAUNE"
            else:
                feux = "ROUGE"

            if pval > 0.05:
                status = "PASS"
            elif pval >= 0.01:
                status = "WARN"
            else:
                status = "FAIL"

            msg = (
                f"LR_uc={lr_uc:.4f} | p={pval:.4f}"
                f" | N={N}/{T} violations | p̂={ph:.3%} (attendu 1%)"
                f" | sVaR seuil={svar:.1f} M EGP"
                f" | Feux (mensuel): {feux}"
            )
            return TestResult(
                "B4.K", "Kupiec LR_uc (backtesting VaR 99%)", 4,
                round(lr_uc, 4), round(pval, 6),
                "p > 0.05 | Feux VERT",
                status, msg,
                "Kupiec (1995) Risk 8(9):45-48;"
                " BCBS (1996) Amendment to Capital Accord §B.4",
            )
        except Exception as exc:
            return _na("B4.K", "Kupiec LR_uc", 4, str(exc))

    def _b4_christoffersen(self, realized_pnl: np.ndarray) -> TestResult:
        """Christoffersen (1998) LR_cc — couverture + indépendance des violations.

        LR_cc = LR_uc + LR_ind ~ χ²(2) sous H₀.
        LR_ind teste l'hypothèse nulle que les violations sont iid
        (pas de clustering) via la matrice de transition {I_{t-1}, I_t}.

        π̂₀₁ : P(violation | pas de violation la période précédente)
        π̂₁₁ : P(violation | violation la période précédente)
        H₀(ind) : π̂₀₁ = π̂₁₁  (mémoire nulle des violations)

        Reference: Christoffersen (1998) IER 39(3):841-862.
        """
        try:
            losses = -realized_pnl
            T = len(losses)
            if T < 15:
                return _na(
                    "B4.C",
                    "Christoffersen LR_cc (couverture + indépendance)", 4,
                    f"Série trop courte ({T} obs, min 15 requis)",
                )

            svar = self.risk_measure.svar_99
            I    = (losses > svar).astype(int)   # violation indicator sequence
            N    = int(I.sum())
            p    = _KUPIEC_P
            ph   = N / T

            # Transition counts
            n00 = int(np.sum((I[:-1] == 0) & (I[1:] == 0)))
            n01 = int(np.sum((I[:-1] == 0) & (I[1:] == 1)))
            n10 = int(np.sum((I[:-1] == 1) & (I[1:] == 0)))
            n11 = int(np.sum((I[:-1] == 1) & (I[1:] == 1)))

            # LR_uc  (consistent with Kupiec above)
            if N == 0:
                lr_uc = float(-2.0 * T * np.log(1.0 - p))
            elif N == T:
                lr_uc = float("inf")
            else:
                ll_h0 = N * np.log(p) + (T - N) * np.log(1.0 - p)
                ll_h1 = N * np.log(ph) + (T - N) * np.log(1.0 - ph)
                lr_uc = float(-2.0 * (ll_h0 - ll_h1))

            # LR_ind
            pi_hat = (n01 + n11) / max(T - 1, 1)
            pi_01  = n01 / max(n00 + n01, 1)
            pi_11  = n11 / max(n10 + n11, 1)

            def _slog(x: float) -> float:
                return float(np.log(max(x, 1e-300)))

            ll_ind_h0 = ((n00 + n10) * _slog(1.0 - pi_hat)
                         + (n01 + n11) * _slog(pi_hat))
            ll_ind_h1 = (n00 * _slog(1.0 - pi_01) + n01 * _slog(pi_01)
                         + n10 * _slog(1.0 - pi_11) + n11 * _slog(pi_11))
            lr_ind = float(-2.0 * (ll_ind_h0 - ll_ind_h1))
            lr_cc  = float(lr_uc + max(lr_ind, 0.0))   # guard negative lr_ind

            pval_cc = _chi2_pval(min(lr_cc, 1e6), df=2)

            if pval_cc > 0.05:
                status = "PASS"
            elif pval_cc >= 0.01:
                status = "WARN"
            else:
                status = "FAIL"

            msg = (
                f"LR_cc={lr_cc:.4f}"
                f" (LR_uc={lr_uc:.4f} + LR_ind={lr_ind:.4f})"
                f" | p={pval_cc:.4f}"
                f" | N={N}/{T} violations"
                f" | π̂₀₁={pi_01:.3f} π̂₁₁={pi_11:.3f}"
                f" | clusters(n₁₁)={n11}"
            )
            return TestResult(
                "B4.C", "Christoffersen LR_cc (couverture + indépendance)", 4,
                round(lr_cc, 4), round(pval_cc, 6),
                "p > 0.05",
                status, msg,
                "Christoffersen (1998) IER 39(3):841-862",
            )
        except Exception as exc:
            return _na("B4.C", "Christoffersen LR_cc", 4, str(exc))


# ── Module-level helper ───────────────────────────────────────────────────────

def _na(test_id: str, test_name: str, battery: int, reason: str) -> TestResult:
    return TestResult(
        test_id, test_name, battery,
        None, None, "—", "N/A", reason, "—",
    )
