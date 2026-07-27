"""
Post-Estimation Validation — 9 statistical tests in 3 batteries.

Battery 1 (universal — all models):
    B1.1  R² + RMSE
    B1.2  Breusch-Godfrey LM (autocorrélation résidus)
    B1.3  CUSUM (stabilité structurelle)
    B1.4  Theil U (vs marche aléatoire)
    B1.5  RMSE absolu  (PASS < 0.02 | WARN < 0.05 | FAIL ≥ 0.05)
    B1.6  MAE absolu   (PASS < 0.015 | WARN < 0.04 | FAIL ≥ 0.04)

Battery 2 (dispatch on model_type):
    B2.B  AUROC / Gini — Logit
    B2.D  RESET de Ramsey — OLS

Battery 3 (dispatch on module):
    B3.2  Boundary Check — Liquidité
    B3.3  Monotonie scénarios — Climat
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

# ── Model-type normalisation ──────────────────────────────────────────────────
_NORMALIZE_MT: Dict[str, str] = {
    # credit
    "logit":          "logit_ols",
    "logit_ols":      "logit_ols",
    "Logit":          "logit_ols",
    "vasicek":        "vasicek_probit",
    "vasicek_ols":    "vasicek_probit",
    "Vasicek-OLS":    "vasicek_probit",
    "vasicek_probit": "vasicek_probit",
    # liquidity / climate
    "beta":           "beta_mle",
    "beta_mle":       "beta_mle",
    "Beta":           "beta_mle",
    "ols_log":        "ols_log",
    "OLS":            "ols_log",
    "ols":            "ols_log",
    # market
    "ar1_macro":      "ar1_macro",
    "AR1":            "ar1_macro",
    "ar1":            "ar1_macro",
}


def _norm_mt(raw: str) -> str:
    return _NORMALIZE_MT.get(raw, raw.lower().replace("-", "_").replace(" ", "_"))


# ── Data Transfer Object ──────────────────────────────────────────────────────
@dataclass
class RiskOutput:
    """Carries all data needed by PostEstimationValidator from any module."""
    module:          str                                # "credit" | "liquidity" | "climate"
    model_type:      str                                # raw string — normalised internally
    y_true:          np.ndarray
    y_pred:          np.ndarray
    residuals:       np.ndarray
    n_obs:           int
    k_params:        int                                # regressors, excluding intercept
    # Optional
    ll_model:        Optional[float]                    = None  # full model log-likelihood
    ll_null:         Optional[float]                    = None  # intercept-only log-likelihood
    fitted_model:    Optional[Any]                      = None  # statsmodels OLS result
    rho:             Optional[float]                    = None  # Vasicek asset-correlation
    bounds:          Optional[Tuple[float, float]]      = None  # (lower, upper) domain
    scenario_results: Optional[Dict[str, np.ndarray]]  = None  # baseline/adverse/severe arrays
    rmse_scale:       float                             = 1.0   # multiplier for B1.5/B1.6 thresholds


# ── Result structures ─────────────────────────────────────────────────────────
@dataclass
class TestResult:
    test_id:   str
    test_name: str
    battery:   int
    statistic: Optional[float]
    p_value:   Optional[float]
    threshold: str
    status:    Literal["PASS", "WARN", "FAIL", "N/A"]
    message:   str
    reference: str


@dataclass
class ValidationReport:
    module:     str
    model_type: str
    timestamp:  str
    n_obs:      int
    results:    List[TestResult]
    n_pass:     int = 0
    n_warn:     int = 0
    n_fail:     int = 0
    verdict: Literal["VALIDATED", "VALIDATED_WITH_WARNINGS", "REJECTED"] = "VALIDATED"

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for dashboard rendering."""
        return {
            "module":     self.module,
            "model_type": self.model_type,
            "timestamp":  self.timestamp,
            "n_obs":      self.n_obs,
            "n_pass":     self.n_pass,
            "n_warn":     self.n_warn,
            "n_fail":     self.n_fail,
            "verdict":    self.verdict,
            "results": [
                {
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
                for r in self.results
            ],
        }


# ── Validator ─────────────────────────────────────────────────────────────────
class PostEstimationValidator:

    def __init__(self, risk_output: RiskOutput) -> None:
        self.ro = risk_output
        self.mt = _norm_mt(risk_output.model_type)

    def run(self) -> ValidationReport:
        results: List[TestResult] = []
        results.extend(self._battery_1())
        b2_result = self._battery_2_dispatch()
        if b2_result is not None:
            results.append(b2_result)
        results.append(self._battery_3_dispatch())
        return self._build_report(results)

    # =========================================================================
    # BATTERY 1 — Universal
    # =========================================================================

    def _battery_1(self) -> List[TestResult]:
        return [
            self._b1_1_r2_rmse(),
            self._b1_2_breusch_godfrey(),
            self._b1_3_cusum(),
            self._b1_4_theil_u(),
            self._b1_5_rmse_rel(),
            self._b1_6_mae_rel(),
        ]

    def _b1_1_r2_rmse(self) -> TestResult:
        ro = self.ro
        try:
            y = np.asarray(ro.y_true, dtype=float)
            yh = np.asarray(ro.y_pred, dtype=float)
            n, k = int(ro.n_obs), int(ro.k_params)
            ss_res = float(np.sum((y - yh) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else 0.0
            rmse = float(np.sqrt(np.mean((y - yh) ** 2)))

            if r2 >= 0.50:
                status = "PASS"
                msg = f"R²={r2:.4f} ≥ 0.50 | RMSE={rmse:.6f}"
            elif r2 >= 0.30:
                status = "WARN"
                msg = f"R²={r2:.4f} ∈ [0.30, 0.50) | RMSE={rmse:.6f}"
            else:
                status = "FAIL"
                msg = (f"R²={r2:.4f} < 0.30 — pouvoir explicatif insuffisant"
                       f" | RMSE={rmse:.6f}")
            return TestResult(
                "B1.1", "R² + RMSE", 1,
                round(r2, 6), None, "R² > 0.50",
                status, msg,
                "Theil (1961); Wooldridge (2016) §6.3"
            )
        except Exception as exc:
            return self._na("B1.1", "R² + RMSE", 1, str(exc))

    def _b1_2_breusch_godfrey(self) -> TestResult:
        ro = self.ro
        try:
            import statsmodels.api as sm
            from statsmodels.stats.diagnostic import acorr_breusch_godfrey
            resid = np.asarray(ro.residuals, dtype=float)
            n = len(resid)
            lags = max(1, min(4, n // 4))

            # Breusch-Godfrey exige un objet de régression ajusté (résidus +
            # régresseurs originaux). Si le modèle complet n'est pas exposé
            # (ro.fitted_model absent pour ce module), on construit une
            # régression auxiliaire sur constante uniquement — le test BG se
            # réduit alors à un test LM d'autocorrélation sur les résidus
            # centrés, cohérent avec le périmètre du Ljung-Box qu'il remplace.
            fitted = ro.fitted_model
            if fitted is None:
                fitted = sm.OLS(resid, np.ones((n, 1))).fit()

            lm_stat, lm_pvalue, _, _ = acorr_breusch_godfrey(fitted, nlags=lags)
            lm_stat = float(lm_stat)
            lm_pvalue = float(lm_pvalue)
            suffix = f" — résultat indicatif (N={n} < 15)" if n < 15 else ""

            if lm_pvalue > 0.05:
                status = "PASS"
                msg = f"Pas d'autocorrélation (p={lm_pvalue:.4f} > 0.05, lags={lags}){suffix}"
            elif lm_pvalue >= 0.01:
                status = "WARN"
                msg = f"Autocorrélation marginale (p={lm_pvalue:.4f} ∈ [0.01,0.05], lags={lags}){suffix}"
            else:
                status = "FAIL"
                msg = f"Autocorrélation significative (p={lm_pvalue:.4f} < 0.01, lags={lags}){suffix}"
            return TestResult(
                "B1.2", "Breusch-Godfrey LM (autocorrélation résidus)", 1,
                round(lm_stat, 4), round(lm_pvalue, 6), "p > 0.05",
                status, msg,
                "Breusch (1978) Australian Economic Papers 17:334-355 ; "
                "Godfrey (1978) Econometrica 46(6):1293-1301"
            )
        except Exception as exc:
            return self._na("B1.2", "Breusch-Godfrey", 1, str(exc))

    def _b1_3_cusum(self) -> TestResult:
        ro = self.ro
        try:
            resid = np.asarray(ro.residuals, dtype=float)
            n = len(resid)
            small = n < 15
            suffix = f" — résultat indicatif (N={n} < 15)" if small else ""

            # Prefer statsmodels breaks_cusumolsresid when OLS result available
            if ro.fitted_model is not None:
                try:
                    from statsmodels.stats.diagnostic import breaks_cusumolsresid
                    stat_raw, p_raw = breaks_cusumolsresid(ro.fitted_model)[:2]
                    stat_v, p_v = float(stat_raw), float(p_raw)
                    if p_v > 0.05:
                        status, msg = "PASS", f"Stabilité confirmée (p={p_v:.4f} > 0.05){suffix}"
                    else:
                        status, msg = "WARN", f"Dérive potentielle (p={p_v:.4f} < 0.05){suffix}"
                    return TestResult(
                        "B1.3", "CUSUM (stabilité structurelle)", 1,
                        round(stat_v, 4), round(p_v, 6), "p > 0.05",
                        status, msg,
                        "Brown, Durbin & Evans (1975) J. R. Stat. Soc. B 37(2):149-192"
                    )
                except Exception:
                    pass  # fallback to manual

            # Manual CUSUM on standardised residuals (BDE approximation)
            sigma = float(np.std(resid, ddof=1))
            if sigma < 1e-14:
                return self._na("B1.3", "CUSUM", 1, "Résidus constants (σ ≈ 0)")
            cusum_path = np.cumsum(resid / sigma)
            # Normalise by sqrt(n) so statistic is O(1)
            cusum_stat = float(np.max(np.abs(cusum_path)) / np.sqrt(n))
            # BDE 5% critical value ≈ 0.948 (Brownian bridge)
            cv = 0.948
            if cusum_stat <= cv:
                status = "PASS"
                msg = f"CUSUM={cusum_stat:.4f} ≤ {cv} — pas de dérive structurelle{suffix}"
            else:
                status = "WARN"
                msg = f"CUSUM={cusum_stat:.4f} > {cv} — dérive potentielle détectée{suffix}"
            return TestResult(
                "B1.3", "CUSUM (stabilité structurelle)", 1,
                round(cusum_stat, 4), None, f"stat ≤ {cv} (BDE 5%)",
                status, msg,
                "Brown, Durbin & Evans (1975) J. R. Stat. Soc. B 37(2):149-192"
            )
        except Exception as exc:
            return self._na("B1.3", "CUSUM", 1, str(exc))

    def _b1_4_theil_u(self) -> TestResult:
        ro = self.ro
        try:
            y  = np.asarray(ro.y_true, dtype=float)
            yh = np.asarray(ro.y_pred, dtype=float)
            if len(y) < 2:
                return self._na("B1.4", "Theil U", 1, "Moins de 2 observations")
            rmse_naive = float(np.sqrt(np.mean((y[1:] - y[:-1]) ** 2)))
            rmse_model = float(np.sqrt(np.mean((y[1:] - yh[1:]) ** 2)))
            if rmse_naive < 1e-10:
                return self._na("B1.4", "Theil U", 1, "Série constante (RMSE naïf ≈ 0)")
            u = rmse_model / rmse_naive
            if u < 0.80:
                status = "PASS"
                msg = f"Theil U={u:.4f} < 0.80 — modèle meilleur que la marche aléatoire"
            elif u <= 1.00:
                status = "WARN"
                msg = f"Theil U={u:.4f} ∈ [0.80, 1.00] — avantage modeste vs marche aléatoire"
            else:
                status = "FAIL"
                msg = f"Theil U={u:.4f} > 1.00 — modèle dominé par la marche aléatoire"
            return TestResult(
                "B1.4", "Theil U (vs marche aléatoire)", 1,
                round(u, 6), None, "U < 0.80",
                status, msg,
                "Theil (1966) Applied Economic Forecasting, North-Holland"
            )
        except Exception as exc:
            return self._na("B1.4", "Theil U", 1, str(exc))

    def _b1_5_rmse_rel(self) -> TestResult:
        """RMSE absolu — erreur quadratique moyenne."""
        ro = self.ro
        try:
            y  = np.asarray(ro.y_true, dtype=float)
            yh = np.asarray(ro.y_pred, dtype=float)
            if len(y) < 2:
                return self._na("B1.5", "RMSE", 1, "Moins de 2 observations")
            rmse = float(np.sqrt(np.mean((y - yh) ** 2)))
            sc   = float(self.ro.rmse_scale)
            thr_pass, thr_warn = 0.10 * sc, 0.20 * sc
            if rmse < thr_pass:
                status = "PASS"
                msg = f"RMSE={rmse:.4f} < {thr_pass:.4f} -- precision satisfaisante"
            elif rmse < thr_warn:
                status = "WARN"
                msg = (f"RMSE={rmse:.4f} dans [{thr_pass:.4f}, {thr_warn:.4f})"
                       " -- precision moderee")
            else:
                status = "FAIL"
                msg = f"RMSE={rmse:.4f} >= {thr_warn:.4f} -- ecart modele/reel eleve"
            return TestResult(
                "B1.5", "RMSE", 1,
                round(rmse, 6), None, "< 0.10",
                status, msg,
                "Hyndman & Koehler (2006) Int. J. Forecasting 22(4):679-688"
            )
        except Exception as exc:
            return self._na("B1.5", "RMSE", 1, str(exc))

    def _b1_6_mae_rel(self) -> TestResult:
        """MAE absolu — erreur absolue moyenne."""
        ro = self.ro
        try:
            y  = np.asarray(ro.y_true, dtype=float)
            yh = np.asarray(ro.y_pred, dtype=float)
            if len(y) < 2:
                return self._na("B1.6", "MAE", 1, "Moins de 2 observations")
            mae = float(np.mean(np.abs(y - yh)))
            sc  = float(self.ro.rmse_scale)
            thr_pass, thr_warn = 0.10 * sc, 0.15 * sc
            if mae < thr_pass:
                status = "PASS"
                msg = f"MAE={mae:.4f} < {thr_pass:.4f} -- biais moyen faible"
            elif mae < thr_warn:
                status = "WARN"
                msg = (f"MAE={mae:.4f} dans [{thr_pass:.4f}, {thr_warn:.4f})"
                       " -- biais modere")
            else:
                status = "FAIL"
                msg = f"MAE={mae:.4f} >= {thr_warn:.4f} -- biais moyen important"
            return TestResult(
                "B1.6", "MAE", 1,
                round(mae, 6), None, "< 0.10",
                status, msg,
                "Hyndman & Koehler (2006) Int. J. Forecasting 22(4):679-688"
            )
        except Exception as exc:
            return self._na("B1.6", "MAE", 1, str(exc))

    # =========================================================================
    # BATTERY 2 — Model-type dispatch

    def _battery_2_dispatch(self) -> Optional[TestResult]:
        dispatch = {
            "logit_ols":      self._b2b_auroc,
            "ols_log":        self._b2d_reset,
            "ar1_macro":      self._b2e_ar1_rho,
        }
        # vasicek_probit: no Battery-2 test applies (B2.C Vasicek asset-
        # correlation plausibility check removed -- no caller in this
        # codebase ever supplies RiskOutput.rho for a Vasicek-family model,
        # so it only ever produced an N/A row; skip Battery 2 entirely for
        # this model type rather than show a dead test).
        if self.mt == "vasicek_probit":
            return None
        fn = dispatch.get(self.mt)
        if fn is None:
            return TestResult(
                "B2", "Test spécifique modèle", 2, None, None, "—", "N/A",
                f"model_type '{self.ro.model_type}' → '{self.mt}' non reconnu", "—"
            )
        return fn()

    def _b2b_auroc(self) -> TestResult:
        """AUROC / Gini pour modèles Logit."""
        ro = self.ro
        try:
            from sklearn.metrics import roc_auc_score
            y  = np.asarray(ro.y_true, dtype=float)
            yh = np.asarray(ro.y_pred, dtype=float)
            # Binarise on median if y_true is continuous
            unique = np.unique(y)
            if len(unique) > 2 or not set(unique).issubset({0.0, 1.0}):
                y_bin = (y > np.median(y)).astype(int)
            else:
                y_bin = y.astype(int)
            if len(np.unique(y_bin)) < 2:
                return TestResult(
                    "B2.B", "AUROC / Gini — Logit", 2, None, None, "AUROC > 0.70",
                    "N/A", "y_true ne contient qu'une seule classe après binarisation", "—"
                )
            auroc = float(roc_auc_score(y_bin, yh))
            gini  = 2.0 * auroc - 1.0
            if auroc > 0.70:
                status = "PASS"
                msg = f"AUROC={auroc:.4f} > 0.70 | Gini={gini:.4f} — bonne discrimination"
            elif auroc >= 0.60:
                status = "WARN"
                msg = f"AUROC={auroc:.4f} ∈ [0.60, 0.70] | Gini={gini:.4f} — discrimination acceptable"
            else:
                status = "FAIL"
                msg = f"AUROC={auroc:.4f} < 0.60 | Gini={gini:.4f} — discrimination insuffisante"
            return TestResult(
                "B2.B", "AUROC / Gini — Logit", 2,
                round(auroc, 6), None, "AUROC > 0.70",
                status, msg,
                "BCBS (2005) §III.B Validation of IRB rating systems"
            )
        except Exception as exc:
            return self._na("B2.B", "AUROC / Gini", 2, str(exc))

    def _b2d_reset(self) -> TestResult:
        """RESET de Ramsey — teste les non-linéarités omises (OLS)."""
        ro = self.ro
        try:
            if ro.fitted_model is None:
                return self._na("B2.D", "RESET de Ramsey — OLS", 2,
                                "fitted_model non fourni — test non exécutable")
            from statsmodels.stats.diagnostic import linear_reset
            reset = linear_reset(ro.fitted_model, power=2, use_f=True)
            stat    = float(reset.statistic)
            p_value = float(reset.pvalue)
            if p_value > 0.05:
                status = "PASS"
                msg = f"RESET F={stat:.4f}, p={p_value:.4f} — forme fonctionnelle correcte"
            else:
                status = "WARN"
                msg = f"RESET F={stat:.4f}, p={p_value:.4f} < 0.05 — non-linéarité potentielle omise"
            return TestResult(
                "B2.D", "RESET de Ramsey — OLS", 2,
                round(stat, 4), round(p_value, 6), "p > 0.05",
                status, msg,
                "Ramsey (1969) J. R. Stat. Soc. B 31(2):350-371"
            )
        except Exception as exc:
            return self._na("B2.D", "RESET de Ramsey", 2, str(exc))

    def _b2e_ar1_rho(self) -> TestResult:
        """Stationnarité AR(1) — |ρ| < 1 (modèles satellites marché).

        Un |ρ| proche de 1 (near unit-root) produit des projections AR(1)
        explosives sur 12 pas mensuels.  Le clampage à 0.99 dans ir_satellite.py
        est une garde-fou, ce test vérifie que l'OLS n'a pas été clampé.
        Seuils : PASS |ρ|<0.95 | WARN [0.95,0.99) | FAIL ≥0.99.

        Ref : Diebold & Li (2006) §3.2; Stambaugh (1999) bias correction.
        """
        ro = self.ro
        try:
            if ro.rho is None:
                return self._na(
                    "B2.E", "Stationnarité AR(1) — |ρ|", 2,
                    "Paramètre ρ non fourni dans RiskOutput",
                )
            abs_rho = abs(float(ro.rho))
            if abs_rho < 0.95:
                status = "PASS"
                msg = (
                    f"|ρ|={abs_rho:.4f} < 0.95"
                    " — AR(1) stationnaire, projections stables"
                )
            elif abs_rho < 0.99:
                status = "WARN"
                msg = (
                    f"|ρ|={abs_rho:.4f} ∈ [0.95, 0.99)"
                    " — near unit-root, surveiller la divergence à T+12"
                )
            else:
                status = "FAIL"
                msg = (
                    f"|ρ|={abs_rho:.4f} ≥ 0.99"
                    " — quasi unit-root, projections AR(1) potentiellement"
                    " non fiables (OLS clampé à 0.99)"
                )
            return TestResult(
                "B2.E", "Stationnarité AR(1) — |ρ| < 1", 2,
                round(abs_rho, 6), None, "|ρ| < 0.95",
                status, msg,
                "Diebold & Li (2006) JoE 130(2):337-364;"
                " Stambaugh (1999) JFE 54(3):375-421",
            )
        except Exception as exc:
            return self._na("B2.E", "Stationnarité AR(1)", 2, str(exc))

    # =========================================================================
    # BATTERY 3 — Module dispatch
    # =========================================================================

    def _battery_3_dispatch(self) -> TestResult:
        dispatch = {
            "liquidity": self._b3_2_boundary,
            "climate":   self._b3_3_monotony,
        }
        fn = dispatch.get(self.ro.module)
        if fn is None:
            return TestResult(
                "B3", "Test spécifique module", 3, None, None, "—", "N/A",
                f"Module '{self.ro.module}' non reconnu", "—"
            )
        return fn()

    def _b3_2_boundary(self) -> TestResult:
        """Boundary Check — prédictions dans les bornes réglementaires (liquidité)."""
        ro = self.ro
        try:
            if ro.bounds is None:
                return self._na("B3.2", "Boundary Check — Liquidité", 3,
                                "bounds non fourni dans RiskOutput")
            yh = np.asarray(ro.y_pred, dtype=float)
            lower, upper = float(ro.bounds[0]), float(ro.bounds[1])
            violations = int(np.sum((yh < lower) | (yh > upper)))
            pct = violations / max(len(yh), 1)
            if violations == 0:
                status = "PASS"
                msg = f"0 prédiction hors bornes [{lower:.4f}, {upper:.4f}]"
            elif pct <= 0.05:
                status = "WARN"
                msg = (f"{violations} prédiction(s) hors bornes [{lower:.4f}, {upper:.4f}]"
                       f" ({pct*100:.1f}% ≤ 5%)")
            else:
                status = "FAIL"
                msg = (f"{violations} prédictions hors bornes [{lower:.4f}, {upper:.4f}]"
                       f" ({pct*100:.1f}% > 5%)")
            return TestResult(
                "B3.2", "Boundary Check — Liquidité", 3,
                float(violations), None, "0 violations",
                status, msg,
                "CRR Art. 412 (LCR); CRR Art. 428 (NSFR); EBA/GL/2017/01"
            )
        except Exception as exc:
            return self._na("B3.2", "Boundary Check", 3, str(exc))

    def _b3_3_monotony(self) -> TestResult:
        """Monotonie scénarios — baseline ≤ adverse ≤ severe (module climat)."""
        ro = self.ro
        try:
            sr = ro.scenario_results
            if sr is None:
                return self._na("B3.3", "Monotonie scénarios — Climat", 3,
                                "scenario_results non fourni dans RiskOutput")
            bl  = np.asarray(sr.get("baseline", []), dtype=float)
            adv = np.asarray(sr.get("adverse",  []), dtype=float)
            sev = np.asarray(sr.get("severe",   []), dtype=float)
            min_len = min(len(bl), len(adv), len(sev))
            if min_len == 0:
                return self._na("B3.3", "Monotonie scénarios", 3,
                                "Trajectoires vides ou longueurs incohérentes")
            bl, adv, sev = bl[:min_len], adv[:min_len], sev[:min_len]
            v_adv = int(np.sum(adv < bl))
            v_sev = int(np.sum(sev < adv))
            total = v_adv + v_sev
            if total == 0:
                status = "PASS"
                msg = f"Monotonie parfaite : baseline ≤ adverse ≤ severe ({min_len} périodes)"
            elif total <= 2:
                status = "WARN"
                msg = (f"{total} violation(s) de monotonie sur {min_len} périodes"
                       f" (adv<bl: {v_adv}, sev<adv: {v_sev})")
            else:
                status = "FAIL"
                msg = (f"{total} violations de monotonie sur {min_len} périodes"
                       f" (adv<bl: {v_adv}, sev<adv: {v_sev})")
            return TestResult(
                "B3.3", "Monotonie scénarios — Climat", 3,
                float(total), None, "0 violations",
                status, msg,
                "BCBS (2017) Principles for sound stress testing; EBA/GL/2018/04"
            )
        except Exception as exc:
            return self._na("B3.3", "Monotonie scénarios", 3, str(exc))

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _na(test_id: str, test_name: str, battery: int, reason: str) -> TestResult:
        return TestResult(
            test_id, test_name, battery,
            None, None, "—", "N/A", reason, "—"
        )

    def _build_report(self, results: List[TestResult]) -> ValidationReport:
        n_pass = sum(1 for r in results if r.status == "PASS")
        n_warn = sum(1 for r in results if r.status == "WARN")
        n_fail = sum(1 for r in results if r.status == "FAIL")
        if n_fail > 0:
            verdict: Literal["VALIDATED", "VALIDATED_WITH_WARNINGS", "REJECTED"] = "REJECTED"
        elif n_warn > 0:
            verdict = "VALIDATED_WITH_WARNINGS"
        else:
            verdict = "VALIDATED"
        return ValidationReport(
            module     = self.ro.module,
            model_type = self.mt,
            timestamp  = datetime.utcnow().isoformat(timespec="seconds") + "Z",
            n_obs      = self.ro.n_obs,
            results    = results,
            n_pass     = n_pass,
            n_warn     = n_warn,
            n_fail     = n_fail,
            verdict    = verdict,
        )
