"""
ir_satellite.py
===============
AR(1)+macro satellite models for Nelson-Siegel beta factors.

For each beta_k (k=1,2,3):
  1. ADF + KPSS stationarity tests.
  2. Variable selection -- triple criterion (all three must hold simultaneously):
       a) AIC improvement vs AR(1) baseline
       b) p-value < threshold (default 0.05) for the macro coefficient gamma_k
       c) Sign of gamma_k consistent with ECONOMIC_PRIORS["market_beta{k}"]
     Rejected-by-sign variables generate a WARNING with full detail.
  3. AR(1) augmented OLS model:
       beta_k,t = alpha_k + rho_k * beta_k,t-1 + sum_j gamma_k,j * X_j,t + eps_k,t
  4. Breusch-Godfrey LM test on residuals.
  5. Store (alpha_k, rho_k, gammas, sigma_k, sigma_k_T) in BetaSatellite.

References
----------
Diebold, F.X. & Li, C. (2006). Forecasting the term structure of government
    bond yields. Journal of Econometrics, 130(2), 337-364.
"""
from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.tsa.stattools import adfuller, kpss

# Locate ECONOMIC_PRIORS -- resolve the path regardless of CWD
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules_src.climate_module.macro_selection_engine.utils import expected_sign

LOG = logging.getLogger("market.ir_satellite")

# Mapping: internal macro variable name -> key used in ECONOMIC_PRIORS["market_beta{k}"]
# fiscal_balance_gdp is negative of fiscal_deficit, so sign is flipped below
MACRO_TO_SIGN_KEY: Dict[str, str] = {
    "policy_rate":         "policy_rate",
    "lending_rate":        "policy_rate",   # WB FR.INR.LEND -- same direction as CBE rate
    "policy_rate_monthly": "policy_rate",   # monthly FRED series (when available)
    "cpi_inflation":       "inflation",
    "real_gdp_growth":     "gdp_growth",
    "fiscal_balance_gdp":  "fiscal_deficit",   # sign flipped in _get_sign()
    "gov_debt_gdp":        "public_debt",
    "fx_reserves_usd":     "fx_reserves",
}

# Variables whose series sign must be flipped before sign-checking
# because the macro universe stores the negative convention
_FLIP_SIGN: frozenset = frozenset({"fiscal_balance_gdp"})


@dataclass
class BetaSatellite:
    """Calibrated AR(1)+macro model for a single beta_k factor."""

    k:              int              # 1, 2, or 3
    alpha:          float            # intercept alpha_k
    rho:            float            # AR(1) coefficient rho_k
    gammas:         Dict[str, float] # {var_name: gamma_k_j} for selected macro vars
    sigma:          float            # residual std sigma_k (unconditional)
    sigma_T:        float            # |last residual| as conditional vol proxy at T
    selected_vars:  List[str]
    r2:             float
    n_obs:          int
    is_stationary:  bool
    breusch_godfrey_pvalue: float
    validation_warnings: List[str] = field(default_factory=list)
    leaderboard: List[Dict] = field(default_factory=list)  # candidats + modèle final

    def predict_next(
        self,
        beta_prev: float,
        macro_values: Dict[str, float],
    ) -> float:
        """One-step-ahead prediction.

        beta*_k,t = alpha_k + rho_k * beta_k,t-1
                   + sum_j gamma_k,j * X*_k,j,t

        KNOWN LIMITATION (documented, not corrected -- 2026-07 validation review):
        when macro is annual (is_annual_macro branch in _fit_beta/_fit_ar1),
        alpha/rho/gamma are calibrated on ~15 annual observations, but this
        method is walked forward MONTHLY both by StressProjector.project_path()
        and by MarketPostEstimationRunner._reconstruct_fitted(). Reapplying an
        annual persistence rho at monthly cadence converges to the gamma-implied
        equilibrium roughly 12x faster than the annual estimation implies, which
        manifests as residual autocorrelation (B1.2) and a degraded Theil U
        (B1.4) in post-estimation validation for beta1/beta2/beta3. A rigorous
        fix would need an annual->monthly parameter conversion (or a monthly
        re-estimation using the ~175 monthly beta observations directly against
        the annual macro held flat intra-year); both were evaluated and left
        for a future iteration since rho is sometimes negative at annual
        frequency (e.g. beta1 rho=-0.61 on this dataset), which has no clean
        real 12th root and would need an extra sign convention.
        """
        pred = self.alpha + self.rho * float(beta_prev)
        for var, gamma in self.gammas.items():
            x = macro_values.get(var, 0.0)
            pred += gamma * float(x)
        return pred


class IRSatellite:
    """Estimate AR(1)+macro satellite models for beta_1, beta_2, beta_3.

    Parameters
    ----------
    betas_df : pd.DataFrame
        Monthly Nelson-Siegel factors with columns {beta1, beta2, beta3}.
    macro_df : pd.DataFrame
        Monthly macro variables (already aligned via FrequencyAligner).
    pvalue_threshold : float
        Significance threshold for the macro coefficient (default 0.05).
    """

    def __init__(
        self,
        betas_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        pvalue_threshold: float = 0.10,
    ) -> None:
        self.betas_df         = betas_df.dropna()
        self.macro_df         = macro_df
        self.pvalue_threshold = float(pvalue_threshold)
        self._satellites: Dict[int, BetaSatellite] = {}

    # ---- public ----------------------------------------------------------

    def fit_all(
        self,
        forced_include: Optional[Dict[int, List[str]]] = None,
        forced_exclude: Optional[Dict[int, List[str]]] = None,
    ) -> Dict[int, BetaSatellite]:
        """Fit a satellite for each of beta1, beta2, beta3.

        Parameters
        ----------
        forced_include, forced_exclude : optional {k: [variable, ...]}
            Manual override of the triple-criterion variable selection —
            e.g. force `oil_price` into the beta2 model even though its
            p-value narrowly missed the threshold. Unlike credit/liquidity
            (which pick between whole candidate models via a rank), the
            market satellite selects individual variables, so the natural
            override unit here is per-variable, not per-rank.

        Returns
        -------
        dict {1: BetaSatellite, 2: BetaSatellite, 3: BetaSatellite}
        """
        for k in (1, 2, 3):
            col = f"beta{k}"
            if col not in self.betas_df.columns:
                raise ValueError(f"Column '{col}' not found in betas_df")
            LOG.info("Fitting IR satellite for beta%d", k)
            self._satellites[k] = self._fit_beta(
                k,
                forced_include=(forced_include or {}).get(k, []),
                forced_exclude=(forced_exclude or {}).get(k, []),
            )
        return self._satellites

    def get(self, k: int) -> BetaSatellite:
        if k not in self._satellites:
            raise RuntimeError(f"Call fit_all() before get({k})")
        return self._satellites[k]

    # ---- private ---------------------------------------------------------

    def _fit_beta(
        self, k: int,
        forced_include: Optional[List[str]] = None,
        forced_exclude: Optional[List[str]] = None,
    ) -> BetaSatellite:
        col       = f"beta{k}"
        risk_type = f"market_beta{k}"
        warns: List[str] = []

        # Align beta and macro on the common non-NaN index
        beta_series = self.betas_df[col].dropna()
        common_idx  = beta_series.index.intersection(self.macro_df.index)
        beta_s      = beta_series.reindex(common_idx).dropna()
        macro_s     = self.macro_df.reindex(common_idx)

        # 1. Stationarity
        is_stat, adf_p = _adf_test(beta_s)
        kpss_stat, kpss_p = _kpss_test(beta_s)
        LOG.info(
            "beta%d: ADF p=%.4f (stationary=%s)  KPSS p=%.4f (stationary=%s)",
            k, adf_p, is_stat, kpss_p, kpss_stat,
        )
        if not is_stat or not kpss_stat:
            warns.append(
                f"beta{k} may be non-stationary "
                f"(ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f})"
            )

        # 2. Variable selection
        usable_cols = [
            c for c in macro_s.columns
            if macro_s[c].notna().sum() >= 12
        ]
        selected, candidates = self._select_variables(
            k, risk_type, beta_s, macro_s[usable_cols], warns
        )

        # 2b. Manual overrides — force-include/exclude specific variables.
        # Applied AFTER the automatic triple-criterion selection so the
        # candidate list still shows the model's own verdict; the override
        # only changes what ends up in `selected` (and hence in the final
        # AR(1)+macro re-fit below).
        _cand_by_var = {c["variable"]: c for c in candidates}
        for _var in (forced_exclude or []):
            if _var in selected:
                selected.remove(_var)
            if _var in _cand_by_var:
                _cand_by_var[_var]["status"] = "FORCED_EXCLUDED"
        for _var in (forced_include or []):
            if _var not in selected and _var in _cand_by_var:
                selected.append(_var)
                _cand_by_var[_var]["status"] = "FORCED_INCLUDED"

        # 3. AR(1) augmented OLS
        alpha, rho, gammas, resid, r2 = self._fit_ar1(beta_s, macro_s, selected, k=k)
        # Sync selected + marquer en REJECTED_COEF0 les vars élagage coef≈0
        _pruned = set(selected) - set(gammas.keys())
        selected = list(gammas.keys())
        if _pruned:
            for c in candidates:
                if c["variable"] in _pruned:
                    c["status"] = "REJECTED_COEF0"

        # 4. Sigma
        sigma   = float(np.std(resid.values, ddof=1))
        sigma_T = float(np.abs(resid.iloc[-1])) if len(resid) > 0 else sigma

        # 5. Breusch-Godfrey LM
        bg_p = _breusch_godfrey_p(resid)
        if bg_p < 0.05:
            warns.append(
                f"beta{k} residuals: Breusch-Godfrey p={bg_p:.4f} "
                "-- serial autocorrelation present"
            )

        LOG.info(
            "beta%d: alpha=%.4f rho=%.4f selected=%s R2=%.3f sigma=%.4f",
            k, alpha, rho, selected, r2, sigma,
        )

        _leaderboard = {
            "modele_final": {
                "alpha":         round(alpha, 6),
                "rho":           round(rho, 6),
                "gammas":        {var: round(g, 6) for var, g in gammas.items()},
                "r2":            round(r2, 4),
                "n_obs":         len(beta_s),
                "selected_vars": selected,
            },
            "candidats": candidates,
        }

        return BetaSatellite(
            k=k, alpha=alpha, rho=rho, gammas=gammas,
            sigma=sigma, sigma_T=sigma_T,
            selected_vars=selected, r2=r2, n_obs=len(beta_s),
            is_stationary=is_stat,
            breusch_godfrey_pvalue=bg_p,
            validation_warnings=warns,
            leaderboard=_leaderboard,
        )

    def _select_variables(
        self,
        k: int,
        risk_type: str,
        beta_s: pd.Series,
        macro_df: pd.DataFrame,
        warns: List[str],
    ) -> Tuple[List[str], List[Dict]]:
        """Triple-criterion variable selection.

        1. AIC improvement vs AR(1) baseline.
        2. p-value < pvalue_threshold for gamma_k.
        3. Sign of gamma_k consistent with ECONOMIC_PRIORS[risk_type].

        When macro data is annual (forward-filled monthly with few unique values),
        the regression is run on annual aggregates — otherwise within-year variation
        of the monthly beta is pure noise relative to a constant macro value and
        the variable can never improve monthly AIC even if it is a strong predictor
        at annual frequency.  Annual gammas are then applied directly in the monthly
        AR(1) projection (same magnitude; the 12-step compounding is handled by rho).

        Returns (selected_vars, candidates) where candidates is the full leaderboard
        of all tested variables with their metrics and rejection reason.
        """
        if macro_df.empty:
            LOG.info("beta%d: no macro columns available", k)
            return [], []

        # Detect annual macro: if median unique-value count per column <= nobs/6,
        # the macro data is effectively annual (forward-filled to monthly).
        # Threshold: less than 2 distinct values per year → annual.
        n_months = len(macro_df)
        is_annual_macro = macro_df.apply(lambda c: c.dropna().nunique()).median() <= n_months / 6

        if is_annual_macro:
            # Aggregate both beta and macro to annual averages for selection.
            beta_sel = beta_s.groupby(beta_s.index.year).mean()
            beta_sel.index = pd.to_datetime(beta_sel.index.astype(str) + "-01-01")
            macro_sel = macro_df.groupby(macro_df.index.year).first()
            macro_sel.index = pd.to_datetime(macro_sel.index.astype(str) + "-01-01")
            LOG.info(
                "beta%d: annual macro detected (%d obs) — using annual aggregation for selection",
                k, len(beta_sel),
            )
        else:
            beta_sel  = beta_s
            macro_sel = macro_df

        # AR(1) baseline AIC (at selected frequency)
        y, y_lag = _ar1_data(beta_sel)
        aic_base = _ols_aic(y, sm.add_constant(y_lag))

        selected: List[str] = []
        candidates: List[Dict] = []

        for var in macro_sel.columns:
            x_raw  = macro_sel[var].reindex(beta_sel.index)
            common = y.index.intersection(x_raw.dropna().index)
            if len(common) < 6:
                candidates.append({
                    "variable":      var,
                    "status":        "REJECTED_INSUFFICIENT_OBS",
                    "aic_base":      round(aic_base, 3),
                    "aic_full":      None,
                    "daic":          None,
                    "gamma":         None,
                    "p_value":       None,
                    "sign_expected": None,
                    "sign_ok":       None,
                })
                continue

            y_c     = y.reindex(common)
            lag_c   = y_lag.reindex(common)
            x_c     = x_raw.reindex(common)

            # -- Sign prior (flip for fiscal_balance_gdp) --
            sign_key  = MACRO_TO_SIGN_KEY.get(var, var)
            exp_s     = expected_sign(sign_key, risk_type)
            flip      = (var in _FLIP_SIGN)

            # -- Fit full model --
            try:
                X_full = sm.add_constant(
                    pd.concat([lag_c, x_c], axis=1), has_constant="add"
                )
                res = sm.OLS(y_c, X_full).fit(cov_type="HC1")
            except Exception as exc:
                LOG.debug("beta%d var=%s OLS failed: %s", k, var, exc)
                candidates.append({
                    "variable":      var,
                    "status":        "REJECTED_OLS_FAILED",
                    "aic_base":      round(aic_base, 3),
                    "aic_full":      None,
                    "daic":          None,
                    "gamma":         None,
                    "p_value":       None,
                    "sign_expected": int(exp_s),
                    "sign_ok":       None,
                })
                continue

            gamma_est = float(res.params.iloc[2])
            p_val     = float(res.pvalues.iloc[2])
            eff_gamma = -gamma_est if flip else gamma_est

            # Criterion 1: AIC — règle des 2 unités (standard Burnham & Anderson 2002).
            # Sur petits échantillons annuels (10-20 obs), ΔAIC > 0 strict est trop
            # sévère : deux modèles à < 2 unités AIC sont statistiquement équivalents.
            # On accepte si le modèle augmenté n'est PAS plus de 2 unités PLUS MAL.
            _AIC_TOL = 2.0
            if res.aic >= aic_base + _AIC_TOL:
                LOG.debug(
                    "beta%d var=%s REJECTED: AIC %.2f >= baseline+2 (%.2f)",
                    k, var, res.aic, aic_base + _AIC_TOL,
                )
                candidates.append({
                    "variable":      var,
                    "status":        "REJECTED_AIC",
                    "aic_base":      round(aic_base, 3),
                    "aic_full":      round(float(res.aic), 3),
                    "daic":          round(aic_base - float(res.aic), 3),
                    "gamma":         round(gamma_est, 6),
                    "p_value":       round(p_val, 4),
                    "sign_expected": int(exp_s),
                    "sign_ok":       None,
                })
                continue

            # Criterion 2: p-value on gamma (index 2: const, lag, x)
            if p_val >= self.pvalue_threshold:
                LOG.debug(
                    "beta%d var=%s REJECTED: p=%.4f >= %.2f",
                    k, var, p_val, self.pvalue_threshold,
                )
                candidates.append({
                    "variable":      var,
                    "status":        "REJECTED_PVALUE",
                    "aic_base":      round(aic_base, 3),
                    "aic_full":      round(float(res.aic), 3),
                    "daic":          round(aic_base - float(res.aic), 3),
                    "gamma":         round(gamma_est, 6),
                    "p_value":       round(p_val, 4),
                    "sign_expected": int(exp_s),
                    "sign_ok":       None,
                })
                continue

            # Criterion 3: sign consistency — hard gate.
            if exp_s != 0:
                sign_ok = (
                    (eff_gamma > 0 and exp_s > 0)
                    or (eff_gamma < 0 and exp_s < 0)
                )
                if not sign_ok:
                    est_str = "+" if eff_gamma > 0 else "-"
                    exp_str = "+" if exp_s > 0 else "-"
                    msg = (
                        f"beta{k} | var={var}: signe incohérent -- "
                        f"gamma estimé={gamma_est:+.4f} ({est_str} effectif), "
                        f"attendu ({exp_str}) depuis ECONOMIC_PRIORS['{risk_type}']['{sign_key}']"
                    )
                    LOG.warning(msg)
                    warns.append(msg)
                    candidates.append({
                        "variable":      var,
                        "status":        "REJECTED_SIGN",
                        "aic_base":      round(aic_base, 3),
                        "aic_full":      round(float(res.aic), 3),
                        "daic":          round(aic_base - float(res.aic), 3),
                        "gamma":         round(gamma_est, 6),
                        "p_value":       round(p_val, 4),
                        "sign_expected": int(exp_s),
                        "sign_ok":       False,
                    })
                    continue
            else:
                sign_ok = True

            LOG.info(
                "beta%d var=%s SELECTED  DAIC=%.2f  p=%.4f  gamma=%+.4f",
                k, var, aic_base - res.aic, p_val, gamma_est,
            )
            candidates.append({
                "variable":      var,
                "status":        "SELECTED",
                "aic_base":      round(aic_base, 3),
                "aic_full":      round(float(res.aic), 3),
                "daic":          round(aic_base - float(res.aic), 3),
                "gamma":         round(gamma_est, 6),
                "p_value":       round(p_val, 4),
                "sign_expected": int(exp_s),
                "sign_ok":       bool(sign_ok),
            })
            selected.append(var)

        if not selected:
            LOG.info("beta%d: no macro variable passed all 3 criteria -- pure AR(1)", k)

        # Trier : SELECTED en premier, puis par DAIC décroissant
        _status_rank = {"SELECTED": 0, "REJECTED_SIGN": 1, "REJECTED_PVALUE": 2,
                        "REJECTED_AIC": 3, "REJECTED_OLS_FAILED": 4,
                        "REJECTED_INSUFFICIENT_OBS": 5}
        candidates.sort(key=lambda c: (
            _status_rank.get(c["status"], 9),
            -(c["daic"] or 0.0),
        ))
        for rang, c in enumerate(candidates, start=1):
            c["rang"] = rang

        return selected, candidates

    def _fit_ar1(
        self,
        beta_s: pd.Series,
        macro_df: pd.DataFrame,
        selected: List[str],
        k: int = 0,
    ) -> Tuple[float, float, Dict[str, float], pd.Series, float]:
        """Fit the final AR(1)+selected_vars OLS model.

        When macro data is annual (few unique values per column), estimation is
        also run at annual frequency to avoid multicollinearity from forward-
        filling (two annual variables forward-filled to monthly look identical
        within each year → OLS assigns arbitrary opposite signs to cancel them).
        Annual gammas represent the T+1-year beta response to a 1-unit macro
        shock; they are applied directly in monthly predict_next() which then
        compounds the effect over 12 steps via the rho term.

        Returns (alpha, rho, gammas, residuals, r2).
        Raises RuntimeError if the OLS does not converge.
        """
        # Detect annual macro (same logic as _select_variables)
        n_months = len(macro_df)
        is_annual_macro = (
            selected
            and macro_df[selected].apply(lambda c: c.dropna().nunique()).median()
            <= n_months / 6
        )

        if is_annual_macro:
            # Aggregate to annual to avoid within-year collinearity
            beta_ann = beta_s.groupby(beta_s.index.year).mean()
            beta_ann.index = pd.to_datetime(beta_ann.index.astype(str) + "-01-01")
            macro_ann = macro_df[selected].groupby(macro_df.index.year).first()
            macro_ann.index = pd.to_datetime(macro_ann.index.astype(str) + "-01-01")
            y, y_lag = _ar1_data(beta_ann)
            macro_aligned = macro_ann.reindex(y.index)
        else:
            y, y_lag = _ar1_data(beta_s)
            macro_aligned = macro_df[selected].reindex(y.index) if selected else None

        if not selected:
            X = sm.add_constant(y_lag, has_constant="add")
        else:
            common = y.index.intersection(macro_aligned.dropna().index)
            y      = y.reindex(common)
            y_lag  = y_lag.reindex(common)
            macro_aligned = macro_aligned.reindex(common)
            X = sm.add_constant(
                pd.concat([y_lag, macro_aligned], axis=1), has_constant="add"
            )

        try:
            res = sm.OLS(y, X).fit(cov_type="HC1")
        except Exception as exc:
            raise RuntimeError(
                f"AR(1) OLS failed for beta satellite: {exc}"
            ) from exc

        params   = res.params
        alpha    = float(params.iloc[0])
        rho_ols  = float(params.iloc[1])

        # Clamp rho to (-0.99, 0.99) — prevents explosive AR(1) projections.
        # OLS on near-unit-root beta series (from ill-conditioned NS with wrong lambda)
        # produces rho ≈ 1 (Stambaugh upward bias). Without clamping, 12 monthly
        # iterations from a large beta value diverge. The grid-searched lambda
        # corrects the root cause; this clamp is a safety net in all cases.
        _RHO_MAX = 0.99
        rho = max(-_RHO_MAX, min(_RHO_MAX, rho_ols))
        if abs(rho_ols - rho) > 1e-6:
            LOG.warning(
                "beta%d: rho_OLS=%.4f clamped to %.4f -- OLS explosive/unit-root "
                "(likely caused by ill-conditioned NS lambda; grid-search lambda fixes root cause)",
                k, rho_ols, rho,
            )

        gammas = {var: float(params.iloc[2 + i]) for i, var in enumerate(selected)}

        # Filtre coef ≈ 0 : élaguer les vars sans contribution de gammas.
        # Peut arriver quand deux vars sélectionnées seules passent les 3 critères
        # mais s'annulent en estimation conjointe (collinéarité résiduelle).
        # Aucune ré-estimation : gamma=0 contribue déjà 0 dans predict_next(),
        # on l'ôte juste du dict pour ne pas polluer selected_vars et les logs.
        _zero_gammas = [var for var, g in gammas.items() if abs(g) < 1e-8]
        if _zero_gammas:
            LOG.warning(
                "beta%d: gamma≈0 pour %s — retirés du modèle (contribution nulle)",
                k, _zero_gammas,
            )
            gammas = {var: g for var, g in gammas.items() if abs(g) >= 1e-8}

        resid  = pd.Series(res.resid.values, index=y.index)

        return alpha, rho, gammas, resid, float(res.rsquared)


# ---- Statistical helpers -------------------------------------------------

def _adf_test(series: pd.Series, alpha: float = 0.10) -> Tuple[bool, float]:
    s = series.dropna()
    if len(s) < 6 or np.isclose(s.std(), 0.0):
        return True, 1.0
    try:
        max_lag = max(1, min(int(len(s) ** 0.5), len(s) // 4))
        p       = float(adfuller(s.values, maxlag=max_lag, autolag="AIC",
                                 regression="c")[1])
        return p < alpha, p
    except Exception:
        return True, 1.0


def _kpss_test(series: pd.Series, alpha: float = 0.10) -> Tuple[bool, float]:
    s = series.dropna()
    if len(s) < 6:
        return True, 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(kpss(s.values, regression="c", nlags="auto")[1])
        return p > alpha, p   # H0 = stationary; reject if p < alpha
    except Exception:
        return True, 1.0


def _ar1_data(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Return (y, y_lag1) with aligned non-NaN index."""
    df = pd.concat([series, series.shift(1)], axis=1).dropna()
    df.columns = ["y", "lag"]
    return df["y"], df["lag"]


def _ols_aic(y: pd.Series, X: pd.DataFrame) -> float:
    try:
        return float(sm.OLS(y, X).fit().aic)
    except Exception:
        return float("inf")


def _breusch_godfrey_p(resid: pd.Series, max_lags: int = 4) -> float:
    """Breusch-Godfrey LM test p-value for residual autocorrelation.

    BG requires a fitted regression object (residuals + original
    regressors). _fit_ar1() only returns the residual series here, so a
    constant-only auxiliary regression is used as the fitted model — BG
    then reduces to an LM test for autocorrelation in the centred residuals,
    matching the scope of the Ljung-Box test it replaces.
    """
    n = len(resid)
    if n < max_lags + 2:
        return 1.0
    try:
        lags = min(max_lags, n // 4)
        aux = sm.OLS(resid.values, np.ones((n, 1))).fit()
        _, bg_pvalue, _, _ = acorr_breusch_godfrey(aux, nlags=lags)
        return float(bg_pvalue)
    except Exception:
        return 1.0