"""
wrapper.py -- Risque de Marche (Market Risk Module)
=====================================================
MarketRiskModule: full pipeline orchestrator for sovereign interest rate risk.

Pipeline
--------
 1. Resolve Excel path (param override or YieldCurveFileDetector)
 2. YieldCurveLoader  -> yield_df (monthly) + macro_df (monthly) + portfolio_meta
 3. NelsonSiegelFitter -> betas_df {beta1, beta2, beta3} (monthly)
 4. IRSatellite       -> calibrated AR(1)+macro for each beta_k
 5. SyntheticPortfolioBuilder -> Portfolio (duration, BPV computed here)
 6. NS baseline curve at last date T
 7. FHSSampler        -> MarketRiskMeasure (sVaR, sES) -- scenario-independent
 8. For each scenario (baseline / adverse / severe):
      StressProjector       -> stressed betas
      NelsonSiegelReconstructor -> Dy(tau)
      PortfolioRepricer     -> DP (EGP millions)
 9. PlatformResult per scenario

Interface
---------
MarketRiskModule(upload_paths, params, scenario).run() -> Dict[str, PlatformResult]

params keys (all optional):
    country        : ISO2/ISO3 (default "EG")
    excel_path     : override file detection
    tbills_egp_bn  : manual T-Bill exposure (overrides Portfolio_CIB sheet)
    tbonds_egp_bn  : manual T-Bond exposure (overrides Portfolio_CIB sheet)
    tbill_weights  : dict {days: weight}
    tbond_weights  : dict {years: weight}
    ns_lambda      : Nelson-Siegel lambda (default 0.0609)
    n_simulations  : FHS bootstrap size (default 10 000)
    horizon_months : stress projection horizon (default 12)
    cache_dir      : macro API cache directory (default "data_cache")
    pvalue_threshold: IRSatellite significance level (default 0.05)

References
----------
Nelson & Siegel (1987), Journal of Business 60(4):473-489.
Diebold & Li (2006), Journal of Econometrics 130(2):337-364.
Barone-Adesi, Giannopoulos & Vosper (1999), J. Futures Markets 19(5):583-602.
BCBS (2009). Revisions to the Basel II market risk framework (Basel II.5).
BCBS (2019). Minimum capital requirements for market risk (FRTB).
Artzner, Delbaen, Eber & Heath (1999). Mathematical Finance, 9(3):203-228.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..base import PlatformModule, PlatformResult
from .yield_curve_detector import YieldCurveFileDetector
from .yield_curve_loader import YieldCurveLoader
from .nelson_siegel import NelsonSiegelFitter, NelsonSiegelReconstructor
from .ir_satellite import IRSatellite
from .stress_projector import StressProjector
from .portfolio import SyntheticPortfolioBuilder, PortfolioRepricer
from .fhs_sampler import FHSSampler, MarketRiskMeasure
from .market_validator import MarketPostEstimationRunner

LOG = logging.getLogger("market.wrapper")

_SCENARIO_LEVELS = ("baseline", "adverse", "severe")

# Default portfolio weights (overridable via params)
_DEFAULT_TBILL_WEIGHTS: Dict[int, float] = {91: 0.34, 182: 0.15, 273: 0.08, 364: 0.43}
_DEFAULT_TBOND_WEIGHTS: Dict[int, float] = {3: 0.9835, 5: 0.0165}

# Country-scoped Nelson-Siegel lambda defaults (fallback only -- NEVER applied
# if the caller passes params["ns_lambda"] explicitly; NEVER applied to a
# country not listed here, i.e. this is not a global default for the module).
#
# EG: on the current uploads/YieldCurve.xlsx (2010-01 -> 2025-12, 192 monthly
# obs, T91j..T10Y), NelsonSiegelFitter.search_lambda()'s unconstrained Ridge-MSE
# objective is non-convex over lambda: it has an interior local minimum near
# lambda=0.0108 (tau*~92.6y), but that point is excluded by design because
# tau* falls far outside the observed 91d-10Y maturity range and is not
# economically interpretable (see search_lambda()'s own lambda_min=1/tau_max
# floor). Within the allowed grid [0.10, 3.0], the search settles at the
# floor lambda=0.10 (tau*=10Y) -- a boundary optimum, not an interior one.
# lambda=0.8968 (tau*=1.115Y) is a judgmental override, NOT the statistical
# optimum: it does not minimize Ridge MSE (2.754 vs 2.295 at lambda=0.10) nor
# the B3.M1 fit RMSE (mean 1.3710pp vs 1.3362pp at lambda=0.10) -- it trades
# long-end accuracy (T3Y/T5Y/T10Y RMSE worsens) for materially better fit on
# the short end (T91j RMSE 0.80 vs 1.28pp, T182j 0.71 vs 1.03pp), which
# dominates this portfolio's duration/BPV (T-Bills = 4 of 6 instruments).
# Documented trade-off choice, not a claim of a superior MSE/RMSE outcome.
_COUNTRY_NS_LAMBDA_DEFAULT: Dict[str, float] = {
    "EG": 0.8968,
}

_MISSING_PORTFOLIO_MSG = (
    "Expositions portefeuille manquantes (tbills_egp_bn / tbonds_egp_bn). "
    "Veuillez uploader le fichier Excel de la courbe des taux avec une feuille "
    "'Portfolio_CIB' contenant les lignes 'tbills_egp_bn' et 'tbonds_egp_bn', "
    "ou fournir ces valeurs via les paramètres du module."
)


class MarketModuleWrapper(PlatformModule):
    """Market risk module -- sovereign yield-curve stress testing."""

    module_id     = "market"
    module_label  = "Risque de Marche"
    is_placeholder = False

    # ---- public API --------------------------------------------------------

    def run(self) -> Dict[str, PlatformResult]:
        warns: List[str] = []
        try:
            return self._run_pipeline(warns)
        except Exception as exc:
            LOG.exception("MarketRiskModule pipeline failed: %s", exc)
            msg = f"Pipeline error: {exc}"
            return {lvl: self._error_result(lvl, msg) for lvl in _SCENARIO_LEVELS}

    def get_kpis(self, results: Dict) -> Dict[str, Any]:
        return {lvl: r.kpis for lvl, r in results.items()}

    # ---- pipeline ----------------------------------------------------------

    def _run_pipeline(self, warns: List[str]) -> Dict[str, PlatformResult]:
        p = self.params or {}

        # 1. Resolve Excel path
        excel_path = self._resolve_excel(p)

        # 2. Load yield curve + macro (monthly aligned)
        loader = YieldCurveLoader(
            excel_path     = excel_path,
            country        = p.get("country", "EG"),
            cache_dir      = p.get("cache_dir", "data_cache"),
            cache_ttl_days = int(p.get("cache_ttl_days", 30)),
        )
        data           = loader.load()
        yield_df       = data.yield_df
        macro_df       = data.macro_df
        portfolio_meta = data.portfolio_meta

        # 2.5 Pre-tests macro (pertinence + qualité — Blocs 0-4, Bloc 5 sauté)
        # 2.5 Pre-tests macro (pertinence + qualité — Blocs 0-4, Bloc 5 sauté)
        _pt_res_mkt = None
        try:
            from ..core.macro_pre_tests import run_macro_pre_tests, PreTestConfig
            if not macro_df.empty:
                _pt_cfg = PreTestConfig(hard_filter_irrelevant=True, flag_unknown=True)
                _pt_res_mkt = run_macro_pre_tests(
                    macro_df=macro_df,
                    target_series=None,
                    config=_pt_cfg,
                    module='market',
                )
                LOG.info(
                    'Macro pre-tests marché : %d/%d vars acceptées, %d rejetées '
                    '(%d hors périmètre stress-test)',
                    _pt_res_mkt.report.get('n_accepted', 0),
                    _pt_res_mkt.report.get('n_total', 0),
                    _pt_res_mkt.report.get('n_rejected', 0),
                    _pt_res_mkt.report.get('n_irrelevant', 0),
                )
        except Exception as _pt_err:
            LOG.warning('Macro pre-tests marché échoués (non bloquant): %s', _pt_err)

        # Appliquer le filtre Bloc 0 : retirer les vars non pertinentes de macro_df
        # avant de les passer aux satellites IR. MARKET_MACRO_VARS sont toujours conservées.
        if _pt_res_mkt is not None and _pt_res_mkt.accepted:
            from ..market.yield_curve_loader import MARKET_MACRO_VARS as _MKT_CORE
            _mkt_always_keep = set(_MKT_CORE)
            _mkt_accepted    = set(_pt_res_mkt.accepted) | _mkt_always_keep
            _mkt_drop        = [c for c in macro_df.columns if c not in _mkt_accepted]
            if _mkt_drop:
                macro_df = macro_df.drop(columns=_mkt_drop)
                LOG.info(
                    "Filtre Bloc 0 marché : %d colonnes non pertinentes retirées "
                    "du pool macro (ex: %s).",
                    len(_mkt_drop), ", ".join(_mkt_drop[:5]),
                )

        # 3. Nelson-Siegel fitting — lambda via grid search unless overridden
        _ns_country = str(p.get("country", "EG")).strip().upper()
        if "ns_lambda" in p and p["ns_lambda"] is not None:
            ns_lambda     = float(p["ns_lambda"])
            ns_lambda_mse = float("nan")
            LOG.info("NS lambda from params: %.4f", ns_lambda)
        elif _ns_country in _COUNTRY_NS_LAMBDA_DEFAULT:
            ns_lambda     = _COUNTRY_NS_LAMBDA_DEFAULT[_ns_country]
            ns_lambda_mse = float("nan")
            LOG.info(
                "NS lambda from country-scoped default (%s): %.4f "
                "(judgmental override, not grid-search optimum -- see "
                "_COUNTRY_NS_LAMBDA_DEFAULT comment)",
                _ns_country, ns_lambda,
            )
        else:
            LOG.info("Running NS lambda grid-search on %d dates …", len(yield_df))
            ns_lambda, ns_lambda_mse = NelsonSiegelFitter.search_lambda(yield_df)
        fitter            = NelsonSiegelFitter(lambda_=ns_lambda, min_maturities=3)
        fitter.search_mse = ns_lambda_mse   # carried into _build_result via fitter
        betas_df          = fitter.fit(yield_df)


        # Bloc 5 — Granger causalité avec beta1 comme cible (level factor NS)
        # Complète les Blocs 0-4 déjà exécutés en STEP 2.5.
        # beta1 = facteur niveau de la courbe des taux → variable macro la plus sensible.
        try:
            from ..core.macro_pre_tests import run_macro_pre_tests, PreTestConfig
            _b5_target = betas_df["beta1"].dropna()
            if len(_b5_target) >= 10 and not macro_df.empty:
                _pt_cfg5 = PreTestConfig(
                    hard_filter_irrelevant=False,
                    flag_unknown=False,
                    min_obs=10,
                )
                _pt5_res = run_macro_pre_tests(
                    macro_df      = macro_df,
                    target_series = _b5_target,
                    config        = _pt_cfg5,
                    module        = 'market',
                )
                # Retirer du pool les vars dont Granger est rejeté (p >= 0.10)
                if _pt5_res.accepted:
                    from ..market.yield_curve_loader import MARKET_MACRO_VARS as _MKT_CORE5
                    _b5_keep = set(_pt5_res.accepted) | set(_MKT_CORE5)
                    _b5_drop = [c for c in macro_df.columns if c not in _b5_keep]
                    if _b5_drop:
                        macro_df = macro_df.drop(columns=_b5_drop)
                        LOG.info(
                            "Bloc 5 marché (Granger→beta1) : %d vars non causales "
                            "retirées du pool (ex: %s).",
                            len(_b5_drop), ", ".join(_b5_drop[:5]),
                        )
        except Exception as _b5_err:
            LOG.warning("Bloc 5 marché échoué (non bloquant): %s", _b5_err)

        n_valid = int(betas_df.dropna().shape[0])
        if n_valid < 12:
            raise RuntimeError(
                f"Only {n_valid} valid NS beta observations -- "
                "min 12 required for satellite calibration."
            )

        # 4. IR satellite models for beta1, beta2, beta3
        ir_sat     = IRSatellite(
            betas_df         = betas_df,
            macro_df         = macro_df,
            pvalue_threshold = float(p.get("pvalue_threshold", 0.10)),
        )
        satellites = ir_sat.fit_all()
        for sat in satellites.values():
            warns.extend(sat.validation_warnings)

        # 5. Synthetic portfolio — expositions obligatoires (params > feuille Excel)
        tbills_bn_raw = p.get("tbills_egp_bn") or portfolio_meta.get("tbills_egp_bn")
        tbonds_bn_raw = p.get("tbonds_egp_bn") or portfolio_meta.get("tbonds_egp_bn")
        if tbills_bn_raw is None or tbonds_bn_raw is None:
            raise ValueError(_MISSING_PORTFOLIO_MSG)
        tbills_bn = float(tbills_bn_raw)
        tbonds_bn = float(tbonds_bn_raw)
        builder   = SyntheticPortfolioBuilder(
            tbills_egp_bn = tbills_bn,
            tbonds_egp_bn = tbonds_bn,
            yield_df      = yield_df,
            tbill_weights = p.get("tbill_weights", _DEFAULT_TBILL_WEIGHTS),
            tbond_weights = p.get("tbond_weights", _DEFAULT_TBOND_WEIGHTS),
            country       = str(p.get("country", "EGY")),
            as_of_date    = "2023-12-31",
        )
        portfolio = builder.build()

        # 6. Baseline curve at last observed date (NS-reconstructed, avoids NaN)
        last_valid    = betas_df.dropna().iloc[-1]
        baseline_curve = fitter.reconstruct(
            float(last_valid["beta1"]),
            float(last_valid["beta2"]),
            float(last_valid["beta3"]),
        )
        reconstructor = NelsonSiegelReconstructor(
            fitter         = fitter,
            baseline_curve = baseline_curve,
        )

        # 7. FHS (sVaR + sES) -- unconditional, scenario-independent
        n_sim = int(p.get("n_simulations", 100_000))
        fhs   = FHSSampler(
            betas_df      = betas_df,
            reconstructor = reconstructor,
            portfolio     = portfolio,
            yield_curve_T = baseline_curve,
            n_simulations = n_sim,
        )
        risk_measure = fhs.run()
        warns.extend(risk_measure.warnings)

        # 8. Scenario projections — PATH B/C + projection macro baseline
        last_betas = betas_df.dropna().iloc[-1]
        horizon    = int(p.get("horizon_months", 12))
        cache_dir  = str(p.get("cache_dir", "data_cache"))

        # Normalise country → ISO2 (accepte "EG", "EGY", ou "Egypt")
        _COUNTRY_NAME_TO_ISO2 = {
            "egypt": "EG", "france": "FR", "germany": "DE",
            "united kingdom": "GB", "mena": "EG", "north africa": "MA",
            "europe": "DE", "gcc": "SA", "morocco": "MA", "tunisia": "TN",
            "nigeria": "NG", "south africa": "ZA", "senegal": "SN",
        }
        _country_raw = str(p.get("country_iso2") or p.get("country", "EG")).strip()
        country = _country_raw.upper()
        if len(country) > 3:
            country = _COUNTRY_NAME_TO_ISO2.get(_country_raw.lower(), country)

        # ── 8a. Fetch données macro annuelles (une seule fois) ───────────────
        _mac_annual: Optional[pd.DataFrame] = None
        try:
            from ..core.imf_weo_fetcher import fetch_credit_macro as _fetch_mac
            _crisis_name_pre = (self.scenario or {}).get("crisis_name", "")
            from ..core.historical_scenarios import CRISIS_CATALOG
            _t_pre_start = (CRISIS_CATALOG[_crisis_name_pre].t_pre_start - 2
                            if _crisis_name_pre and _crisis_name_pre in CRISIS_CATALOG
                            else 2000)
            _mac_full_ann, _ = _fetch_mac(
                country    = country,
                cache_dir  = cache_dir,
                start_year = _t_pre_start,
            )
            if _mac_full_ann is not None and not _mac_full_ann.empty:
                _mac_annual = (_mac_full_ann.set_index("year")
                               if "year" in _mac_full_ann.columns
                               else _mac_full_ann)
        except Exception as _fetch_err:
            LOG.warning("Marché: fetch macro annuelle échoué (%s) — fallback last_macro", _fetch_err)

        # ── 8b. Projection macro baseline (AR / VAR / VECM) ─────────────────
        # Remplace la projection plate (last observed) par une trajectoire modélisée.
        from ..market.yield_curve_loader import MARKET_MACRO_VARS as _MKT_VARS
        last_macro_obs = (macro_df.dropna(how="all").iloc[-1]
                          if not macro_df.empty else pd.Series(dtype=float))
        last_macro_proj = last_macro_obs.copy()   # fallback = plate

        if _mac_annual is not None:
            try:
                from ..core.macro_projector import project_macro as _proj_mac_mkt
                _horizon_year = int(pd.Timestamp.now().year) + max(1, round(horizon / 12))
                _mkt_proj_vars = [v for v in _MKT_VARS
                                  if v in _mac_annual.columns
                                  and not _mac_annual[v].dropna().empty]
                if _mkt_proj_vars:
                    _mkt_proj_res = _proj_mac_mkt(
                        historical    = _mac_annual,
                        variables     = _mkt_proj_vars,
                        horizon_years = [_horizon_year],
                    )
                    _proj_row = _mkt_proj_res.projected.loc[_horizon_year]
                    for _v in _mkt_proj_vars:
                        if _v in last_macro_proj.index:
                            last_macro_proj[_v] = float(_proj_row[_v])
                    LOG.info(
                        "macro_projector marché : baseline projeté T+%d an(s) via %s "
                        "— policy_rate=%.2f%% gdp=%.2f%% cpi=%.2f%%",
                        max(1, round(horizon / 12)), _mkt_proj_res.model_used,
                        last_macro_proj.get("policy_rate", float("nan")),
                        last_macro_proj.get("real_gdp_growth", float("nan")),
                        last_macro_proj.get("cpi_inflation", float("nan")),
                    )
            except Exception as _mkt_proj_err:
                LOG.warning(
                    "macro_projector marché échoué (non bloquant) — "
                    "fallback last_macro observé : %s", _mkt_proj_err
                )

        # ── 8c. Chocs de crise PATH B/C ──────────────────────────────────────
        crisis_shocks_1x = _compute_crisis_market_shocks(
            scenario     = self.scenario,
            country      = country,
            cache_dir    = cache_dir,
            macro_annual = _mac_annual,   # données pré-fetchées → pas de double-fetch
        )

        # ── 8d. Boucle niveaux (baseline / adverse / severe) ─────────────────
        results: Dict[str, PlatformResult] = {}
        _val_delta_y:      Dict[str, pd.Series] = {}
        _val_losses:       Dict[str, float]     = {}
        _val_stressed_crv: Dict[str, pd.Series] = {}
        for lvl in _SCENARIO_LEVELS:
            if lvl == "baseline":
                shocks: Dict[str, float] = {}
                _lm = last_macro_proj          # baseline projeté (AR/VAR/VECM)
            elif lvl == "adverse":
                shocks = {k: v * 0.5 for k, v in crisis_shocks_1x.items()}
                _lm = last_macro_proj
            else:  # severe
                shocks = crisis_shocks_1x.copy()
                _lm = last_macro_proj

            LOG.info(
                "PATH B/C marché [%s] shocks: rate_bps=%+.1f  gdp_delta=%+.3f  "
                "inflation_delta=%+.3f",
                lvl,
                shocks.get("rate_bps", 0.0),
                shocks.get("gdp_delta", 0.0),
                shocks.get("inflation_delta", 0.0),
            )

            projector = StressProjector(
                satellites  = satellites,
                last_betas  = last_betas,
                last_macro  = _lm,
            )
            beta_path  = projector.project_path(shocks, horizon_months=horizon)
            stressed_b = {f"beta{k}": beta_path[f"beta{k}"][-1] for k in (1, 2, 3)}

            delta_y      = reconstructor.delta_curve(
                stressed_b["beta1"],
                stressed_b["beta2"],
                stressed_b["beta3"],
            )
            delta_y_pf = delta_y.drop("T10Y", errors="ignore")

            repricer   = PortfolioRepricer(
                portfolio     = portfolio,
                yield_curve_T = baseline_curve,
                delta_y       = delta_y_pf,
            )
            dp_total, breakdown = repricer.reprice(method="auto")
            loss                = -dp_total

            # Collect for post-estimation validation (step 9)
            _val_delta_y[lvl]      = delta_y
            _val_losses[lvl]       = loss
            _val_stressed_crv[lvl] = pd.Series({
                m: round(
                    float(baseline_curve.get(m, 0.0))
                    + float(delta_y.get(m, 0.0) if m in delta_y.index else 0.0),
                    4,
                )
                for m in baseline_curve.index
            })

            results[lvl] = self._build_result(
                scenario_id    = lvl,
                loss           = loss,
                dp_total       = dp_total,
                stressed_b     = stressed_b,
                beta_path      = beta_path,
                delta_y        = delta_y,
                breakdown      = breakdown,
                portfolio      = portfolio,
                risk_measure   = risk_measure,
                betas_df       = betas_df,
                warns          = list(warns),
                baseline_curve = baseline_curve,
                fitter         = fitter,
                yield_df       = yield_df,
                satellites     = satellites,
            )

        # ── 9. Post-estimation validation (non-blocking) ─────────────────────
        try:
            _mkt_val = MarketPostEstimationRunner(
                satellites   = satellites,
                betas_df     = betas_df,
                macro_df     = macro_df,
                fitter       = fitter,
                yield_df     = yield_df,
                portfolio    = portfolio,
                risk_measure = risk_measure,
            )
            _val_report = _mkt_val.run(
                delta_y_per_scenario = _val_delta_y,
                losses_per_scenario  = _val_losses,
                stressed_curves      = _val_stressed_crv,
            )
            _val_dict = _val_report.to_dict()
            results["baseline"].charts_data["validation_report"] = _val_dict
            results["baseline"].kpis["validation_verdict"] = (
                _val_report.overall_verdict
            )
            if _val_report.overall_verdict != "VALIDATED":
                _n_fail = sum(
                    1 for r in (
                        _val_report.b3_results + _val_report.backtest_results
                    )
                    if r.status == "FAIL"
                )
                results["baseline"].warnings.append(
                    f"Post-estimation validation: {_val_report.overall_verdict}"
                    f" ({_n_fail} FAIL(s))"
                )
            LOG.info(
                "Post-estimation validation marché: %s",
                _val_report.overall_verdict,
            )
        except Exception as _val_exc:
            LOG.warning(
                "Post-estimation validation marché échouée (non bloquant): %s",
                _val_exc,
            )

        return results

    # ---- result builder ----------------------------------------------------

    def _build_result(
        self,
        scenario_id:    str,
        loss:           float,
        dp_total:       float,
        stressed_b:     Dict[str, float],
        delta_y:        pd.Series,
        breakdown:      Dict[str, float],
        portfolio,
        risk_measure:   MarketRiskMeasure,
        betas_df:       pd.DataFrame,
        warns:          List[str],
        baseline_curve: "pd.Series | None" = None,
        fitter=None,
        yield_df:       "pd.DataFrame | None" = None,
        beta_path:      "Dict[str, list] | None" = None,
        satellites:     "Dict | None" = None,
    ) -> PlatformResult:

        betas_clean = betas_df.dropna()
        ts = betas_clean[["beta1", "beta2", "beta3"]].copy()

        kpis = {
            "svar_99":             round(risk_measure.svar_99, 4),
            "ses_975":             round(risk_measure.ses_975, 4),
            "portfolio_duration":  round(portfolio.portfolio_duration, 4),
            "portfolio_bpv_m_egp": round(portfolio.portfolio_bpv_egp_m, 4),
            "total_notional_bn":   round(portfolio.total_notional_egp_bn, 4),
            "delta_p_m_egp":       round(dp_total, 4),
            "loss_m_egp":          round(loss, 4),
            "beta1_stressed":      round(stressed_b.get("beta1", float("nan")), 4),
            "beta2_stressed":      round(stressed_b.get("beta2", float("nan")), 4),
            "beta3_stressed":      round(stressed_b.get("beta3", float("nan")), 4),
            "delta_y_91j_pp":      round(float(delta_y.get("T91j",  0.0)), 4),
            "delta_y_364j_pp":     round(float(delta_y.get("T364j", 0.0)), 4),
            "delta_y_3Y_pp":       round(float(delta_y.get("T3Y",   0.0)), 4),
            "delta_y_5Y_pp":       round(float(delta_y.get("T5Y",   0.0)), 4),
            "ValidationGate":      risk_measure.validation_gate_verdict,
            "stress_window":
                f"{risk_measure.stress_window_start.date()} -> "
                f"{risk_measure.stress_window_end.date()}",
            "n_simulations":       risk_measure.n_simulations,
            "ns_lambda":           round(fitter.lambda_, 4) if fitter else float("nan"),
            "ns_lambda_mse":       round(getattr(fitter, "search_mse", float("nan")), 6) if fitter else float("nan"),
        }

        # ── Charts data: base fields ────────────────────────────────────────
        charts_data: Dict[str, Any] = {
            "beta_history":          _to_serialisable(betas_clean),
            "delta_y":               delta_y.to_dict(),
            "instrument_breakdown":  breakdown,
            "loss_distribution_pct": _percentiles(risk_measure.loss_distribution),
        }

        # ── Charts data: enriched fields (require baseline_curve / fitter) ──
        if baseline_curve is not None:
            # Graph 1 — yield curve observed (NS-reconstructed at last date T)
            charts_data["yield_curve_observed"] = {
                m: round(float(baseline_curve.get(m, 0)), 4)
                for m in baseline_curve.index
            }
            # Graph 1 — stressed curve for this scenario (T+12 endpoint)
            charts_data["yield_curve_stressed"] = {
                m: round(float(baseline_curve.get(m, 0)) + float(delta_y.get(m, 0) if m in delta_y.index else 0), 4)
                for m in baseline_curve.index
            }

            # Graph 1 — monthly trajectory T+1 → T+12 reconstructed from beta_path
            if (
                beta_path is not None
                and fitter is not None
                and yield_df is not None
                and not yield_df.empty
            ):
                _last_dt   = yield_df.index[-1]
                _n_steps   = len(beta_path["beta1"])
                _show_mats = ["T91j", "T182j", "T273j", "T364j", "T3Y", "T5Y", "T10Y"]
                _monthly_dates = [
                    str((_last_dt + pd.DateOffset(months=m + 1)).date())
                    for m in range(_n_steps)
                ]
                _monthly_path: Dict[str, Any] = {"dates": _monthly_dates}
                for _mat in _show_mats:
                    if _mat not in baseline_curve.index:
                        continue
                    _yields_m = []
                    for _i in range(_n_steps):
                        try:
                            _yc = fitter.reconstruct(
                                beta_path["beta1"][_i],
                                beta_path["beta2"][_i],
                                beta_path["beta3"][_i],
                            )
                            _yields_m.append(round(float(_yc.get(_mat, 0)), 4))
                        except Exception:
                            _yields_m.append(None)
                    _monthly_path[_mat] = _yields_m
                charts_data["yield_curve_monthly_path"] = _monthly_path

        # Beta monthly path T+1 → T+12 (for beta history chart)
        if beta_path is not None and yield_df is not None and not yield_df.empty:
            _last_dt2 = yield_df.index[-1]
            _n2 = len(beta_path["beta1"])
            charts_data["beta_monthly_path"] = {
                "dates": [
                    str((_last_dt2 + pd.DateOffset(months=m + 1)).date())
                    for m in range(_n2)
                ],
                "beta1": [round(float(v), 4) for v in beta_path["beta1"]],
                "beta2": [round(float(v), 4) for v in beta_path["beta2"]],
                "beta3": [round(float(v), 4) for v in beta_path["beta3"]],
            }

        if yield_df is not None and not yield_df.empty:
            # Graph 7 — actual observed yields at last available date
            last_row = yield_df.iloc[-1].dropna()
            charts_data["ns_fit_observed"] = {
                m: round(float(last_row[m]), 4) for m in last_row.index
            }

            # Graph 1 — full historical yield curve fan (only stored once in baseline)
            if scenario_id == "baseline":
                _MAT_COLS = ["T91j", "T182j", "T273j", "T364j", "T3Y", "T5Y", "T10Y"]
                _present  = [m for m in _MAT_COLS if m in yield_df.columns]
                _hist_vals: list = []
                for _, _row in yield_df[_present].iterrows():
                    _hist_vals.append([
                        round(float(_row[m]), 4) if pd.notna(_row.get(m)) else None
                        for m in _present
                    ])
                charts_data["yield_curve_history"] = {
                    "dates":      [str(d.date()) for d in yield_df.index],
                    "maturities": _present,
                    "values":     _hist_vals,
                }
                charts_data["yield_curve_last_date"] = str(yield_df.index[-1].date())

        # Always expose last date (needed by stress_window chart regardless of scenario)
        if yield_df is not None and not yield_df.empty:
            charts_data.setdefault(
                "yield_curve_last_date", str(yield_df.index[-1].date())
            )

        if fitter is not None and not betas_clean.empty:
            # Graph 7 — NS smooth curve at 50 interpolated points (91j → 10Y)
            last_b = betas_clean.iloc[-1]
            tau_fine = np.linspace(91 / 365, 10.0, 50)
            mats_fine = {f"t{i}": float(t) for i, t in enumerate(tau_fine)}
            try:
                ns_vals = fitter.reconstruct(
                    float(last_b["beta1"]), float(last_b["beta2"]), float(last_b["beta3"]),
                    maturities=mats_fine,
                )
                charts_data["ns_fit_curve"] = {
                    "x": [round(t, 4) for t in tau_fine.tolist()],
                    "y": [round(float(v), 4) for v in ns_vals.values],
                }
            except Exception as _e:
                LOG.debug("ns_fit_curve computation skipped: %s", _e)

        # Graph 6 — rolling 12-month beta-variance (stress window identification)
        delta_betas = betas_clean.diff().dropna()
        db_2005 = delta_betas[delta_betas.index >= "2005-01-01"]
        W = 12
        if len(db_2005) >= W:
            stress_vol: Dict[str, float] = {}
            for i in range(len(db_2005) - W + 1):
                window   = db_2005.iloc[i: i + W]
                date_key = str(db_2005.index[i + W - 1].date())
                stress_vol[date_key] = round(float(window.var(ddof=1).sum()), 8)
            charts_data["stress_window_volatility"] = stress_vol
            charts_data["stress_window_start"] = str(risk_measure.stress_window_start.date())
            charts_data["stress_window_end"]   = str(risk_measure.stress_window_end.date())

        # ── Leaderboard satellites IR (beta1 / beta2 / beta3) ──────────────
        if satellites is not None:
            charts_data["satellite_leaderboards"] = {
                f"beta{k}": sat.leaderboard
                for k, sat in satellites.items()
                if sat.leaderboard
            }

        return PlatformResult(
            module_id    = self.module_id,
            module_label = self.module_label,
            scenario_id  = scenario_id,
            total_loss   = round(loss, 4),
            kpis         = kpis,
            time_series  = ts,
            charts_data  = charts_data,
            metadata     = {
                "stress_window_start":    str(risk_measure.stress_window_start.date()),
                "stress_window_end":      str(risk_measure.stress_window_end.date()),
                "ValidationGate_verdict": risk_measure.validation_gate_verdict,
            },
            warnings     = warns,
        )

    # ---- helpers -----------------------------------------------------------

    def _resolve_excel(self, p: Dict) -> Path:
        """Return Excel path: explicit override → any uploaded Excel → auto-detection."""
        up = self.upload_paths or {}

        # 1. Explicit override from upload_paths["excel_path"] or params
        explicit = up.get("excel_path") or p.get("excel_path")
        if explicit:
            path = Path(explicit)
            if not path.exists():
                raise FileNotFoundError(f"Specified excel_path does not exist: {path}")
            LOG.info("Using specified excel_path: %s", path)
            return path

        # 2. Fallback: any uploaded .xlsx/.xls in upload_paths (any role)
        #    Validate via content inspection before returning.
        from .yield_curve_detector import _validate_by_content
        for role, fpath in up.items():
            if role in ("upload_dir",) or not isinstance(fpath, str):
                continue
            candidate = Path(fpath)
            if candidate.suffix.lower() not in (".xlsx", ".xls") or not candidate.exists():
                continue
            if _validate_by_content(candidate):
                LOG.info("Yield-curve file found via upload_paths (role=%s): %s", role, candidate.name)
                return candidate
            LOG.debug("Skipping uploaded file (role=%s, no yield-curve sheet): %s", role, candidate.name)

        # 3. Last resort: scan upload_dir with full detector
        upload_dir = up.get("upload_dir", "uploads")
        detector   = YieldCurveFileDetector(upload_dir)
        return detector.detect()

    @staticmethod
    def _error_result(scenario_id: str, message: str) -> PlatformResult:
        return PlatformResult(
            module_id    = "market",
            module_label = "Risque de Marche",
            scenario_id  = scenario_id,
            total_loss   = 0.0,
            kpis         = {"status": "error", "message": message},
            time_series  = pd.DataFrame(columns=["date", "beta1", "beta2", "beta3"]),
            charts_data  = {},
            metadata     = {"error": message},
            warnings     = [message],
        )



def _compute_crisis_market_shocks(
    scenario:     dict,
    country:      str,
    cache_dir:    str,
    macro_annual: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Calcule les chocs marché (rate_bps, gdp_delta, inflation_delta) depuis
    les données API réelles de la crise — PATH B/C exclusif, aucune valeur hardcodée.

    Formule (IMF standard) :
        delta_1x = valeur_pic_crise - moyenne_pre_crise

    Scaling :
        adverse = 0.5 * delta_1x
        severe  = 1.0 * delta_1x

    Args
    ----
    scenario     : dict scenario (self.scenario) avec clé 'crisis_name'
    country      : ISO2 du pays (ex. 'EG')
    cache_dir    : répertoire cache API
    macro_annual : DataFrame annuel pré-fetchée (optionnel, évite double-fetch)
                   Si None, fetch via fetch_credit_macro().

    Returns
    -------
    dict : {rate_bps, gdp_delta, inflation_delta} pour le niveau severe (1x).
           Adverse = 0.5x appliqué par l'appelant.

    Raises
    ------
    RuntimeError si crisis_name absent ou absent du CRISIS_CATALOG.
    """
    from ..core.historical_scenarios import get_crisis, compute_historical_shocks, CRISIS_CATALOG

    crisis_name = (scenario or {}).get("crisis_name", "") or ""
    if not crisis_name:
        raise RuntimeError(
            "Module marché — PATH B/C : 'crisis_name' absent du scénario. "
            "Seuls les scénarios historiques avec une entrée dans CRISIS_CATALOG "
            "sont supportés (valeurs hardcodées supprimées). "
            f"Scénarios disponibles : {list(CRISIS_CATALOG.keys())}"
        )

    crisis = get_crisis(crisis_name)

    # Utiliser les données pré-fetchées si disponibles; sinon fetcher
    if macro_annual is not None:
        mac_idx = macro_annual
    else:
        from ..core.imf_weo_fetcher import fetch_credit_macro
        macro_full, _ = fetch_credit_macro(
            country   = country,
            cache_dir = cache_dir,
            start_year = crisis.t_pre_start - 2,
        )
        if macro_full is None or macro_full.empty:
            raise RuntimeError(
                f"Module marché — PATH B/C : données macro indisponibles pour '{country}'. "
                "Vérifiez la connexion API IMF/WB."
            )
        mac_idx = macro_full.set_index("year") if "year" in macro_full.columns else macro_full

    raw_deltas = compute_historical_shocks(mac_idx, crisis)

    if not raw_deltas:
        raise RuntimeError(
            f"Module marché — PATH B/C : aucun delta calculé pour la crise "
            f"'{crisis_name}' sur le pays '{country}'. "
            "Vérifiez que les données API couvrent la fenêtre de la crise "
            f"[{crisis.t_pre_start}–{crisis.t_peak}]."
        )

    shocks_1x: dict = {}

    # Taux directeur : delta en pp → convertir en basis points pour StressProjector
    for col in ("policy_rate", "lending_rate"):
        if col in raw_deltas:
            shocks_1x["rate_bps"] = raw_deltas[col] * 100   # pp → bps
            LOG.info("PATH B/C marché: rate_bps = %+.1f bps (source col='%s')",
                     shocks_1x["rate_bps"], col)
            break

    # Inflation CPI : delta en % (même unité que cpi_inflation dans macro_df)
    if "cpi_inflation" in raw_deltas:
        shocks_1x["inflation_delta"] = raw_deltas["cpi_inflation"]
        LOG.info("PATH B/C marché: inflation_delta = %+.3f pp", shocks_1x["inflation_delta"])

    # PIB réel : delta en % (même unité que real_gdp_growth dans macro_df)
    if "real_gdp_growth" in raw_deltas:
        shocks_1x["gdp_delta"] = raw_deltas["real_gdp_growth"]
        LOG.info("PATH B/C marché: gdp_delta = %+.3f pp", shocks_1x["gdp_delta"])

    if not shocks_1x:
        raise RuntimeError(
            f"Module marché — PATH B/C : aucune variable macro reconnue dans les "
            f"deltas calculés pour '{crisis_name}'. "
            f"Colonnes disponibles : {list(raw_deltas.keys())}. "
            "Attendu : policy_rate / cpi_inflation / real_gdp_growth."
        )

    LOG.info(
        "PATH B/C marché [%s | %s]: %d chocs calculés depuis données API — "
        "rate_bps=%+.1f | gdp_delta=%+.3f | inflation_delta=%+.3f",
        crisis_name, country, len(shocks_1x),
        shocks_1x.get("rate_bps", 0.0),
        shocks_1x.get("gdp_delta", 0.0),
        shocks_1x.get("inflation_delta", 0.0),
    )
    return shocks_1x

def _percentiles(arr: np.ndarray) -> Dict[str, float]:
    if len(arr) == 0:
        return {}
    ps = [1, 5, 25, 50, 75, 95, 99]
    return {f"p{p}": round(float(np.percentile(arr, p)), 4) for p in ps}


def _to_serialisable(df: pd.DataFrame) -> Dict:
    """Convert DataFrame to a JSON-safe dict (Timestamp index -> str)."""
    d = df.copy()
    d.index = d.index.astype(str)
    return d.to_dict(orient="index")
