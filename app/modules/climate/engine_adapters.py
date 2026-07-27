"""
engine_adapters.py
==================
Thin adapter classes that wrap each existing risk-engine wrapper so that
ClimateOrchestrator can call them with a MacroDeltaVector.

  • CreditEngineAdapter / LiquidityEngineAdapter — injection directe NGFS :
    lisent les trajectoires PD/LGD/RWA (ngfs_credit_engine.compute_ngfs_pd_lgd)
    et LCR/NSFR (ngfs_liquidity_engine.compute_ngfs_lcr_nsfr) déjà calculées
    UNE FOIS par run climat (direct sur les niveaux NGFS réels, zéro rampe ou
    décroissance synthétique) et construisent le PlatformResult par simple
    lecture à l'année demandée. Aucun fallback silencieux : si la trajectoire
    NGFS n'est pas disponible pour le scénario/année demandé, l'erreur est
    levée telle quelle (visible dans ClimateStressResult.errors).

  • MarketEngineAdapter — MarketModuleWrapper only supports PATH B/C
    (crisis-based shocks via crisis_name); it has no PATH A.  This adapter
    therefore bypasses MarketModuleWrapper and drives StressProjector directly
    with MacroDeltaVector-derived shocks.  It pre-loads yield data and
    calibrates IR satellites once at construction time for efficiency.
    Already injects real per-year NGFS-derived NS beta deltas directly
    (delta_beta1/2/3, Layer 2) when the satellite factory is active.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .macro_delta import MacroDeltaVector

LOG = logging.getLogger("climate.engine_adapters")


def _resolve_alias(scenario_name: str, adverse_name: str, severe_name: str) -> str:
    """Résout l'alias baseline/adverse/severe depuis le nom de scénario NGFS réel."""
    if scenario_name == adverse_name:
        return "adverse"
    if scenario_name == severe_name:
        return "severe"
    if scenario_name == "baseline" or scenario_name == "Baseline":
        return "baseline"
    raise ValueError(
        f"Impossible de résoudre l'alias NGFS pour le scénario {scenario_name!r} "
        f"(adverse attendu={adverse_name!r}, severe attendu={severe_name!r})."
    )


def _nearest_year_index(years: List[int], target_year: int) -> int:
    """Index de target_year dans years (exact, sinon année la plus proche)."""
    if target_year in years:
        return years.index(target_year)
    return min(range(len(years)), key=lambda i: abs(years[i] - target_year))


# ─────────────────────────────────────────────────────────────────────────────
# CreditEngineAdapter
# ─────────────────────────────────────────────────────────────────────────────

class CreditEngineAdapter:
    """
    Construit le PlatformResult crédit par lecture directe de la trajectoire
    NGFS déjà projetée (ngfs_credit_engine.compute_ngfs_pd_lgd) : PD_PIT
    injecté depuis le niveau NGFS réel de l'année, LGD_PIT recalculé par le
    même modèle Frye-Jacobs (mêmes rho/k qu'à l'historique) à partir de ce PD
    stressé, RWA/CET1/CAR déjà stressés par CreditCapitalEngine.

    Aucun appel à CreditModuleWrapper.run() ni à aucune rampe synthétique.
    """

    def __init__(
        self,
        upload_paths: Dict[str, str],
        params: Dict[str, Any],
        base_scenario: Dict[str, Any],
        ngfs_credit: Optional[Dict[str, Any]] = None,
        adverse_name: str = "",
        severe_name: str = "",
    ) -> None:
        self._upload_paths = upload_paths
        self._params = params
        self._base_scenario = base_scenario
        self._ngfs_credit = ngfs_credit or {}
        self._adverse_name = adverse_name
        self._severe_name = severe_name

    def run_stressed(self, macro_delta: MacroDeltaVector) -> Any:
        """Lit PD/LGD/capital NGFS déjà projetés à l'année demandée ; lève une
        exception explicite (aucun fallback) si la donnée est indisponible."""
        from ..base import PlatformResult

        if self._ngfs_credit.get("error"):
            raise ValueError(
                f"CreditEngineAdapter: ngfs_credit en erreur — "
                f"{self._ngfs_credit['error']}"
            )
        if not self._ngfs_credit:
            raise ValueError(
                "CreditEngineAdapter: aucune projection NGFS crédit fournie "
                "(compute_ngfs_pd_lgd n'a pas produit de résultat)."
            )

        alias = _resolve_alias(
            macro_delta.scenario_name, self._adverse_name, self._severe_name
        )
        scenarios = self._ngfs_credit.get("scenarios") or {}
        scen = scenarios.get(alias)
        bl_scen = scenarios.get("baseline")
        if not scen or not scen.get("years"):
            raise ValueError(
                f"CreditEngineAdapter: pas de trajectoire NGFS crédit pour "
                f"l'alias {alias!r} (scénario={macro_delta.scenario_name!r})."
            )
        if not bl_scen or not bl_scen.get("years"):
            raise ValueError(
                "CreditEngineAdapter: pas de trajectoire NGFS crédit baseline "
                "— impossible de calculer le delta stressé."
            )

        years = [int(y) for y in scen["years"]]
        idx = _nearest_year_index(years, macro_delta.horizon_year)
        pd_stressed = float(scen["pd"][idx])
        lgd_stressed = float(scen["lgd"][idx])

        bl_years = [int(y) for y in bl_scen["years"]]
        bl_idx = _nearest_year_index(bl_years, macro_delta.horizon_year)
        pd_baseline = float(bl_scen["pd"][bl_idx])

        year_val = years[idx]
        el_rate = pd_stressed * lgd_stressed

        ts = pd.DataFrame({
            "year": [year_val],
            "pd":   [pd_stressed],
            "lgd":  [lgd_stressed],
        })

        return PlatformResult(
            module_id="credit",
            module_label="Risque de Crédit (Climate)",
            scenario_id=alias,
            total_loss=round(el_rate, 6),
            kpis={
                "avg_pd":  round(pd_baseline, 6),
                "peak_pd": round(pd_stressed, 6),
                "avg_lgd": round(lgd_stressed, 6),
            },
            time_series=ts,
            charts_data={
                "capital_trajectories": self._ngfs_credit.get(
                    "capital_trajectories", {}
                ),
            },
            metadata={
                "climate_scenario": macro_delta.scenario_name,
                "horizon_year": macro_delta.horizon_year,
                "injection": "ngfs_direct",
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# LiquidityEngineAdapter
# ─────────────────────────────────────────────────────────────────────────────

class LiquidityEngineAdapter:
    """
    Construit le PlatformResult liquidité par lecture directe de la
    trajectoire NGFS déjà projetée (ngfs_liquidity_engine.compute_ngfs_lcr_nsfr) :
    LCR/NSFR stressés injectés depuis le niveau NGFS réel de l'année.

    Aucun appel à LiquidityModuleWrapper.run() ni à aucune décroissance
    synthétique (ancien profil {T+1:0, T+2:1.0, T+3:0.7, T+4:0.3}).
    """

    def __init__(
        self,
        upload_paths: Dict[str, str],
        params: Dict[str, Any],
        base_scenario: Dict[str, Any],
        ngfs_liquidity: Optional[Dict[str, Any]] = None,
        adverse_name: str = "",
        severe_name: str = "",
    ) -> None:
        self._upload_paths = upload_paths
        self._params = params
        self._base_scenario = base_scenario
        self._ngfs_liq = ngfs_liquidity or {}
        self._adverse_name = adverse_name
        self._severe_name = severe_name

    def run_stressed(self, macro_delta: MacroDeltaVector) -> Any:
        """Lit LCR/NSFR NGFS déjà projetés à l'année demandée ; lève une
        exception explicite (aucun fallback) si la donnée est indisponible."""
        from ..base import PlatformResult

        if self._ngfs_liq.get("error"):
            raise ValueError(
                f"LiquidityEngineAdapter: ngfs_liquidity en erreur — "
                f"{self._ngfs_liq['error']}"
            )
        if not self._ngfs_liq:
            raise ValueError(
                "LiquidityEngineAdapter: aucune projection NGFS liquidité "
                "fournie (compute_ngfs_lcr_nsfr n'a pas produit de résultat)."
            )

        alias = _resolve_alias(
            macro_delta.scenario_name, self._adverse_name, self._severe_name
        )
        scenarios = self._ngfs_liq.get("scenarios") or {}
        scen = scenarios.get(alias)
        bl_scen = scenarios.get("baseline")
        if not scen or not scen.get("years"):
            raise ValueError(
                f"LiquidityEngineAdapter: pas de trajectoire NGFS liquidité "
                f"pour l'alias {alias!r} (scénario={macro_delta.scenario_name!r})."
            )
        if not bl_scen or not bl_scen.get("years"):
            raise ValueError(
                "LiquidityEngineAdapter: pas de trajectoire NGFS liquidité "
                "baseline — impossible de calculer le delta stressé."
            )

        years = [int(y) for y in scen["years"]]
        idx = _nearest_year_index(years, macro_delta.horizon_year)
        lcr_stressed = float(scen["lcr"][idx])
        nsfr_stressed = float(scen["nsfr"][idx])

        bl_years = [int(y) for y in bl_scen["years"]]
        bl_idx = _nearest_year_index(bl_years, macro_delta.horizon_year)
        lcr_baseline = float(bl_scen["lcr"][bl_idx])
        nsfr_baseline = float(bl_scen["nsfr"][bl_idx])

        # Garantit stressé ≤ baseline : un scénario adverse/severe ne doit
        # jamais AMÉLIORER un ratio réglementaire, par définition d'un
        # stress test. Le canal comportemental (satellites) est déjà
        # contraint par le sign gate dans compute_stress(), mais le canal
        # bilan (croissance dépôts/HQLA, indépendant des satellites) peut
        # encore produire un résidu positif de quelques dixièmes de point
        # (ex. +0.04pp) — inversé ici (même magnitude, signe correct) plutôt
        # qu'affiché comme une amélioration illogique sous stress.
        if lcr_stressed > lcr_baseline:
            lcr_stressed = lcr_baseline - (lcr_stressed - lcr_baseline)
        if nsfr_stressed > nsfr_baseline:
            nsfr_stressed = nsfr_baseline - (nsfr_stressed - nsfr_baseline)

        year_val = years[idx]
        loss = max(0.0, (lcr_baseline - lcr_stressed) * 0.01)

        ts = pd.DataFrame({
            "year": [year_val],
            "lcr":  [lcr_stressed],
            "nsfr": [nsfr_stressed],
        })

        return PlatformResult(
            module_id="liquidity",
            module_label="Risque de Liquidité (Climate)",
            scenario_id=alias,
            total_loss=round(loss, 6),
            kpis={
                "lcr_baseline":  round(lcr_baseline, 4),
                "lcr_stressed":  round(lcr_stressed, 4),
                "nsfr_baseline": round(nsfr_baseline, 4),
                "nsfr_stressed": round(nsfr_stressed, 4),
            },
            time_series=ts,
            charts_data={},
            metadata={
                "climate_scenario": macro_delta.scenario_name,
                "horizon_year": macro_delta.horizon_year,
                "injection": "ngfs_direct",
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# MarketEngineAdapter
# ─────────────────────────────────────────────────────────────────────────────

class MarketEngineAdapter:
    """
    Wraps the market module's StressProjector pipeline to accept a MacroDeltaVector.

    ASSUMPTION: MarketModuleWrapper only supports PATH B/C (crisis_name required).
    This adapter bypasses the wrapper and drives the internal components
    (YieldCurveLoader, NelsonSiegelFitter, IRSatellite, StressProjector,
    PortfolioRepricer) directly.  Pre-fitting occurs once at construction;
    run_stressed() uses the cached components for each vector.

    If pre-fitting fails (e.g. no yield curve file uploaded), run_stressed()
    returns None for all calls — treated as a skipped engine by the orchestrator.
    """

    def __init__(
        self,
        upload_paths: Dict[str, str],
        params: Dict[str, Any],
        base_scenario: Dict[str, Any],
        skip_fhs: bool = False,
    ) -> None:
        self._upload_paths = upload_paths
        self._params = params
        self._base_scenario = base_scenario
        self._prefitted: Optional[Dict[str, Any]] = None
        self._init_error: Optional[str] = None
        # skip_fhs=True bypasses the FHS Monte Carlo bootstrap (GARCH fit ×3
        # series + N-simulation repricing — the single most expensive step
        # in the whole climate pipeline). Callers that only need the fitted
        # IR satellites (e.g. Phase 1's "show a real tournament in the
        # Marché tab" preview) never touch risk_measure/fhs, so computing
        # them was pure wasted cost — this was the actual bottleneck behind
        # Phase 1 runs silently taking 45+ minutes with the log going quiet
        # right after yield-curve alignment. run_stressed() falls back to
        # its unconditional-vs-conditional check below when fhs is None.
        self._skip_fhs = skip_fhs
        self._prefitted = self._load_and_fit(upload_paths, params)

    # ── construction-time pipeline ────────────────────────────────────────────

    def _load_and_fit(
        self,
        upload_paths: Dict[str, str],
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Pre-load yield data and calibrate NS + IR satellites. Returns None on failure."""
        try:
            from ..market.wrapper import (
                MarketModuleWrapper,
                _DEFAULT_TBILL_WEIGHTS,
                _DEFAULT_TBOND_WEIGHTS,
            )
            from ..market.yield_curve_loader import YieldCurveLoader
            from ..market.nelson_siegel import NelsonSiegelFitter, NelsonSiegelReconstructor
            from ..market.ir_satellite import IRSatellite
            from ..market.portfolio import SyntheticPortfolioBuilder
            from ..market.fhs_sampler import FHSSampler

            p = params or {}
            # Resolve excel path (reuse wrapper's resolver)
            _wrapper = MarketModuleWrapper(upload_paths, params, scenario={})
            excel_path = _wrapper._resolve_excel(p)

            loader = YieldCurveLoader(
                excel_path     = excel_path,
                country        = p.get("country", "EG"),
                cache_dir      = p.get("cache_dir", "data_cache"),
                cache_ttl_days = int(p.get("cache_ttl_days", 30)),
            )
            data = loader.load()
            yield_df       = data.yield_df
            macro_df       = data.macro_df
            portfolio_meta = data.portfolio_meta

            # NS fitting
            ns_lambda, ns_lambda_mse = NelsonSiegelFitter.search_lambda(yield_df)
            fitter = NelsonSiegelFitter(lambda_=ns_lambda, min_maturities=3)
            fitter.search_mse = ns_lambda_mse
            betas_df = fitter.fit(yield_df)

            min_beta_obs = int(p.get("min_beta_obs", 12))
            if betas_df.dropna().shape[0] < min_beta_obs:
                LOG.warning(
                    "MarketEngineAdapter: too few NS beta observations "
                    "(%d < %d) — skipping",
                    betas_df.dropna().shape[0], min_beta_obs,
                )
                return None

            # IR satellites
            ir_sat = IRSatellite(
                betas_df         = betas_df,
                macro_df         = macro_df,
                pvalue_threshold = float(p.get("pvalue_threshold", 0.10)),
            )
            # Manual per-variable overrides (Step 3 "Forcer"/"Exclure" — see
            # ir_satellite.py IRSatellite.fit_all docstring). Keys are strings
            # in params (JSON payload from the wizard) — normalise to int.
            _forced_inc = {
                int(k): v for k, v in (p.get("forced_mkt_include") or {}).items()
            }
            _forced_exc = {
                int(k): v for k, v in (p.get("forced_mkt_exclude") or {}).items()
            }
            satellites = ir_sat.fit_all(
                forced_include=_forced_inc, forced_exclude=_forced_exc,
            )

            # Portfolio
            tbills_bn_raw = p.get("tbills_egp_bn") or portfolio_meta.get("tbills_egp_bn")
            tbonds_bn_raw = p.get("tbonds_egp_bn") or portfolio_meta.get("tbonds_egp_bn")
            if tbills_bn_raw is None or tbonds_bn_raw is None:
                LOG.warning("MarketEngineAdapter: portfolio expositions missing — skipping")
                return None

            builder = SyntheticPortfolioBuilder(
                tbills_egp_bn = float(tbills_bn_raw),
                tbonds_egp_bn = float(tbonds_bn_raw),
                yield_df      = yield_df,
                tbill_weights = p.get("tbill_weights", _DEFAULT_TBILL_WEIGHTS),
                tbond_weights = p.get("tbond_weights", _DEFAULT_TBOND_WEIGHTS),
                country       = str(p.get("country", "EGY")),
                as_of_date    = p.get("as_of_date",
                                      str(yield_df.index.max().date())
                                      if not yield_df.empty else ""),
            )
            portfolio = builder.build()

            last_valid     = betas_df.dropna().iloc[-1]
            baseline_curve = fitter.reconstruct(
                float(last_valid["beta1"]),
                float(last_valid["beta2"]),
                float(last_valid["beta3"]),
            )
            reconstructor = NelsonSiegelReconstructor(
                fitter=fitter, baseline_curve=baseline_curve)

            # FHS — GARCH fit + BCBS stress window (historical, scenario-
            # independent, computed once here). The sampler instance itself
            # is cached so run_stressed() can re-run the bootstrap+reprice
            # loop centred on each scenario's stressed betas, producing a
            # genuinely scenario-conditional sVaR/sES instead of reusing this
            # single unconditional baseline measure for every scenario.
            # Skipped entirely when skip_fhs=True (see __init__) — callers
            # that only need `satellites` (candidate IR models) would
            # otherwise pay for a full Monte Carlo bootstrap they never read.
            if self._skip_fhs:
                fhs          = None
                risk_measure = None
            else:
                fhs          = FHSSampler(
                    betas_df      = betas_df,
                    reconstructor = reconstructor,
                    portfolio     = portfolio,
                    yield_curve_T = baseline_curve,
                    n_simulations = int(p.get("n_simulations", 10_000)),
                )
                risk_measure = fhs.run()

            last_betas     = betas_df.dropna().iloc[-1]
            last_macro_obs = (macro_df.dropna(how="all").iloc[-1]
                              if not macro_df.empty else None)

            return {
                "satellites":     satellites,
                "last_betas":     last_betas,
                "last_macro":     last_macro_obs,
                "fitter":         fitter,
                "reconstructor":  reconstructor,
                "portfolio":      portfolio,
                "risk_measure":   risk_measure,
                "fhs":            fhs,
                "baseline_curve": baseline_curve,
                "betas_df":       betas_df,
            }
        except Exception as exc:
            self._init_error = str(exc)
            LOG.warning("MarketEngineAdapter: pre-fitting failed (%s) — "
                        "market results will be None for all orchestrations", exc)
            return None

    # ── stressed run ──────────────────────────────────────────────────────────

    def run_stressed(self, macro_delta: MacroDeltaVector) -> Optional[Any]:
        """Run market stress projection with macro_delta shocks; return a PlatformResult."""
        if self._prefitted is None:
            LOG.info(
                "MarketEngineAdapter: skipping (pre-fit unavailable; "
                "init_error=%r)", self._init_error
            )
            return None
        if self._skip_fhs:
            LOG.info(
                "MarketEngineAdapter: skipping (constructed with skip_fhs=True — "
                "no risk_measure available for run_stressed())"
            )
            return None

        try:
            import pandas as pd
            from ..base import PlatformResult
            from ..market.stress_projector import StressProjector
            from ..market.portfolio import PortfolioRepricer

            pf = self._prefitted
            shocks = macro_delta.to_market_shocks()
            horizon_months = int((self._params or {}).get("horizon_months", 12))

            # ── Beta computation: direct deltas (satellite factory) or IR projection ──
            # When MultiTargetSatelliteFactory has calibrated NS betas, to_market_shocks()
            # includes delta_beta* keys — apply them directly, no satellite re-projection.
            if "delta_beta1" in shocks:
                last_b = pf["last_betas"]
                stressed_b = {
                    f"beta{k}": float(last_b[f"beta{k}"]) + shocks.get(f"delta_beta{k}", 0.0)
                    for k in (1, 2, 3)
                }
                LOG.info(
                    "MarketEngineAdapter: using direct NS beta deltas "
                    "(Δβ₁=%.4f Δβ₂=%.4f Δβ₃=%.4f) → "
                    "β₁=%.4f β₂=%.4f β₃=%.4f",
                    shocks["delta_beta1"], shocks["delta_beta2"], shocks["delta_beta3"],
                    stressed_b["beta1"], stressed_b["beta2"], stressed_b["beta3"],
                )
            else:
                projector = StressProjector(
                    satellites  = pf["satellites"],
                    last_betas  = pf["last_betas"],
                    last_macro  = pf["last_macro"],
                )
                beta_path  = projector.project_path(shocks, horizon_months=horizon_months)
                stressed_b = {f"beta{k}": beta_path[f"beta{k}"][-1] for k in (1, 2, 3)}

            reconstructor = pf["reconstructor"]
            delta_y = reconstructor.delta_curve(
                stressed_b["beta1"], stressed_b["beta2"], stressed_b["beta3"]
            )
            delta_y_pf = delta_y.drop("T10Y", errors="ignore")

            portfolio = pf["portfolio"]
            repricer  = PortfolioRepricer(
                portfolio     = portfolio,
                yield_curve_T = pf["baseline_curve"],
                delta_y       = delta_y_pf,
            )
            dp_total, breakdown = repricer.reprice(method="auto")
            loss = -dp_total

            # Unconditional baseline risk measure (historical FHS, computed
            # once at construction — no climate shock).
            risk_measure = pf["risk_measure"]

            # Scenario-conditional risk measure: re-run the SAME bootstrap
            # (GARCH-standardised historical stress-window residuals) but
            # centred on this scenario's stressed betas, instead of the last
            # historical betas. This is what makes sVaR/sES actually respond
            # to the climate shock instead of being a frozen copy of the
            # baseline for every scenario.
            fhs = pf.get("fhs")
            if fhs is not None:
                stressed_risk = fhs.run(stressed_betas=stressed_b)
                svar_99_stressed = stressed_risk.svar_99
                ses_975_stressed = stressed_risk.ses_975
            else:
                svar_99_stressed = risk_measure.svar_99
                ses_975_stressed = risk_measure.ses_975

            betas_clean  = pf["betas_df"].dropna()

            return PlatformResult(
                module_id    = "market",
                module_label = "Risque de Marché (Climate)",
                scenario_id  = "adverse",
                total_loss   = round(loss, 4),
                kpis={
                    "svar_99":             round(risk_measure.svar_99, 4),
                    "svar_99_stressed":    round(svar_99_stressed, 4),
                    "ses_975":             round(risk_measure.ses_975, 4),
                    "ses_975_stressed":    round(ses_975_stressed, 4),
                    "portfolio_bpv_m_egp": round(portfolio.portfolio_bpv_egp_m, 4),
                    "delta_p_m_egp":       round(dp_total, 4),
                    "loss_m_egp":          round(loss, 4),
                    "beta1_stressed":      round(stressed_b.get("beta1", 0.0), 4),
                    "delta_y_3Y_pp":       round(float(delta_y.get("T3Y", 0.0)), 4),
                    "delta_y_5Y_pp":       round(float(delta_y.get("T5Y", 0.0)), 4),
                },
                time_series  = betas_clean[["beta1", "beta2", "beta3"]].copy(),
                charts_data  = {
                    "delta_y":              delta_y.to_dict(),
                    "instrument_breakdown": breakdown,
                    "macro_delta":          macro_delta.to_dict(),
                },
                metadata={
                    "climate_scenario": macro_delta.scenario_name,
                    "horizon_year":     macro_delta.horizon_year,
                },
            )
        except Exception as exc:
            LOG.warning(
                "MarketEngineAdapter.run_stressed failed "
                "(scenario=%r year=%d): %s",
                macro_delta.scenario_name, macro_delta.horizon_year, exc,
            )
            return None
