"""
macro_delta.py
==============
Standardised shock vector and aggregated result types for the Climate
Orchestrator pattern.  All types live inside the climate module and are
never imported by any other risk module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

LOG = logging.getLogger("climate.macro_delta")


# ─────────────────────────────────────────────────────────────────────────────
# MacroDeltaVector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MacroDeltaVector:
    """Carries NGFS-translated macro shocks to any risk engine for one (scenario, year) pair."""

    scenario_name: str
    horizon_year: int

    # Macro variable deltas — all expressed as percentage-point changes vs baseline
    delta_gdp_pct: float = 0.0            # GDP growth delta in pp  (−3.0 = −3 pp)
    delta_inflation_pct: float = 0.0      # CPI inflation delta in pp
    delta_policy_rate_bp: float = 0.0     # Policy rate delta in basis points
    delta_fx_pct: float = 0.0             # FX rate delta in pp (negative = depreciation of LCU)
    delta_sovereign_spread_bp: float = 0.0  # Sovereign spread delta in bp (proxy via gov debt / lending rate)
    delta_oil_price_pct: float = 0.0      # Oil price delta in pp
    delta_unemployment_pct: float = 0.0   # Unemployment delta in pp

    # Traceability
    source_scenario_id: str = ""          # NGFS internal scenario ID

    # ── Direct target-level deltas (from MultiTargetSatelliteFactory) ─────────
    # These fields are populated by ClimateMacroAdapter after the satellite
    # factory has run.  When non-zero they take precedence over macro-proxy
    # approximations inside to_credit_shocks() / to_market_shocks().
    #
    # Units:
    #   delta_pd_pp   — PD delta in percentage points  (e.g. +2.0 = PD rose 2 pp)
    #   delta_lcr_pp  — ΔLCR in pp vs baseline         (e.g. −15.0 = LCR fell 15 pp)
    #   delta_nsfr_pp — ΔNSFR in pp vs baseline
    #   delta_beta1/2/3 — NS beta deltas (absolute, not pp)
    delta_pd_pp:   float = 0.0
    delta_lcr_pp:  float = 0.0
    delta_nsfr_pp: float = 0.0
    delta_beta1:   float = 0.0
    delta_beta2:   float = 0.0
    delta_beta3:   float = 0.0

    def has_direct_targets(self) -> bool:
        """True when at least one direct satellite delta has been populated."""
        return any([
            self.delta_pd_pp   != 0.0,
            self.delta_lcr_pp  != 0.0,
            self.delta_nsfr_pp != 0.0,
            self.delta_beta1   != 0.0,
            self.delta_beta2   != 0.0,
            self.delta_beta3   != 0.0,
        ])

    def has_monetary_transmission(self) -> bool:
        """
        True when at least one classic monetary-channel input (policy rate,
        inflation, FX, sovereign spread) carries a non-zero shock for this
        scenario/year, OR a direct beta1/2/3 satellite delta was computed
        (Layer 2 bypasses the Layer-1 macro-proxy columns entirely).

        False means the market/yield-curve engine has nothing to respond to
        for this scenario: sVaR/sES will replay flat and identical across
        scenarios. This is a genuine NGFS source-data coverage limit, not a
        bug — e.g. a GEM-E3/CT file models only the real economy
        (production, trade, employment, emissions) and never carries
        interest-rate, inflation, FX, or sovereign-debt variables for any
        scenario, so there is no monetary shock to translate no matter how
        the mapping is written. Used by ClimateStressResult.to_dashboard_dict()
        to attach an explicit "no transmission channel" note instead of
        silently reporting a flat result as if it were a real finding.
        """
        return any([
            self.delta_policy_rate_bp      != 0.0,
            self.delta_inflation_pct       != 0.0,
            self.delta_fx_pct              != 0.0,
            self.delta_sovereign_spread_bp != 0.0,
            self.delta_beta1 != 0.0,
            self.delta_beta2 != 0.0,
            self.delta_beta3 != 0.0,
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (for Jinja2 templates and JSON responses)."""
        return {
            "scenario_name":             self.scenario_name,
            "horizon_year":              self.horizon_year,
            "delta_gdp_pct":             self.delta_gdp_pct,
            "delta_inflation_pct":       self.delta_inflation_pct,
            "delta_policy_rate_bp":      self.delta_policy_rate_bp,
            "delta_fx_pct":              self.delta_fx_pct,
            "delta_sovereign_spread_bp": self.delta_sovereign_spread_bp,
            "delta_oil_price_pct":       self.delta_oil_price_pct,
            "delta_unemployment_pct":    self.delta_unemployment_pct,
            "source_scenario_id":        self.source_scenario_id,
            # Direct satellite deltas (0.0 when satellite not calibrated)
            "delta_pd_pp":               self.delta_pd_pp,
            "delta_lcr_pp":              self.delta_lcr_pp,
            "delta_nsfr_pp":             self.delta_nsfr_pp,
            "delta_beta1":               self.delta_beta1,
            "delta_beta2":               self.delta_beta2,
            "delta_beta3":               self.delta_beta3,
        }

    def to_credit_shocks(self) -> Dict[str, float]:
        """Convert to the shock-key dict expected by CreditModuleWrapper scenario levels."""
        return {
            "gdp_delta":           self.delta_gdp_pct / 100.0,
            "rate_bps":            self.delta_policy_rate_bp,
            "unemployment_delta":  self.delta_unemployment_pct / 100.0,
            "inflation_delta":     self.delta_inflation_pct / 100.0,
            "fx_pct":              self.delta_fx_pct / 100.0,
            "oil_pct":             self.delta_oil_price_pct / 100.0,
        }

    def to_liquidity_shocks(self) -> Dict[str, float]:
        """Convert to the shock-key dict expected by LiquidityModuleWrapper scenario levels."""
        # Liquidity wrapper uses the same shock-key naming convention as credit
        return self.to_credit_shocks()

    def to_market_shocks(self) -> Dict[str, float]:
        """
        Convert to the shock-key dict used by StressProjector / MarketEngineAdapter.

        When delta_beta* are non-zero (direct satellite output from
        MultiTargetSatelliteFactory), they are included in the returned dict.
        MarketEngineAdapter checks for these keys and bypasses IR satellite
        projection when present, applying the deltas directly to last_betas.
        """
        shocks: Dict[str, float] = {
            "rate_bps":        self.delta_policy_rate_bp,
            "gdp_delta":       self.delta_gdp_pct / 100.0,
            "inflation_delta": self.delta_inflation_pct / 100.0,
        }
        # Propagate direct NS beta deltas when the satellite factory ran
        if self.delta_beta1 != 0.0 or self.delta_beta2 != 0.0 or self.delta_beta3 != 0.0:
            shocks["delta_beta1"] = self.delta_beta1
            shocks["delta_beta2"] = self.delta_beta2
            shocks["delta_beta3"] = self.delta_beta3
        return shocks


# ─────────────────────────────────────────────────────────────────────────────
# ClimateStressResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClimateStressResult:
    """Aggregates PlatformResult objects from all three risk engines for one (scenario, year) pair."""

    scenario: str
    horizon_year: int
    macro_delta: MacroDeltaVector

    # One PlatformResult per engine — None when the engine failed
    credit: Optional[Any] = None      # PlatformResult | None
    liquidity: Optional[Any] = None   # PlatformResult | None
    market: Optional[Any] = None      # PlatformResult | None

    run_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    errors: Dict[str, str] = field(default_factory=dict)   # engine_name → error message

    # ── RAG thresholds (configurable, defaults per spec) ─────────────────────
    rag_amber_threshold: float = 0.10   # 10 % worse → amber
    rag_red_threshold: float = 0.25     # 25 % worse → red

    def is_complete(self) -> bool:
        """Return True only when all three engine results are non-None."""
        return (self.credit is not None
                and self.liquidity is not None
                and self.market is not None)

    def _rag(self, baseline_val: float, stressed_val: float, worse_is_higher: bool = True) -> str:
        """Compute RAG signal for a single metric."""
        if baseline_val == 0:
            return "amber"
        pct_change = (stressed_val - baseline_val) / abs(baseline_val)
        if worse_is_higher:
            delta = pct_change
        else:
            delta = -pct_change   # lower is worse (LCR, NSFR, etc.)
        if delta <= self.rag_amber_threshold:
            return "green"
        elif delta <= self.rag_red_threshold:
            return "amber"
        return "red"

    def to_dashboard_dict(self) -> Dict[str, Any]:
        """Return a flat dict ready for Jinja2 rendering."""
        rows = []

        # ── Credit rows ───────────────────────────────────────────────────────
        if self.credit is not None:
            kpis = self.credit.kpis or {}
            # avg_pd as proxy for EL
            bl_pd = float(kpis.get("avg_pd", 0.0) or 0.0)
            st_pd = float(kpis.get("peak_pd", bl_pd) or bl_pd)
            rows.append({
                "module": "Crédit",
                "metric": "EL (% PD moyenne)",
                "baseline": round(bl_pd * 100, 4),
                "stressed": round(st_pd * 100, 4),
                "delta": round((st_pd - bl_pd) * 100, 4),
                "unit": "%",
                "rag": self._rag(bl_pd, st_pd, worse_is_higher=True),
                "error": None,
            })
            # RWA from capital KPIs if available
            _cd = self.credit.charts_data or {}
            _cap = _cd.get("capital_trajectories", {})
            if _cap:
                # Each level here is {"x": [years], "y": [values]}, not a plain
                # list — indexing the dict itself with [-1] raises KeyError(-1).
                # scenario_id (set by CreditEngineAdapter via _resolve_alias) is
                # the "adverse"/"severe" alias matching capital_trajectories'
                # keys — NOT self.scenario, which holds the raw NGFS scenario
                # name (e.g. "Net Zero 2050"). Using a hardcoded "adverse" here
                # previously made the RWA KPI show the adverse trajectory even
                # when displaying the severe scenario.
                _alias = self.credit.scenario_id or "adverse"
                bl_rwa_traj = _cap.get("rwa_stressed", {}).get("baseline", {}) or {}
                st_rwa_traj = _cap.get("rwa_stressed", {}).get(_alias, {}) or {}
                bl_rwa_x = bl_rwa_traj.get("x", []) if isinstance(bl_rwa_traj, dict) else []
                bl_rwa_y = bl_rwa_traj.get("y", []) if isinstance(bl_rwa_traj, dict) else []
                st_rwa_x = st_rwa_traj.get("x", []) if isinstance(st_rwa_traj, dict) else []
                st_rwa_y = st_rwa_traj.get("y", []) if isinstance(st_rwa_traj, dict) else []

                def _year_value(xs, ys, year):
                    # Pick the value matching this ClimateStressResult's own
                    # horizon_year instead of always taking the trajectory's
                    # last point (previously froze the KPI on year 2050
                    # regardless of the dashboard's year selector).
                    target = str(int(year))
                    if target in xs:
                        return ys[xs.index(target)]
                    return ys[-1] if ys else None

                bl_rwa_v = _year_value(bl_rwa_x, bl_rwa_y, self.horizon_year)
                st_rwa_v = _year_value(st_rwa_x, st_rwa_y, self.horizon_year)
                if bl_rwa_v is not None and st_rwa_v is not None:
                    # capital_engine.py computes rwa_stressed in raw EGP (same
                    # scale as the uploaded capital CSV's CET1/total_capital
                    # columns — verified there via the CAR sanity check).
                    # Convert to EGP millions to match this platform's monetary
                    # KPI convention (EAD, sVaR, portfolio BPV are all "M EGP");
                    # labelling the raw value "Mds" without this conversion
                    # inflated the displayed figure by ~1e3.
                    bl_rwa_m = float(bl_rwa_v) / 1_000_000.0
                    st_rwa_m = float(st_rwa_v) / 1_000_000.0
                    rows.append({
                        "module": "Crédit",
                        "metric": "RWA stressé",
                        "baseline": round(bl_rwa_m, 2),
                        "stressed": round(st_rwa_m, 2),
                        "delta": round(st_rwa_m - bl_rwa_m, 2),
                        "unit": "M",
                        "rag": self._rag(bl_rwa_m, st_rwa_m, worse_is_higher=True),
                        "error": None,
                    })
        else:
            rows.append({
                "module": "Crédit",
                "metric": "EL / RWA",
                "baseline": None, "stressed": None, "delta": None, "unit": "",
                "rag": "amber",
                "error": self.errors.get("credit", "Erreur de calcul"),
            })

        # ── Liquidity rows ────────────────────────────────────────────────────
        if self.liquidity is not None:
            kpis = self.liquidity.kpis or {}
            bl_lcr  = float(kpis.get("lcr_baseline",  100.0) or 100.0)
            st_lcr  = float(kpis.get("lcr_stressed",  bl_lcr) or bl_lcr)
            bl_nsfr = float(kpis.get("nsfr_baseline", 100.0) or 100.0)
            st_nsfr = float(kpis.get("nsfr_stressed", bl_nsfr) or bl_nsfr)
            rows.append({
                "module": "Liquidité",
                "metric": "LCR Δ (pp)",
                "baseline": round(bl_lcr, 2),
                "stressed": round(st_lcr, 2),
                "delta": round(st_lcr - bl_lcr, 2),
                "unit": "pp",
                "rag": self._rag(bl_lcr, st_lcr, worse_is_higher=False),
                "error": None,
            })
            rows.append({
                "module": "Liquidité",
                "metric": "NSFR Δ (pp)",
                "baseline": round(bl_nsfr, 2),
                "stressed": round(st_nsfr, 2),
                "delta": round(st_nsfr - bl_nsfr, 2),
                "unit": "pp",
                "rag": self._rag(bl_nsfr, st_nsfr, worse_is_higher=False),
                "error": None,
            })
        else:
            rows.append({
                "module": "Liquidité",
                "metric": "LCR Δ / NSFR Δ",
                "baseline": None, "stressed": None, "delta": None, "unit": "",
                "rag": "amber",
                "error": self.errors.get("liquidity", "Erreur de calcul"),
            })

        # ── Market rows ───────────────────────────────────────────────────────
        if self.market is not None:
            kpis = self.market.kpis or {}
            bl_svar = float(kpis.get("svar_99", 0.0) or 0.0)
            # svar_99_stressed: FHS bootstrap re-centred on this scenario's
            # stressed NS betas (see MarketEngineAdapter.run_stressed()) —
            # falls back to the unconditional baseline only if the stressed
            # measure wasn't computed (e.g. older cached run).
            st_svar = float(kpis.get("svar_99_stressed", bl_svar) or bl_svar)
            bl_bpv  = float(kpis.get("portfolio_bpv_m_egp", 0.0) or 0.0)
            st_bpv  = float(kpis.get("delta_p_m_egp", 0.0) or 0.0)

            # Explicit, machine-readable documentation of the "flat/identical
            # across scenarios" case (see has_monetary_transmission()) — a
            # data-coverage limit of the NGFS source, not a defect. Attached
            # to every Market row instead of just the dashboard banner, so
            # any consumer of the JSON/API output (not only the chart) can
            # detect and surface it.
            _mkt_note = None if self.macro_delta.has_monetary_transmission() else (
                "Aucune variable monétaire (taux directeur, inflation, change, "
                "spread souverain) disponible dans le fichier NGFS source pour "
                "ce scénario — le moteur marché ne peut pas faire diverger "
                "sVaR/sES de la baseline ni entre scénarios. Limite de "
                "couverture des données NGFS (ex. fichier GEM-E3/CT, modèle "
                "d'équilibre général qui ne projette pas de variables "
                "monétaires), pas une erreur de calcul."
            )

            rows.append({
                "module": "Marché",
                "metric": "sVaR 99% (M)",
                "baseline": round(bl_svar, 4),
                "stressed": round(st_svar, 4),
                "delta": round(st_svar - bl_svar, 4),
                "unit": "M",
                "rag": self._rag(bl_svar, st_svar, worse_is_higher=True),
                "error": None,
                "note": _mkt_note,
            })
            # sES 97.5% — same conditional/unconditional pattern as sVaR
            # (see MarketEngineAdapter.run_stressed / FHSSampler). Additive
            # row for the climate dashboard's Row 5 market metrics chart.
            bl_ses = float(kpis.get("ses_975", 0.0) or 0.0)
            st_ses = float(kpis.get("ses_975_stressed", bl_ses) or bl_ses)
            rows.append({
                "module": "Marché",
                "metric": "sES 97.5% (M)",
                "baseline": round(bl_ses, 4),
                "stressed": round(st_ses, 4),
                "delta": round(st_ses - bl_ses, 4),
                "unit": "M",
                "rag": self._rag(bl_ses, st_ses, worse_is_higher=True),
                "error": None,
                "note": _mkt_note,
            })
            rows.append({
                "module": "Marché",
                "metric": "ΔP portefeuille (M)",
                "baseline": 0.0,
                "stressed": round(st_bpv, 4),
                "delta": round(st_bpv, 4),
                "unit": "M",
                "rag": self._rag(abs(bl_bpv) if bl_bpv else 1.0, abs(st_bpv),
                                  worse_is_higher=True),
                "error": None,
                "note": _mkt_note,
            })
        else:
            rows.append({
                "module": "Marché",
                "metric": "sVaR / BPV",
                "baseline": None, "stressed": None, "delta": None, "unit": "",
                "rag": "amber",
                "error": self.errors.get("market", "Erreur de calcul"),
            })

        # market_delta_y: full per-maturity yield-curve delta for this
        # (scenario, year), already computed by MarketEngineAdapter.run_stressed
        # (charts_data["delta_y"]) but never previously surfaced past `rows`.
        # Additive — used by Row 3 Bloc C to reconstruct the stressed curve
        # as market_baseline_curve[maturity] + market_delta_y[maturity].
        market_delta_y: Dict[str, float] = {}
        if self.market is not None:
            _mcd = self.market.charts_data or {}
            market_delta_y = _mcd.get("delta_y", {}) or {}

        return {
            "scenario":       self.scenario,
            "horizon_year":   self.horizon_year,
            "macro_delta":    self.macro_delta.to_dict(),
            "rows":           rows,
            "market_delta_y": market_delta_y,
            # Top-level, machine-readable mirror of the per-row "note" above —
            # lets any API/export consumer check this once instead of
            # inspecting every Market row. None when self.market itself is
            # unavailable (a different failure mode, see "errors" instead).
            "market_transmission_available": (
                self.macro_delta.has_monetary_transmission()
                if self.market is not None else None
            ),
            "is_complete":    self.is_complete(),
            "errors":         self.errors,
            "run_timestamp":  self.run_timestamp.isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Market projection series (sVaR / sES) — climate dashboard Row 3/Row 5
# ─────────────────────────────────────────────────────────────────────────────

def get_market_projection_series(
    csr_results: Dict[str, Dict[str, dict]],
    adverse_name: str,
    severe_name: str,
    metric: str = "sVaR 99% (M)",
):
    """
    Builds a ProjectionSeries for a market metric (sVaR, sES) by scanning
    already-serialised ClimateStressResult.to_dashboard_dict() rows across
    every horizon year run for the adverse/severe scenarios.

    Additive only — pure aggregation over data ClimateOrchestrator already
    computed (csr_results = record.module_results.climate.baseline
    .charts_data.climate_stress_results.results); no new engine call.

    No historical sVaR/sES series exists anywhere on this platform (Step 0
    audit confirmed it) — historical_available is always False here; the
    dashboard must render a visible "Historique non disponible" note,
    never fabricate historical data (hard constraint 4).

    Parameters
    ----------
    csr_results  : {scenario_name: {year_str: to_dashboard_dict()}}
    adverse_name, severe_name : actual NGFS scenario names for this run
                    (dynamically resolved — see climate_macro_adapter.py).
    metric       : row "metric" label to extract, e.g. "sVaR 99% (M)".

    Returns
    -------
    ProjectionSeries | None if neither scenario has any data for this metric.
    """
    from ..base import ProjectionSeries

    def _series_for(scenario_name: str):
        years_data = csr_results.get(scenario_name, {}) or {}
        dates, stressed, baseline = [], [], []
        for yr in sorted(years_data.keys(), key=lambda y: int(y)):
            rows = (years_data.get(yr) or {}).get("rows", [])
            row = next(
                (r for r in rows if r.get("metric") == metric and not r.get("error")),
                None,
            )
            if row is None:
                continue
            dates.append(int(yr))
            stressed.append(row.get("stressed"))
            baseline.append(row.get("baseline"))
        return dates, stressed, baseline

    adv_dates, adv_vals, adv_base = _series_for(adverse_name)
    sev_dates, sev_vals, sev_base = _series_for(severe_name)

    if not adv_dates and not sev_dates:
        return None

    # Baseline (FHS unconditional sVaR) doesn't vary by scenario by
    # construction — prefer whichever scenario has data for a given year.
    base_dates = adv_dates if len(adv_dates) >= len(sev_dates) else sev_dates
    base_vals = adv_base if len(adv_dates) >= len(sev_dates) else sev_base

    unit = "M EGP" if "sVaR" in metric or "sES" in metric else ""

    return ProjectionSeries(
        variable_name=metric,
        historical_dates=[], historical_values=[],
        baseline_dates=base_dates, baseline_values=base_vals,
        adverse_dates=adv_dates, adverse_values=adv_vals,
        severe_dates=sev_dates, severe_values=sev_vals,
        unit=unit,
        historical_available=False,
    )
