"""
test_climate_orchestrator.py
=============================
Unit tests for the Climate Orchestrator components.

All risk engines are mocked — no real engine code is called.

Tests
-----
1. MacroDeltaVector.to_dict() round-trips correctly.
2. ClimateMacroAdapter returns correct sign for a known delta
   (baseline GDP=3.0%, stressed GDP=0.5% → delta_gdp_pct=-2.5).
3. ClimateOrchestrator.run_all_scenarios() returns a result for every
   (scenario, year) combination in the input.
4. If one engine raises an exception, the orchestrator catches it, sets
   that result to None, and continues — does NOT re-raise.
5. ClimateStressResult.is_complete() returns False when any engine result
   is None.
"""
from __future__ import annotations

import sys
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# ── Make the app package importable without a Flask context ───────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Lazy-import helpers so individual test failures stay isolated ─────────────

def _import_macro_delta():
    from app.modules.climate.macro_delta import MacroDeltaVector, ClimateStressResult
    return MacroDeltaVector, ClimateStressResult


def _import_adapter():
    from app.modules.climate.climate_macro_adapter import ClimateMacroAdapter
    return ClimateMacroAdapter


def _import_orchestrator():
    from app.modules.climate.climate_orchestrator import ClimateOrchestrator
    from app.modules.climate.climate_macro_adapter import ClimateMacroAdapter
    return ClimateOrchestrator, ClimateMacroAdapter


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_platform_result(module_id="credit", scenario_id="adverse"):
    """Create a minimal mock PlatformResult."""
    mr = MagicMock()
    mr.module_id   = module_id
    mr.scenario_id = scenario_id
    mr.kpis        = {"avg_pd": 0.05, "peak_pd": 0.08}
    mr.total_loss  = 0.05
    return mr


def _make_mock_engine(return_value=None, raises=None):
    """
    Return an object with a run_stressed() method.
    If raises is set, run_stressed() raises that exception.
    Otherwise it returns return_value.
    """
    engine = MagicMock()
    if raises is not None:
        engine.run_stressed.side_effect = raises
    else:
        engine.run_stressed.return_value = return_value
    return engine


def _make_ngfs_projections(adverse="Net Zero 2050", severe="Below 2°C"):
    """Build a minimal ngfs_projections dict for tests."""
    years = [2025, 2026, 2027]
    return {
        "macro_trajectories": {
            "Baseline":    {yr: {"real_gdp_growth": 3.0,  "cpi_inflation": 5.0,
                                  "policy_rate": 10.0, "unemployment_rate": 8.0}
                            for yr in years},
            adverse:       {yr: {"real_gdp_growth": 0.5,  "cpi_inflation": 8.0,
                                  "policy_rate": 12.0, "unemployment_rate": 10.0}
                            for yr in years},
            severe:        {yr: {"real_gdp_growth": -2.0, "cpi_inflation": 12.0,
                                  "policy_rate": 15.0, "unemployment_rate": 13.0}
                            for yr in years},
        },
        "baseline_scenario": "Baseline",
        "adverse_scenario":  adverse,
        "severe_scenario":   severe,
        "ngfs_credit": {
            "scenarios": {
                "baseline": {"years": years, "pd": [0.03]*3, "lgd": [0.45]*3},
                "adverse":  {"years": years, "pd": [0.05]*3, "lgd": [0.48]*3},
                "severe":   {"years": years, "pd": [0.08]*3, "lgd": [0.52]*3},
            }
        },
        "ngfs_liquidity": {
            "scenarios": {
                "baseline": {"years": years, "lcr": [130.0]*3, "nsfr": [112.0]*3},
                "adverse":  {"years": years, "lcr": [115.0]*3, "nsfr": [108.0]*3},
                "severe":   {"years": years, "lcr": [100.0]*3, "nsfr": [103.0]*3},
            }
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — MacroDeltaVector.to_dict() round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestMacroDeltaVector:
    def test_to_dict_round_trip(self):
        """to_dict() should serialise all fields; numeric values must survive round-trip."""
        MacroDeltaVector, _ = _import_macro_delta()

        vec = MacroDeltaVector(
            scenario_name        = "Net Zero 2050",
            horizon_year         = 2027,
            delta_gdp_pct        = -3.5,
            delta_inflation_pct  =  2.1,
            delta_policy_rate_bp = 150.0,
            delta_fx_pct         = -8.0,
            delta_sovereign_spread_bp = 75.0,
            delta_oil_price_pct  = -12.0,
            delta_unemployment_pct    =  1.5,
            source_scenario_id   = "ngfs_nz2050",
        )

        d = vec.to_dict()

        assert d["scenario_name"]            == "Net Zero 2050"
        assert d["horizon_year"]             == 2027
        assert d["delta_gdp_pct"]            == -3.5
        assert d["delta_inflation_pct"]      ==  2.1
        assert d["delta_policy_rate_bp"]     == 150.0
        assert d["delta_fx_pct"]             == -8.0
        assert d["delta_sovereign_spread_bp"] == 75.0
        assert d["delta_oil_price_pct"]      == -12.0
        assert d["delta_unemployment_pct"]   ==  1.5
        assert d["source_scenario_id"]       == "ngfs_nz2050"

    def test_to_credit_shocks_conversion(self):
        """to_credit_shocks() converts pp → fraction and bp → bp."""
        MacroDeltaVector, _ = _import_macro_delta()

        vec = MacroDeltaVector(
            scenario_name="Test", horizon_year=2025,
            delta_gdp_pct=-2.5,
            delta_policy_rate_bp=50.0,
            delta_unemployment_pct=1.0,
        )
        shocks = vec.to_credit_shocks()

        assert abs(shocks["gdp_delta"]          - (-0.025)) < 1e-9
        assert abs(shocks["rate_bps"]           - 50.0)     < 1e-9
        assert abs(shocks["unemployment_delta"] - 0.01)     < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — ClimateMacroAdapter sign and magnitude
# ─────────────────────────────────────────────────────────────────────────────

class TestClimateMacroAdapter:
    def test_gdp_delta_sign_and_magnitude(self):
        """
        Baseline GDP=3.0%, stressed GDP=0.5% → delta_gdp_pct should be −2.5.
        """
        ClimateMacroAdapter = _import_adapter()
        adapter = ClimateMacroAdapter()

        ngfs = {
            "macro_trajectories": {
                "Baseline":    {2025: {"real_gdp_growth": 3.0}},
                "Net Zero 2050": {2025: {"real_gdp_growth": 0.5}},
            }
        }
        vec = adapter.compute_delta(ngfs, "Net Zero 2050", 2025, baseline_name="Baseline")

        assert abs(vec.delta_gdp_pct - (-2.5)) < 1e-9, (
            f"Expected delta_gdp_pct=-2.5, got {vec.delta_gdp_pct}"
        )
        assert vec.scenario_name == "Net Zero 2050"
        assert vec.horizon_year  == 2025

    def test_inflation_and_policy_rate_deltas(self):
        """Policy rate delta: baseline=10%, stressed=12% → Δ = +200 bp."""
        ClimateMacroAdapter = _import_adapter()
        adapter = ClimateMacroAdapter()

        ngfs = {
            "macro_trajectories": {
                "Baseline":    {2026: {"policy_rate": 10.0, "cpi_inflation": 5.0}},
                "Net Zero 2050": {2026: {"policy_rate": 12.0, "cpi_inflation": 8.0}},
            }
        }
        vec = adapter.compute_delta(ngfs, "Net Zero 2050", 2026)

        assert abs(vec.delta_policy_rate_bp - 200.0) < 1e-6, (
            f"Expected delta_policy_rate_bp=200, got {vec.delta_policy_rate_bp}"
        )
        assert abs(vec.delta_inflation_pct - 3.0) < 1e-9

    def test_missing_data_returns_zero_delta(self):
        """When no macro_trajectories and no computed results, zero vector is returned."""
        ClimateMacroAdapter = _import_adapter()
        adapter = ClimateMacroAdapter()

        vec = adapter.compute_delta({}, "UnknownScenario", 2030)

        assert vec.delta_gdp_pct        == 0.0
        assert vec.delta_policy_rate_bp == 0.0
        assert vec.scenario_name        == "UnknownScenario"

    def test_plausibility_warning_logged(self, caplog):
        """A delta beyond the plausibility bound triggers a WARNING log."""
        import logging
        ClimateMacroAdapter = _import_adapter()
        adapter = ClimateMacroAdapter()

        ngfs = {
            "macro_trajectories": {
                "Baseline":    {2025: {"real_gdp_growth":  3.0}},
                "Extreme":     {2025: {"real_gdp_growth": -50.0}},   # -53pp delta
            }
        }

        with caplog.at_level(logging.WARNING, logger="climate.macro_adapter"):
            adapter.compute_delta(ngfs, "Extreme", 2025)

        assert any(
            "suspect" in r.message.lower() or "borne" in r.message.lower()
            for r in caplog.records
        ), "Expected a WARNING about suspicious delta magnitude"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — ClimateOrchestrator.run_all_scenarios() completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestClimateOrchestratorCompleteness:
    def test_result_for_every_scenario_year(self):
        """
        run_all_scenarios() must return a ClimateStressResult for every
        (scenario, year) pair in the input.
        """
        ClimateOrchestrator, ClimateMacroAdapter = _import_orchestrator()
        _, ClimateStressResult = _import_macro_delta()

        mock_result = _make_mock_platform_result()

        credit_eng    = _make_mock_engine(return_value=mock_result)
        liquidity_eng = _make_mock_engine(return_value=mock_result)
        market_eng    = _make_mock_engine(return_value=mock_result)
        adapter       = ClimateMacroAdapter()

        orch = ClimateOrchestrator(
            credit_engine    = credit_eng,
            liquidity_engine = liquidity_eng,
            market_engine    = market_eng,
            adapter          = adapter,
        )

        scenarios    = ["Net Zero 2050", "Below 2°C"]
        horizon_years = [2025, 2026, 2027]
        ngfs = _make_ngfs_projections()

        results = orch.run_all_scenarios(ngfs, horizon_years, scenarios)

        # Every combination must be present
        for scen in scenarios:
            assert scen in results, f"Missing scenario key: {scen}"
            for yr in horizon_years:
                assert yr in results[scen], (
                    f"Missing year {yr} in scenario {scen}"
                )
                entry = results[scen][yr]
                assert isinstance(entry, ClimateStressResult)

    def test_all_engines_called_per_combination(self):
        """Each engine's run_stressed() must be called once per (scenario, year) pair."""
        ClimateOrchestrator, ClimateMacroAdapter = _import_orchestrator()

        mock_result   = _make_mock_platform_result()
        credit_eng    = _make_mock_engine(return_value=mock_result)
        liquidity_eng = _make_mock_engine(return_value=mock_result)
        market_eng    = _make_mock_engine(return_value=mock_result)
        adapter       = ClimateMacroAdapter()

        orch = ClimateOrchestrator(credit_eng, liquidity_eng, market_eng, adapter)

        scenarios     = ["Net Zero 2050"]
        horizon_years = [2025, 2026]
        ngfs          = _make_ngfs_projections()

        orch.run_all_scenarios(ngfs, horizon_years, scenarios)

        n_expected = len(scenarios) * len(horizon_years)
        assert credit_eng.run_stressed.call_count    == n_expected
        assert liquidity_eng.run_stressed.call_count == n_expected
        assert market_eng.run_stressed.call_count    == n_expected


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Engine exception isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestClimateOrchestratorErrorIsolation:
    def test_failing_engine_does_not_raise(self):
        """
        When one engine raises RuntimeError, the orchestrator must NOT re-raise;
        the failing engine's result must be None; the other engines must succeed.
        """
        ClimateOrchestrator, ClimateMacroAdapter = _import_orchestrator()

        good_result   = _make_mock_platform_result()
        credit_eng    = _make_mock_engine(raises=RuntimeError("simulated credit failure"))
        liquidity_eng = _make_mock_engine(return_value=good_result)
        market_eng    = _make_mock_engine(return_value=good_result)
        adapter       = ClimateMacroAdapter()

        orch = ClimateOrchestrator(credit_eng, liquidity_eng, market_eng, adapter)

        ngfs          = _make_ngfs_projections()
        horizon_years = [2025]
        scenarios     = ["Net Zero 2050"]

        # Must not raise
        results = orch.run_all_scenarios(ngfs, horizon_years, scenarios)

        entry = results["Net Zero 2050"][2025]
        assert entry.credit    is None,       "credit must be None when engine raises"
        assert entry.liquidity is good_result, "liquidity must still succeed"
        assert entry.market    is good_result, "market must still succeed"
        assert "credit" in entry.errors,       "error dict must record failing engine"

    def test_all_engines_fail_gracefully(self):
        """All three engines failing must still produce ClimateStressResult objects."""
        ClimateOrchestrator, ClimateMacroAdapter = _import_orchestrator()

        boom = ValueError("boom")
        orch = ClimateOrchestrator(
            credit_engine    = _make_mock_engine(raises=boom),
            liquidity_engine = _make_mock_engine(raises=boom),
            market_engine    = _make_mock_engine(raises=boom),
            adapter          = ClimateMacroAdapter(),
        )

        results = orch.run_all_scenarios(
            _make_ngfs_projections(), [2025], ["Net Zero 2050"]
        )

        entry = results["Net Zero 2050"][2025]
        assert entry.credit    is None
        assert entry.liquidity is None
        assert entry.market    is None
        assert len(entry.errors) == 3

    def test_run_summary_counts_failures(self):
        """run_summary() must accurately report per-engine failure counts."""
        ClimateOrchestrator, ClimateMacroAdapter = _import_orchestrator()

        boom = RuntimeError("boom")
        orch = ClimateOrchestrator(
            credit_engine    = _make_mock_engine(raises=boom),
            liquidity_engine = _make_mock_engine(raises=boom),
            market_engine    = _make_mock_engine(return_value=_make_mock_platform_result()),
            adapter          = ClimateMacroAdapter(),
        )

        orch.run_all_scenarios(_make_ngfs_projections(), [2025, 2026], ["Net Zero 2050"])

        summary = orch.run_summary()
        assert summary["credit"]["failures"]    == 2   # 1 scen × 2 years
        assert summary["liquidity"]["failures"]  == 2
        assert summary["market"]["successes"]    == 2
        assert summary["market"]["failures"]     == 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — ClimateStressResult.is_complete()
# ─────────────────────────────────────────────────────────────────────────────

class TestClimateStressResult:
    def _make_result(self, credit=None, liquidity=None, market=None):
        _, ClimateStressResult = _import_macro_delta()
        from app.modules.climate.macro_delta import MacroDeltaVector
        return ClimateStressResult(
            scenario     = "Test",
            horizon_year = 2025,
            macro_delta  = MacroDeltaVector("Test", 2025),
            credit       = credit,
            liquidity    = liquidity,
            market       = market,
        )

    def test_is_complete_all_present(self):
        """is_complete() must return True when all three results are non-None."""
        mr = _make_mock_platform_result()
        r  = self._make_result(credit=mr, liquidity=mr, market=mr)
        assert r.is_complete() is True

    def test_is_complete_credit_missing(self):
        """is_complete() must return False when credit is None."""
        mr = _make_mock_platform_result()
        r  = self._make_result(credit=None, liquidity=mr, market=mr)
        assert r.is_complete() is False

    def test_is_complete_liquidity_missing(self):
        mr = _make_mock_platform_result()
        r  = self._make_result(credit=mr, liquidity=None, market=mr)
        assert r.is_complete() is False

    def test_is_complete_market_missing(self):
        mr = _make_mock_platform_result()
        r  = self._make_result(credit=mr, liquidity=mr, market=None)
        assert r.is_complete() is False

    def test_is_complete_all_none(self):
        r = self._make_result()
        assert r.is_complete() is False

    def test_to_dashboard_dict_structure(self):
        """to_dashboard_dict() must return expected keys for a complete result."""
        _, ClimateStressResult = _import_macro_delta()
        from app.modules.climate.macro_delta import MacroDeltaVector

        # Mock liquidity result with kpis
        liq = MagicMock()
        liq.kpis = {
            "lcr_baseline": 130.0, "lcr_stressed": 115.0,
            "nsfr_baseline": 112.0, "nsfr_stressed": 108.0,
        }
        liq.charts_data = {}

        cr = MagicMock()
        cr.kpis = {"avg_pd": 0.05, "peak_pd": 0.08}
        cr.charts_data = {}

        mkt = MagicMock()
        mkt.kpis = {"svar_99": 1.2, "portfolio_bpv_m_egp": 5.0, "delta_p_m_egp": -3.0}
        mkt.charts_data = {}

        r = ClimateStressResult(
            scenario="Net Zero 2050", horizon_year=2025,
            macro_delta=MacroDeltaVector("Net Zero 2050", 2025, delta_gdp_pct=-2.5),
            credit=cr, liquidity=liq, market=mkt,
        )
        d = r.to_dashboard_dict()

        assert d["scenario"]     == "Net Zero 2050"
        assert d["horizon_year"] == 2025
        assert d["is_complete"]  is True
        assert "rows" in d
        assert len(d["rows"]) > 0

        # Every row must have rag in {green, amber, red}
        for row in d["rows"]:
            if not row.get("error"):
                assert row["rag"] in ("green", "amber", "red"), (
                    f"Unexpected RAG value: {row['rag']}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — ClimateMacroAdapter.compute_all_deltas() shape
# ─────────────────────────────────────────────────────────────────────────────

class TestAdapterComputeAll:
    def test_compute_all_returns_full_grid(self):
        """compute_all_deltas() must return every (scenario, year) combination."""
        ClimateMacroAdapter = _import_adapter()
        adapter = ClimateMacroAdapter()

        scenarios = ["Net Zero 2050", "Below 2°C"]
        years     = [2025, 2026, 2027]
        ngfs      = _make_ngfs_projections()

        grid = adapter.compute_all_deltas(ngfs, scenarios, years)

        assert set(grid.keys()) == set(scenarios)
        for scen in scenarios:
            assert set(grid[scen].keys()) == set(years)
