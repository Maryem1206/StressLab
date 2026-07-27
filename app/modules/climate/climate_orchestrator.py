"""
climate_orchestrator.py
=======================
ClimateOrchestrator — the core component that coordinates all three risk
engines across every (NGFS scenario, horizon year) pair.

Design rules:
  • Receives the three risk engines (adapters) via dependency injection —
    it NEVER instantiates them internally.
  • Catches any per-engine exception, logs it, sets that result to None,
    and continues with the other engines.
  • Returns a nested dict {scenario_name: {year: ClimateStressResult}}.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from .climate_macro_adapter import ClimateMacroAdapter
from .macro_delta import ClimateStressResult, MacroDeltaVector

LOG = logging.getLogger("climate.orchestrator")


class ClimateOrchestrator:
    """
    Orchestrates climate-transmission stress runs across all three risk engines.

    Parameters
    ----------
    credit_engine : any object with run_stressed(MacroDeltaVector) -> PlatformResult|None
    liquidity_engine : same interface
    market_engine : same interface
    adapter : ClimateMacroAdapter
    """

    # Engine names used in logging, errors, and run_summary()
    _ENGINE_NAMES = ("credit", "liquidity", "market")

    def __init__(
        self,
        credit_engine: Any,
        liquidity_engine: Any,
        market_engine: Any,
        adapter: ClimateMacroAdapter,
        rag_amber_threshold: float = 0.10,
        rag_red_threshold: float = 0.25,
    ) -> None:
        self._engines = {
            "credit":    credit_engine,
            "liquidity": liquidity_engine,
            "market":    market_engine,
        }
        self._adapter = adapter
        self._rag_amber_threshold = rag_amber_threshold
        self._rag_red_threshold = rag_red_threshold

        # Counters for run_summary()
        self._successes: Dict[str, int] = {n: 0 for n in self._ENGINE_NAMES}
        self._failures:  Dict[str, int] = {n: 0 for n in self._ENGINE_NAMES}
        self._skipped:   Dict[str, int] = {n: 0 for n in self._ENGINE_NAMES}

    # ── main entry point ──────────────────────────────────────────────────────

    def run_all_scenarios(
        self,
        ngfs_projections: Dict[str, Any],
        horizon_years: List[int],
        scenarios: List[str],
    ) -> Dict[str, Dict[int, ClimateStressResult]]:
        """
        Run all three engines for every (scenario, year) combination.

        Parameters
        ----------
        ngfs_projections : dict produced by ClimateMacroAdapter-compatible sources
        horizon_years    : list of integer years to stress-test
        scenarios        : list of NGFS scenario names to iterate over

        Returns
        -------
        {scenario_name: {year: ClimateStressResult}}
        """
        # Reset counters for this run
        for n in self._ENGINE_NAMES:
            self._successes[n] = 0
            self._failures[n]  = 0
            self._skipped[n]   = 0

        results: Dict[str, Dict[int, ClimateStressResult]] = {}

        for scenario_name in scenarios:
            results[scenario_name] = {}

            for year in horizon_years:
                LOG.info(
                    "ClimateOrchestrator: running scenario=%r year=%d",
                    scenario_name, year,
                )

                # 1. Compute the MacroDeltaVector for this (scenario, year)
                try:
                    macro_delta = self._adapter.compute_delta(
                        ngfs_projections, scenario_name, year)
                except Exception as exc:
                    LOG.error(
                        "ClimateOrchestrator: adapter.compute_delta failed "
                        "(scenario=%r year=%d): %s — using zero delta",
                        scenario_name, year, exc,
                    )
                    macro_delta = MacroDeltaVector(
                        scenario_name=scenario_name,
                        horizon_year=year,
                        source_scenario_id=f"{scenario_name}:{year}:fallback_zero",
                    )

                # 2. Run each engine — catch and log failures individually
                engine_results: Dict[str, Any]     = {}
                engine_errors:  Dict[str, str]     = {}

                for engine_name in self._ENGINE_NAMES:
                    engine = self._engines[engine_name]
                    if engine is None:
                        self._skipped[engine_name] += 1
                        LOG.info(
                            "ClimateOrchestrator: engine=%s skipped (None injected)",
                            engine_name,
                        )
                        continue

                    try:
                        result = engine.run_stressed(macro_delta)
                        engine_results[engine_name] = result
                        if result is not None:
                            self._successes[engine_name] += 1
                        else:
                            self._skipped[engine_name] += 1
                    except Exception as exc:
                        self._failures[engine_name] += 1
                        err_msg = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        engine_errors[engine_name] = err_msg
                        engine_results[engine_name] = None
                        LOG.error(
                            "ClimateOrchestrator: engine=%s raised an exception "
                            "(scenario=%r year=%d) — result set to None. Error: %s",
                            engine_name, scenario_name, year, err_msg,
                        )

                # 3. Assemble ClimateStressResult
                stress_result = ClimateStressResult(
                    scenario=scenario_name,
                    horizon_year=year,
                    macro_delta=macro_delta,
                    credit=engine_results.get("credit"),
                    liquidity=engine_results.get("liquidity"),
                    market=engine_results.get("market"),
                    run_timestamp=datetime.now(timezone.utc),
                    errors=engine_errors,
                    rag_amber_threshold=self._rag_amber_threshold,
                    rag_red_threshold=self._rag_red_threshold,
                )

                results[scenario_name][year] = stress_result

                LOG.info(
                    "ClimateOrchestrator: done scenario=%r year=%d — "
                    "complete=%s credit=%s liquidity=%s market=%s",
                    scenario_name, year,
                    stress_result.is_complete(),
                    "OK" if stress_result.credit    is not None else "NONE",
                    "OK" if stress_result.liquidity  is not None else "NONE",
                    "OK" if stress_result.market     is not None else "NONE",
                )

        return results

    # ── diagnostics ───────────────────────────────────────────────────────────

    def run_summary(self) -> Dict[str, Dict[str, int]]:
        """
        Return counts of successes, failures, and skips per engine since the
        last call to run_all_scenarios().
        """
        return {
            name: {
                "successes": self._successes[name],
                "failures":  self._failures[name],
                "skipped":   self._skipped[name],
            }
            for name in self._ENGINE_NAMES
        }
