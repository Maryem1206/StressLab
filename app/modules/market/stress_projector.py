"""
stress_projector.py
===================
Project stressed Nelson-Siegel betas under macro shock scenarios.

The scenario engine (scenario_manager.py) delivers additive shocks
(gdp_delta, rate_bps, inflation_delta, fx_pct, ...).  These are
converted to absolute macro variable levels and injected into the
AR(1)+macro satellite to obtain stressed betas at the end of the
projection horizon.

References
----------
Diebold, F.X. & Li, C. (2006). Forecasting the term structure of
    government bond yields. Journal of Econometrics, 130(2), 337-364.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from .ir_satellite import BetaSatellite

LOG = logging.getLogger("market.stress_projector")

# Mapping: scenario shock key -> macro variable name in macro_df
# rate_bps is in basis points -> converted to pp by SHOCK_CONV
SHOCK_TO_MACRO: Dict[str, str] = {
    "rate_bps":        "policy_rate",
    "inflation_delta": "cpi_inflation",
    "gdp_delta":       "real_gdp_growth",
}

# Additional variables that must receive the same rate shock as policy_rate.
# lending_rate tracks the CBE rate cycle closely and may be selected by the
# satellite when FRED monthly data is unavailable.
_RATE_LINKED_VARS: tuple = ("lending_rate", "policy_rate_monthly")

# Unit conversion multipliers (shock_value * conv -> delta in native units of macro var)
SHOCK_CONV: Dict[str, float] = {
    "rate_bps": 0.01,   # bp -> percentage points
}


class StressProjector:
    """Project beta1, beta2, beta3 under a single scenario shock dict.

    Parameters
    ----------
    satellites : dict
        {k: BetaSatellite} from IRSatellite.fit_all().
    last_betas : pd.Series
        Last observed betas {beta1, beta2, beta3} (from betas_df.iloc[-1]).
    last_macro : pd.Series
        Last observed macro values indexed by variable name.

    Usage
    -----
    projector = StressProjector(satellites, last_betas, last_macro)
    stressed  = projector.project({"rate_bps": 250, "gdp_delta": -0.03},
                                  horizon_months=12)
    # stressed = {"beta1": float, "beta2": float, "beta3": float}
    """

    def __init__(
        self,
        satellites: Dict[int, BetaSatellite],
        last_betas: pd.Series,
        last_macro: pd.Series,
    ) -> None:
        self.satellites = satellites
        self.last_betas = last_betas
        self.last_macro = last_macro

    def project(
        self,
        scenario_shocks: Dict[str, float],
        horizon_months: int = 12,
    ) -> Dict[str, float]:
        """Return stressed betas after `horizon_months` of projection.

        Returns
        -------
        dict {"beta1": float, "beta2": float, "beta3": float}
        """
        path = self.project_path(scenario_shocks, horizon_months)
        result = {f"beta{k}": path[f"beta{k}"][-1] for k in (1, 2, 3)}
        LOG.info(
            "StressProjector: beta1=%.4f  beta2=%.4f  beta3=%.4f  "
            "(horizon=%d mo, shocks=%s)",
            result["beta1"], result["beta2"], result["beta3"],
            horizon_months, scenario_shocks,
        )
        return result

    def project_path(
        self,
        scenario_shocks: Dict[str, float],
        horizon_months: int = 12,
    ) -> Dict[str, list]:
        """Return the full monthly beta trajectory over `horizon_months`.

        Returns
        -------
        dict {
            "beta1": [v_t1, v_t2, ..., v_t12],
            "beta2": [...],
            "beta3": [...],
        }
        Each list has exactly `horizon_months` entries (T+1 through T+horizon).
        """
        stressed_macro = self._build_stressed_macro(scenario_shocks)
        paths: Dict[str, list] = {f"beta{k}": [] for k in (1, 2, 3)}

        for k in (1, 2, 3):
            sat = self.satellites[k]
            beta_prev = float(self.last_betas.get(f"beta{k}", 0.0))
            for _ in range(max(1, int(horizon_months))):
                beta_prev = sat.predict_next(beta_prev, stressed_macro)
                paths[f"beta{k}"].append(beta_prev)

        return paths

    def _build_stressed_macro(
        self, shocks: Dict[str, float]
    ) -> Dict[str, float]:
        """Construct the stressed macro dict = last observed + deltas."""
        macro_vals = {k: float(v) for k, v in self.last_macro.items()
                      if pd.notna(v)}

        for shock_key, var_name in SHOCK_TO_MACRO.items():
            if shock_key in shocks:
                delta = shocks[shock_key] * SHOCK_CONV.get(shock_key, 1.0)
                macro_vals[var_name] = macro_vals.get(var_name, 0.0) + delta

        # All rate-linked variables (lending_rate, policy_rate_monthly) must
        # receive the same delta as policy_rate when a rate_bps shock is given.
        if "rate_bps" in shocks:
            rate_delta = shocks["rate_bps"] * SHOCK_CONV["rate_bps"]
            for var in _RATE_LINKED_VARS:
                if var in macro_vals:
                    macro_vals[var] += rate_delta
                elif var == "policy_rate_monthly":
                    macro_vals[var] = macro_vals.get("policy_rate", 0.0)

        return macro_vals