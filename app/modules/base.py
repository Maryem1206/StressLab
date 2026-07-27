from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

@dataclass
class PlatformResult:
    module_id: str
    module_label: str
    scenario_id: str         # "baseline" | "adverse" | "severe"
    total_loss: float
    kpis: Dict[str, Any]
    time_series: pd.DataFrame
    charts_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        ts = self.time_series.copy()
        if "year" in ts.columns:
            ts = ts.set_index("year")
        ts.index = ts.index.astype(str)
        return {
            "module_id": self.module_id, "module_label": self.module_label,
            "scenario_id": self.scenario_id, "total_loss": self.total_loss,
            "kpis": self.kpis, "time_series": ts.to_dict(orient="index"),
            "charts_data": {k: v for k, v in self.charts_data.items()
                            if not isinstance(v, pd.DataFrame)},
            "metadata": self.metadata, "warnings": self.warnings,
        }

@dataclass
class ProjectionSeries:
    """
    Uniform time-series shape for the climate dashboard's Row 3/Row 5 charts
    (Crédit PD, Liquidité LCR/NSFR, ratios réglementaires CAR/Tier1, sVaR/sES):
    one historical series + one per scenario tier (baseline/adverse/severe).

    Additive only — introduced for the dashboard redesign, does not replace
    or modify any existing charts_data structure. get_projection_series()
    helpers (credit/liquidity/wrapper.py, climate/macro_delta.py) adapt
    already-computed trajectories into this shape; they do not recompute
    anything.
    """
    variable_name: str
    historical_dates: List[Any] = field(default_factory=list)
    historical_values: List[Optional[float]] = field(default_factory=list)
    baseline_dates: List[Any] = field(default_factory=list)
    baseline_values: List[Optional[float]] = field(default_factory=list)
    adverse_dates: List[Any] = field(default_factory=list)
    adverse_values: List[Optional[float]] = field(default_factory=list)
    severe_dates: List[Any] = field(default_factory=list)
    severe_values: List[Optional[float]] = field(default_factory=list)
    unit: str = ""
    historical_available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable_name": self.variable_name,
            "historical_dates": self.historical_dates,
            "historical_values": self.historical_values,
            "baseline_dates": self.baseline_dates,
            "baseline_values": self.baseline_values,
            "adverse_dates": self.adverse_dates,
            "adverse_values": self.adverse_values,
            "severe_dates": self.severe_dates,
            "severe_values": self.severe_values,
            "unit": self.unit,
            "historical_available": self.historical_available,
        }


def projection_series_from_trajectory_dict(
    traj: Dict[str, Any],
    variable_name: str,
    unit: str = "",
) -> Optional[ProjectionSeries]:
    """
    Adapts the platform's existing {"historical"/"baseline"/"adverse"/
    "severe": {"x": [...], "y": [...]}} trajectory dict convention (already
    used by credit's pd_trajectories/capital_trajectories and liquidity's
    lcr_trajectories/nsfr_trajectories) into a ProjectionSeries. Pure
    adapter — reads already-computed data, computes nothing new.

    Returns None if traj has no usable data for any of the 4 tiers.
    """
    if not traj:
        return None

    def _xy(level: str):
        d = traj.get(level) or {}
        return list(d.get("x", []) or []), list(d.get("y", []) or [])

    hist_x, hist_y = _xy("historical")
    base_x, base_y = _xy("baseline")
    adv_x, adv_y = _xy("adverse")
    sev_x, sev_y = _xy("severe")

    if not any([hist_x, base_x, adv_x, sev_x]):
        return None

    return ProjectionSeries(
        variable_name=variable_name,
        historical_dates=hist_x, historical_values=hist_y,
        baseline_dates=base_x, baseline_values=base_y,
        adverse_dates=adv_x, adverse_values=adv_y,
        severe_dates=sev_x, severe_values=sev_y,
        unit=unit,
        historical_available=bool(hist_x),
    )


class PlatformModule(ABC):
    module_id: str = "unknown"
    module_label: str = "Unknown"
    is_placeholder: bool = False

    def __init__(self, upload_paths: Dict, params: Dict, scenario: Dict):
        self.upload_paths = upload_paths
        self.params = params
        self.scenario = scenario   # always has .levels: {baseline, adverse, severe}

    @abstractmethod
    def run(self) -> Dict[str, PlatformResult]:
        """Always returns dict with keys: baseline, adverse, severe."""
        ...

    @abstractmethod
    def get_kpis(self, results: Dict) -> Dict[str, Any]: ...
