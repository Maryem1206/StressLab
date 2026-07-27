
from dataclasses import dataclass, field
from typing import Any, Dict
import pandas as pd

@dataclass
class Scenario:
    scenario_id: str
    macro: pd.DataFrame
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskOutput:
    risk_type: str
    scenario_id: str
    loss: float
    metrics: Dict[str, float]
    time_series: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
