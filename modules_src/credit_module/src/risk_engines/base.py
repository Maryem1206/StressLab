
from abc import ABC, abstractmethod
class BaseRiskEngine(ABC):
    risk_type = "unknown"
    def __init__(self, config, assumptions):
        self.config = config
        self.assumptions = assumptions
    @abstractmethod
    def run(self, scenario, projected_params=None): ...
