"""macro_selection_engine — Two-stage variable selection for stress testing."""

from .main import run_engine
from .multi_scenario import run_three_scenarios
from .utils import EngineConfig, ECONOMIC_PRIORS, expected_sign
from .baseline_projector import (
    project_baseline,
    BaselineProjection,
    summarize_projection,
    summarize_fallbacks,
)
from .variable_resolver import resolve_variables, ResolverResult, print_resolution_log
from .model_stress_validator import (
    validate_models,
    StressValidationReport,
    ModelValidationResult,
    get_stress_scores_dict,
)
from .scenario_family_filter import filter_scenario_families

__all__ = [
    "run_engine",
    "run_three_scenarios",
    "EngineConfig",
    "ECONOMIC_PRIORS",
    "expected_sign",
    "project_baseline",
    "BaselineProjection",
    "summarize_projection",
    "summarize_fallbacks",
    "resolve_variables",
    "ResolverResult",
    "print_resolution_log",
    "validate_models",
    "StressValidationReport",
    "ModelValidationResult",
    "get_stress_scores_dict",
    "filter_scenario_families",
]
