"""
main.py
=======
End-to-end orchestration of the Macro Variable Selection Engine.

Two stages:
    Stage 1 -> NGFS-only filtering (no target).
    Stage 2 -> Historical statistical validation (with target).

Final output: a structured JSON-serializable dict consumable by the platform.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .data_loader import load_historical, load_ngfs
from .geo_selector import select_geography
from .historical_validation import Stage2Result, run_stage2
from .model_estimation import FittedModel
from .ngfs_filtering import Stage1Result, stage1_filter
from .utils import EngineConfig, get_logger

log = get_logger(__name__)


# -----------------------------------------------------------------------------
# JSON-safe helpers
# -----------------------------------------------------------------------------
def _df_to_json(df: pd.DataFrame, max_rows: int = 200) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    return df.head(max_rows).replace({np.nan: None}).to_dict(orient="records")


def _model_to_json(m: FittedModel) -> Dict[str, Any]:
    def _clean(d: Dict[str, float]) -> Dict[str, float]:
        return {k: (None if (isinstance(v, float) and np.isnan(v)) else float(v))
                for k, v in d.items()}
    return {
        "name": m.name,
        "family": m.family,
        "variables": list(m.variables),
        "coefficients": _clean(m.coefficients),
        "std_errors": _clean(m.std_errors),
        "pvalues": _clean(m.pvalues),
        "n_obs": int(m.n_obs),
        "aic": (None if np.isnan(m.aic) else float(m.aic)),
        "bic": (None if np.isnan(m.bic) else float(m.bic)),
        "r2_or_pseudo": (None if np.isnan(m.r2) else float(m.r2)),
        "converged": bool(m.converged),
    }


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def run_engine(cfg: EngineConfig) -> Dict[str, Any]:
    """Run the full two-stage pipeline and return a structured result dict."""
    log.info("Engine starting | country=%s | scenario=%s | channel=%s | target=%s",
             cfg.country, cfg.scenario, cfg.risk_channel, cfg.target_variable)

    # -------- Load --------
    ngfs_long = load_ngfs(cfg)
    ngfs_country, resolved_region = select_geography(ngfs_long, cfg)

    # -------- Stage 1 --------
    s1: Stage1Result = stage1_filter(ngfs_country, cfg)
    if not s1.selected_variables:
        raise RuntimeError("Stage 1 selected no variables — relax filters.")

    # -------- Stage 2 --------
    hist_df = load_historical(cfg)
    s2: Stage2Result = run_stage2(s1.selected_variables, hist_df, cfg)

    # -------- Top-N + best --------
    top_models_df = s2.ranking_table.head(cfg.top_n_models)
    top_names = set(top_models_df["model"].tolist()) if not top_models_df.empty else set()
    top_models = [m for m in s2.fitted_models if m.name in top_names]
    # preserve ranking order
    rank_index = {n: i for i, n in enumerate(top_models_df["model"].tolist())}
    top_models.sort(key=lambda m: rank_index.get(m.name, 1e9))

    best_model = top_models[0] if top_models else None

    # -------- JSON-serializable output --------
    output: Dict[str, Any] = {
        "metadata": {
            "country": cfg.country,
            "resolved_region": resolved_region,
            "scenario": cfg.scenario,
            "baseline_scenario": cfg.baseline_scenario,
            "risk_channel": cfg.risk_channel,
            "risk_type": cfg.risk_type,
            "target_variable": cfg.target_variable,
        },
        "stage1_ngfs": {
            "selected_variables": s1.selected_variables,
            "n_selected": len(s1.selected_variables),
            "filters": {
                "variability_top_pct": cfg.s1_keep_top_variance_pct,
                "min_shock_magnitude": cfg.s1_min_shock_magnitude,
                "economic_whitelist": cfg.s1_apply_economic_whitelist,
            },
            "diagnostics": {
                "variability_table": _df_to_json(s1.variability_table.reset_index()
                                                 .rename(columns={"index": "variable"})
                                                 if s1.variability_table is not None else pd.DataFrame()),
                "shock_table": _df_to_json(s1.shock_table.reset_index()
                                            .rename(columns={"index": "variable"})
                                            if s1.shock_table is not None else pd.DataFrame()),
                "economic_relevance_table": _df_to_json(
                    s1.economic_relevance_table.reset_index()
                    if s1.economic_relevance_table is not None else pd.DataFrame()),
            },
        },
        "stage2_historical": {
            "mapping": {
                "ngfs_to_historical": s2.mapping.matches if s2.mapping else {},
                "method_per_match": s2.mapping.method_per_match if s2.mapping else {},
                "unmatched_dropped": s2.mapping.unmatched if s2.mapping else [],
            },
            "correlation_with_target": _df_to_json(s2.correlation_table),
            "vif_removed": s2.vif_removed,
            "vif_final": _df_to_json(s2.vif_final),
            "backward_kept": s2.backward_kept,
            "backward_trace": _df_to_json(s2.backward_trace),
        },
        "ranking_table": _df_to_json(s2.ranking_table, max_rows=50),
        "selected_models": [_model_to_json(m) for m in top_models],
        "best_model": _model_to_json(best_model) if best_model else None,
        "diagnostics": {
            "correlation_matrix": (s2.correlation_matrix
                                   .replace({np.nan: None})
                                   .to_dict()),
        },
    }

    # Persist
    if cfg.output_path:
        with open(cfg.output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        log.info("Wrote engine output: %s", cfg.output_path)

    return output
