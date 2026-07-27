"""Macro Transmission Layer (Module 2) — credit calibration orchestrator.

Calibrates one PD satellite model per declared credit segment via a full
variable-selection tournament (3 families x adaptive combinations), and
calibrates one Frye-Jacobs LGD per segment.

Each segment must declare in ``config/models.yaml > credit > segments``:
- ``target``                 : default-rate column in risk_history
- ``candidate_variables``    : macro variables to test
- ``expected_signs``         : economic sign per variable (+1 / -1 / 0)
- ``pd_floor`` / ``pd_cap``  : prediction caps
- ``lgd_ttc_source``         : 'empirical' or 'regulatory' (default: 'regulatory')
- ``lgd_ttc``                : regulatory anchor used when source='regulatory'
                               or as fallback when empirical history is missing
- ``lgd_history_column``     : (optional) override for the historical LGD column
                               name used when source='empirical'.
                               Default: ``lgd_<segment_name>``.

LGD_TTC resolution rules
========================
- ``regulatory`` (default): use ``lgd_ttc`` from config. Standard Basel III
  IRB-Foundation values are 45 % for senior unsecured corporate / sovereign /
  banks, and 75 % for subordinated, per BCBS (2017) "Basel III: Finalising
  post-crisis reforms". For retail there is no F-IRB default; institutions
  must use their own A-IRB estimates or supervisory floors.
- ``empirical``: compute the historical mean of the segment's LGD-PIT series
  ``lgd_history_column`` (defaults to ``lgd_<segment>``) over the calibration
  window. This matches Frye & Jacobs (2012) practice of using ``ELGD = EL/PD``
  as the long-run mean. If the column is missing or empty the calibrator
  falls back to ``lgd_ttc`` and logs a warning.

Application to a scenario macro path returns a DataFrame indexed by year
with one PD and one LGD column per segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..core.logger import get_logger
from .frye_jacobs_lgd import FryeJacobsLGD
from .pd_models import _BasePDModel
from .variable_selection import (
    SelectionResult,
    StandardScaler1D,
    VariableSelector,
)

LOG = get_logger()


# ---------------------------------------------------------------------------
# Calibrated segment container
# ---------------------------------------------------------------------------

@dataclass
class CalibratedSegment:
    """Holds the calibrated PD model + FJ-LGD for one segment."""
    name: str                      # logical segment name (e.g. "corporate", "retail")
    target: str                    # historical default-rate column (e.g. "pd_corporate")
    pd_model: _BasePDModel         # winning PD satellite model
    pd_features: List[str]         # selected features (subset of candidates)
    pd_family: str                 # "beta" / "logit" / "vasicek"
    scaler: StandardScaler1D | None
    pd_floor: float
    pd_cap: float
    pd_ttc: float                  # mean of historical default rate
    lgd_ttc: float                 # resolved via regulatory or empirical source
    lgd_ttc_source: str            # 'regulatory' or 'empirical'
    lgd_ttc_n_obs: int             # number of historical LGD obs used (0 if regulatory)
    fj_lgd: FryeJacobsLGD          # calibrated FJ-LGD
    selection_summary: Dict[str, Any]   # leaderboard top, n_obs, etc.

    # ------------------------------------------------------------------ predict
    def predict_pd(self, scenario_macro: pd.DataFrame) -> pd.Series:
        """PIT PD trajectory under a scenario macro path."""
        X = scenario_macro[self.pd_features].astype(float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        pd_pit = self.pd_model.predict(X)
        return pd_pit.clip(lower=self.pd_floor, upper=self.pd_cap)

    def predict_lgd(self, pd_pit: pd.Series) -> pd.Series:
        return self.fj_lgd.predict(pd_pit)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "pd_family": self.pd_family,
            "pd_features": list(self.pd_features),
            "pd_floor": float(self.pd_floor),
            "pd_cap": float(self.pd_cap),
            "pd_ttc": float(self.pd_ttc),
            "lgd_ttc": float(self.lgd_ttc),
            "lgd_ttc_source": self.lgd_ttc_source,
            "lgd_ttc_n_obs": int(self.lgd_ttc_n_obs),
            "pd_model": self.pd_model.to_dict(),
            "scaler": self.scaler.to_dict() if self.scaler is not None else None,
            "fj_lgd": self.fj_lgd.to_dict(),
            "selection_summary": self.selection_summary,
        }


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class CreditCalibrator:
    """Calibrate one PD model + one FJ-LGD per credit segment.

    The configuration is read from ``config/models.yaml`` under the
    ``credit`` block. See module docstring for required fields.
    """

    DEFAULT_REGULATORY_LGD: float = 0.45  # Basel III F-IRB senior unsecured

    def __init__(self, models_config: dict) -> None:
        self.cfg = models_config
        credit_cfg = models_config.get("credit", {})
        if not credit_cfg:
            raise ValueError("models.yaml is missing a 'credit' block")
        self.credit_cfg = credit_cfg

        vs_cfg = credit_cfg.get("variable_selection", {})
        self.families = list(vs_cfg.get("families", ["beta", "logit", "vasicek"]))
        self.standardize = bool(vs_cfg.get("standardize", True))
        self.user_max_combo = vs_cfg.get("max_combo_size", None)

        cal_cfg = models_config.get("calibration", {})
        self.train_start = cal_cfg.get("train_start", None)
        self.train_end = cal_cfg.get("train_end", None)

        self.segments_cfg = credit_cfg["segments"]

    # ----------------------------------------------------------- public API

    def calibrate_all(self, history: pd.DataFrame) -> Dict[str, CalibratedSegment]:
        """Run the full calibration pipeline.

        ``history`` is a DataFrame indexed by year containing macro columns +
        per-segment default-rate columns + (optionally) historical LGD columns
        when ``lgd_ttc_source: empirical`` is requested for any segment.
        """
        sub = history.copy()
        if self.train_start is not None:
            sub = sub.loc[sub.index >= self.train_start]
        if self.train_end is not None:
            sub = sub.loc[sub.index <= self.train_end]

        out: Dict[str, CalibratedSegment] = {}
        for seg_name, seg_cfg in self.segments_cfg.items():
            out[seg_name] = self._calibrate_segment(seg_name, seg_cfg, sub)
        return out

    # ----------------------------------------------------------- internal

    def _calibrate_segment(self, name: str, cfg: dict,
                           history: pd.DataFrame) -> CalibratedSegment:
        target = cfg["target"]
        if target not in history.columns:
            raise ValueError(f"Segment '{name}': target '{target}' missing in history")

        pd_floor = float(cfg.get("pd_floor", 1e-4))
        pd_cap = float(cfg.get("pd_cap", 0.50))
        candidates = list(cfg["candidate_variables"])
        expected = {k: int(v) for k, v in cfg.get("expected_signs", {}).items()}
        lgd_floor = float(cfg.get("lgd_floor", 0.05))
        lgd_cap = float(cfg.get("lgd_cap", 0.95))

        # 1. Variable selection tournament
        selector = VariableSelector(
            candidate_variables=candidates,
            expected_signs=expected,
            families=self.families,
            standardize=self.standardize,
            max_combo_size=self.user_max_combo,
        )
        sel: SelectionResult = selector.select(history, target=target)

        # 2. PD_TTC = mean of historical default rate
        pd_series = history[target].dropna().astype(float)
        pd_ttc = float(np.clip(pd_series.mean(), 1e-6, 1 - 1e-6))

        # 3. LGD_TTC resolution (regulatory vs empirical)
        lgd_ttc, lgd_source, lgd_n = self._resolve_lgd_ttc(name, cfg, history)

        # 4. Frye-Jacobs LGD calibrated on (PD_TTC, LGD_TTC)
        fj = FryeJacobsLGD.calibrate(
            pd_ttc=pd_ttc, lgd_ttc=lgd_ttc, floor=lgd_floor, cap=lgd_cap,
        )

        LOG.info(
            f"[credit_calibration] segment={name} target={target} "
            f"PD_TTC={pd_ttc:.4f} LGD_TTC={lgd_ttc:.4f} (source={lgd_source}, "
            f"n={lgd_n}) rho={fj.rho:.4f} k={fj.k:.4f}"
        )

        return CalibratedSegment(
            name=name,
            target=target,
            pd_model=sel.best_model,
            pd_features=list(sel.best_features),
            pd_family=sel.best_family,
            scaler=sel.scaler,
            pd_floor=pd_floor,
            pd_cap=pd_cap,
            pd_ttc=pd_ttc,
            lgd_ttc=lgd_ttc,
            lgd_ttc_source=lgd_source,
            lgd_ttc_n_obs=lgd_n,
            fj_lgd=fj,
            selection_summary={
                "n_obs": int(sel.n_obs),
                "max_combo_size": int(sel.max_combo_size),
                "n_candidates": len(sel.candidate_variables),
                "leaderboard_top10": sel.leaderboard.head(10).to_dict(orient="records"),
            },
        )

    # ----------------------------------------------------------- LGD_TTC

    def _resolve_lgd_ttc(self, segment: str, cfg: dict,
                         history: pd.DataFrame) -> tuple[float, str, int]:
        """Resolve LGD_TTC for a segment.

        Returns (lgd_ttc, source, n_obs).

        - source='empirical': compute historical mean from
          ``cfg['lgd_history_column']`` (default: ``lgd_<segment>``).
        - source='regulatory' (default): use ``cfg['lgd_ttc']`` directly.
        - If 'empirical' is requested but the column is missing/empty,
          fall back to 'regulatory' and emit a warning.
        """
        source = str(cfg.get("lgd_ttc_source", "regulatory")).lower()
        regulatory_value = float(cfg.get("lgd_ttc", self.DEFAULT_REGULATORY_LGD))
        regulatory_value = float(np.clip(regulatory_value, 1e-4, 1 - 1e-4))

        if source == "empirical":
            col = cfg.get("lgd_history_column", f"lgd_{segment}")
            if col in history.columns:
                lgd_series = history[col].dropna().astype(float)
                lgd_series = lgd_series[(lgd_series > 0) & (lgd_series < 1)]
                if len(lgd_series) > 0:
                    lgd_emp = float(np.clip(lgd_series.mean(), 1e-4, 1 - 1e-4))
                    LOG.info(
                        f"[credit_calibration] segment={segment} "
                        f"LGD_TTC empirical mean = {lgd_emp:.4f} "
                        f"(n={len(lgd_series)} from column '{col}')"
                    )
                    return lgd_emp, "empirical", int(len(lgd_series))
                LOG.warning(
                    f"[credit_calibration] segment={segment} "
                    f"requested empirical LGD_TTC but column '{col}' is empty "
                    f"after filtering. Falling back to regulatory "
                    f"value {regulatory_value:.4f}."
                )
            else:
                LOG.warning(
                    f"[credit_calibration] segment={segment} "
                    f"requested empirical LGD_TTC but column '{col}' not in "
                    f"history. Falling back to regulatory value "
                    f"{regulatory_value:.4f}."
                )
            return regulatory_value, "regulatory_fallback", 0

        if source != "regulatory":
            LOG.warning(
                f"[credit_calibration] segment={segment} unknown "
                f"lgd_ttc_source='{source}', defaulting to 'regulatory'."
            )
        return regulatory_value, "regulatory", 0


# ---------------------------------------------------------------------------
# Project to scenario
# ---------------------------------------------------------------------------

def project_credit_parameters(scenario_macro: pd.DataFrame,
                              segments: Dict[str, CalibratedSegment]) -> pd.DataFrame:
    """Apply each calibrated segment to a scenario macro path.

    Returns a DataFrame indexed by year with columns:
        pd_<segment_name>, lgd_<segment_name>
    for every segment.
    """
    out = pd.DataFrame(index=scenario_macro.index)
    for name, seg in segments.items():
        try:
            pd_pit = seg.predict_pd(scenario_macro)
        except KeyError as exc:
            raise ValueError(
                f"Segment '{name}': scenario macro is missing one of the "
                f"selected features {seg.pd_features}: {exc}"
            ) from exc
        lgd_pit = seg.predict_lgd(pd_pit)
        out[f"pd_{name}"] = pd_pit.values
        out[f"lgd_{name}"] = lgd_pit.values
    return out
