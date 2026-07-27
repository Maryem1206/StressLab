"""Credit Risk Engine — academic pipeline.

Pipeline (per credit segment declared in config):
    macro path     -> selected satellite model (Beta / Logit / Vasicek)  -> PD(t)
    PD(t)          -> Frye-Jacobs LGD (calibrated on (PD_TTC, LGD_TTC))   -> LGD(t)
    EAD            -> user input from assumptions.yaml > portfolio        -> EAD(t)
    Expected Loss  : EL_seg(t) = PD_seg(t) * LGD_seg(t) * EAD_seg(t)

Returns a standardized ``RiskOutput`` with:
    loss              : sum of ELs across years and segments
    metrics           : peak EL, average PD/LGD per segment, total EL per segment
    time_series       : per-year loss + per-segment columns + risk parameter trajectories

EAD is **not** modeled: it is provided directly by the user. The engine accepts
EAD in three different shapes and resolves them uniformly per segment per year:

  1. **Scalar** (single number)
        portfolio:
          corporate:
            ead: 100_000_000_000      # held flat across the horizon

  2. **Year-indexed mapping** (explicit time-varying schedule)
        portfolio:
          corporate:
            ead:
              2025: 100_000_000_000
              2026: 105_000_000_000
              2027: 110_000_000_000

  3. **External file** (CSV or Excel) with columns ``year`` and ``ead``
        portfolio:
          corporate:
            ead_file: data/portfolio/ead_corporate.csv
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from ..core.data_structures import RiskOutput, Scenario
from ..core.logger import get_logger
from .base import BaseRiskEngine

LOG = get_logger()


class CreditRiskEngine(BaseRiskEngine):
    risk_type = "credit"

    def run(self, scenario: Scenario,
            projected_params: pd.DataFrame | None = None) -> RiskOutput:
        if projected_params is None:
            raise ValueError("CreditRiskEngine requires projected_params from "
                             "project_credit_parameters()")

        years = scenario.macro.index
        portfolio_cfg = self.assumptions["portfolio"]

        seg_names = self._discover_segments(projected_params)
        ead_paths = self._build_ead_paths(seg_names, portfolio_cfg, years)

        ts = pd.DataFrame(index=years)
        per_segment_totals: Dict[str, float] = {}
        per_segment_avg_pd: Dict[str, float] = {}
        per_segment_avg_lgd: Dict[str, float] = {}
        per_segment_avg_ead: Dict[str, float] = {}

        for seg_name in seg_names:
            pd_col = f"pd_{seg_name}"
            lgd_col = f"lgd_{seg_name}"
            if pd_col not in projected_params.columns or lgd_col not in projected_params.columns:
                raise ValueError(
                    f"Segment '{seg_name}': projected params missing columns "
                    f"{pd_col} / {lgd_col}. Make sure models.yaml declares this "
                    f"segment under credit.segments and that calibration ran."
                )
            pd_pit = projected_params[pd_col].reindex(years).astype(float).values
            lgd_pit = projected_params[lgd_col].reindex(years).astype(float).values
            ead_pit = ead_paths[seg_name].reindex(years).astype(float).values

            el = pd_pit * lgd_pit * ead_pit
            ts[f"el_{seg_name}"] = el
            ts[f"pd_{seg_name}"] = pd_pit
            ts[f"lgd_{seg_name}"] = lgd_pit
            ts[f"ead_{seg_name}"] = ead_pit

            per_segment_totals[seg_name] = float(np.sum(el))
            per_segment_avg_pd[seg_name] = float(np.mean(pd_pit))
            per_segment_avg_lgd[seg_name] = float(np.mean(lgd_pit))
            per_segment_avg_ead[seg_name] = float(np.mean(ead_pit))

        # Universal "loss" column required by the platform contract
        loss_cols = [c for c in ts.columns if c.startswith("el_")]
        ts["loss"] = ts[loss_cols].sum(axis=1).values

        total = float(ts["loss"].sum())
        peak = float(ts["loss"].max()) if len(ts) else 0.0

        metrics: Dict[str, float] = {
            "total_loss": total,
            "peak_annual_loss": peak,
        }
        for seg in seg_names:
            metrics[f"total_loss_{seg}"] = per_segment_totals[seg]
            metrics[f"avg_pd_{seg}"] = per_segment_avg_pd[seg]
            metrics[f"avg_lgd_{seg}"] = per_segment_avg_lgd[seg]
            metrics[f"avg_ead_{seg}"] = per_segment_avg_ead[seg]

        meta = {
            "segmentation": list(seg_names),
            "model": "PD (selected satellite) x LGD (Frye-Jacobs) x EAD (user input)",
            "ead_resolution": {seg: self._describe_ead_path(ead_paths[seg])
                               for seg in seg_names},
        }
        LOG.info(f"[credit] scenario={scenario.scenario_id} total EL={total:,.0f} "
                 f"segments={list(seg_names)}")
        return RiskOutput(
            risk_type=self.risk_type,
            scenario_id=scenario.scenario_id,
            loss=total,
            metrics=metrics,
            time_series=ts,
            metadata=meta,
        )

    # ================================================================ helpers

    @staticmethod
    def _discover_segments(projected_params: pd.DataFrame) -> List[str]:
        """Return segment names present as columns ``pd_<seg>``."""
        return [c[len("pd_"):] for c in projected_params.columns if c.startswith("pd_")]

    def _build_ead_paths(self,
                         seg_names: List[str],
                         portfolio_cfg: Mapping,
                         years) -> Dict[str, pd.Series]:
        """Build a per-segment EAD time-series indexed by ``years``.

        Resolution rules per segment, applied in order:
          1. ``portfolio[seg]['ead']``       — scalar OR mapping {year: amount}
          2. ``portfolio[seg]['ead_file']``  — path to CSV/Excel
          3. ``portfolio[seg]['ead_initial']`` (legacy compat) — scalar
          4. ``portfolio['sectors'][seg]`` with same keys (legacy nested layout)
        """
        out: Dict[str, pd.Series] = {}
        for seg in seg_names:
            ead_value = self._lookup_ead_entry(seg, portfolio_cfg)
            if ead_value is None:
                raise ValueError(
                    f"Segment '{seg}' is calibrated but no EAD found in "
                    f"assumptions.yaml > portfolio. Add a section "
                    f"`{seg}: {{ead: <amount>}}` (scalar), "
                    f"`ead: {{<year>: <amount>, ...}}` (time-varying), "
                    f"or `ead_file: <path.csv|.xlsx>`."
                )
            out[seg] = self._materialize_ead(seg, ead_value, years)
        return out

    @staticmethod
    def _lookup_ead_entry(segment: str, portfolio_cfg: Mapping):
        """Locate the raw EAD entry for a segment.

        Returns one of:
          - a scalar number  (cas A: held flat)
          - a dict {year:int -> value}  (cas B: explicit schedule)
          - a dict {'ead_file': path}  (cas C: external file)
          - None if not found.
        """
        if segment in portfolio_cfg and isinstance(portfolio_cfg[segment], dict):
            d = portfolio_cfg[segment]
            if "ead_file" in d:
                return {"ead_file": d["ead_file"]}
            if "ead" in d:
                return d["ead"]
            if "ead_initial" in d:
                return d["ead_initial"]

        sectors = portfolio_cfg.get("sectors", {})
        if isinstance(sectors, Mapping) and segment in sectors and isinstance(sectors[segment], dict):
            d = sectors[segment]
            if "ead_file" in d:
                return {"ead_file": d["ead_file"]}
            if "ead" in d:
                return d["ead"]
            if "ead_initial" in d:
                return d["ead_initial"]
        return None

    def _materialize_ead(self, segment: str, ead_value: Any, years) -> pd.Series:
        """Convert a raw EAD entry into a year-indexed Series."""
        # Cas A — scalar (constant across horizon)
        if isinstance(ead_value, (int, float)) and not isinstance(ead_value, bool):
            return pd.Series(float(ead_value), index=years, name=f"ead_{segment}")

        # Cas C — external file
        if isinstance(ead_value, dict) and "ead_file" in ead_value:
            return self._read_ead_file(segment, ead_value["ead_file"], years)

        # Cas B — year-indexed mapping
        if isinstance(ead_value, dict):
            try:
                int_keyed = {int(k): float(v) for k, v in ead_value.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Segment '{segment}': EAD mapping must be {{year:int -> amount}}, "
                    f"got {ead_value!r}"
                ) from exc
            return self._reindex_with_ffill(segment, int_keyed, years)

        raise ValueError(f"Segment '{segment}': invalid EAD type {type(ead_value).__name__}")

    @staticmethod
    def _read_ead_file(segment: str, path: str, years) -> pd.Series:
        """Read EAD time-series from CSV or Excel.

        Required columns: ``year``, ``ead``. Optionally a ``segment`` column to
        filter rows when the same file holds multiple segments.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Segment '{segment}': EAD file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif ext in (".csv", ".tsv", ".txt"):
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        else:
            raise ValueError(f"Segment '{segment}': unsupported EAD file extension {ext}")

        df.columns = [str(c).strip().lower() for c in df.columns]
        if "year" not in df.columns or "ead" not in df.columns:
            raise ValueError(
                f"Segment '{segment}': EAD file {path} must have columns "
                f"['year','ead']. Found: {list(df.columns)}"
            )
        if "segment" in df.columns:
            df = df[df["segment"].astype(str).str.lower() == segment.lower()]
            if df.empty:
                raise ValueError(
                    f"Segment '{segment}': no rows matched in {path} "
                    f"(filtered on segment column)."
                )

        mapping = {int(y): float(v) for y, v in zip(df["year"], df["ead"])
                   if pd.notna(y) and pd.notna(v)}
        return CreditRiskEngine._reindex_with_ffill(segment, mapping, years)

    @staticmethod
    def _reindex_with_ffill(segment: str, mapping: Dict[int, float], years) -> pd.Series:
        """Reindex an int-keyed mapping onto the scenario horizon.

        Forward-fill missing years with the last known value; back-fill leading
        years with the first known value. Raises if mapping is empty.
        """
        if not mapping:
            raise ValueError(f"Segment '{segment}': empty EAD mapping.")
        s = pd.Series(mapping).sort_index()
        s.index = s.index.astype(int)
        target = pd.Index([int(y) for y in years], name="year")
        out = s.reindex(target).ffill().bfill()
        out.name = f"ead_{segment}"
        return out

    @staticmethod
    def _describe_ead_path(s: pd.Series) -> Dict[str, float]:
        return {
            "min": float(s.min()),
            "max": float(s.max()),
            "first": float(s.iloc[0]),
            "last": float(s.iloc[-1]),
            "constant": bool(np.isclose(s.min(), s.max())),
        }
