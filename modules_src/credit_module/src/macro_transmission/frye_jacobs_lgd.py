"""Frye-Jacobs systematic LGD model.

Reference
---------
Frye, J. and Jacobs Jr., M. (2012).
"Credit Loss and Systematic Loss Given Default",
The Journal of Credit Risk 8(1), pp. 1-32.

Equation (1) of the paper, expressed for a PIT default rate:

    LGD(PD) = Phi[ (Phi^{-1}(PD) - k) / sqrt(1 - rho) ] / PD

with the calibration constant

    k = Phi^{-1}(PD_TTC) - Phi^{-1}(PD_TTC * LGD_TTC)

ensuring LGD(PD_TTC) = LGD_TTC at the through-the-cycle anchor point.

The asset correlation rho follows the Basel III IRB formula
(BCBS, "An Explanatory Note on the Basel II IRB Risk Weight Functions",
2005), interpolating between 12% and 24%:

    rho(PD) = 0.12 * (1 - exp(-50*PD)) / (1 - exp(-50))
            + 0.24 * (1 - (1 - exp(-50*PD)) / (1 - exp(-50)))

This module is fully country-agnostic. PD_TTC is computed from the
historical default rate. LGD_TTC is provided by the user (regulatory
floor, internal observation, or expert judgment).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

EPS = 1e-9


def basel_rho(pd_value: float) -> float:
    """Basel III IRB asset-correlation formula for corporate exposures."""
    pd_value = float(np.clip(pd_value, EPS, 1 - EPS))
    e50 = np.exp(-50.0)
    term = (1.0 - np.exp(-50.0 * pd_value)) / (1.0 - e50)
    return float(0.12 * term + 0.24 * (1.0 - term))


@dataclass
class FryeJacobsLGD:
    """Calibrated FJ-LGD function for one segment."""
    pd_ttc: float
    lgd_ttc: float
    rho: float = field(default=float("nan"))
    k: float = field(default=float("nan"))
    floor: float = 0.05
    cap: float = 0.95

    @classmethod
    def calibrate(cls,
                  pd_ttc: float,
                  lgd_ttc: float,
                  rho: Optional[float] = None,
                  floor: float = 0.05,
                  cap: float = 0.95) -> "FryeJacobsLGD":
        """Calibrate from (PD_TTC, LGD_TTC) anchor.

        If ``rho`` is None, the Basel III IRB formula is used.
        """
        pd_ttc_c = float(np.clip(pd_ttc, EPS, 1 - EPS))
        lgd_ttc_c = float(np.clip(lgd_ttc, EPS, 1 - EPS))
        if rho is None:
            rho = basel_rho(pd_ttc_c)
        # Calibration constant ensuring LGD(PD_TTC) = LGD_TTC
        k = float(norm.ppf(pd_ttc_c) - norm.ppf(pd_ttc_c * lgd_ttc_c))
        return cls(pd_ttc=pd_ttc_c, lgd_ttc=lgd_ttc_c,
                   rho=float(rho), k=k, floor=floor, cap=cap)

    def predict(self, pd_pit) -> np.ndarray:
        """Return PIT LGD given PIT PD (scalar, array, or pandas Series)."""
        if isinstance(pd_pit, pd.Series):
            idx = pd_pit.index
            arr = pd_pit.astype(float).values
            out = self._predict_array(arr)
            return pd.Series(out, index=idx, name="lgd_pit")
        arr = np.atleast_1d(np.asarray(pd_pit, dtype=float))
        return self._predict_array(arr)

    def _predict_array(self, pd_arr: np.ndarray) -> np.ndarray:
        pd_arr = np.clip(pd_arr, EPS, 1 - EPS)
        num = norm.ppf(pd_arr) - self.k
        denom = float(np.sqrt(max(1.0 - self.rho, EPS)))
        # FJ formula: LGD(PD) = Phi(num / denom) / PD
        lgd = norm.cdf(num / denom) / pd_arr
        return np.clip(lgd, self.floor, self.cap)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pd_ttc": float(self.pd_ttc),
            "lgd_ttc": float(self.lgd_ttc),
            "rho": float(self.rho),
            "k": float(self.k),
            "floor": float(self.floor),
            "cap": float(self.cap),
            "model": "FryeJacobs2012",
        }
