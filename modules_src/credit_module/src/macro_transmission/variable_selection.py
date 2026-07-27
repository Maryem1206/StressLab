"""Variable selection tournament for PD modeling.

Country-agnostic. Given a historical dataset and a list of candidate
macro variables, this module:

1. Determines the maximum combination size adaptively based on the
   number of available observations (rule of thumb: at least 10
   observations per estimated parameter, following Harrell, F.E.,
   "Regression Modeling Strategies", 2nd ed., Springer, 2015).
2. Generates all 1-, 2-, ..., k-variable combinations.
3. Fits each combination with each requested model family
   (Beta, Logit, Vasicek by default).
4. Scores each candidate by:
       (a) economic-sign consistency  (descending — primary)
       (b) in-sample R^2              (descending — secondary)
       (c) RMSE                       (ascending  — tie-breaker)
5. Returns the best model + a full leaderboard for audit.

This standardizes the variable-selection procedure used in case studies
(e.g. an Egyptian corporate portfolio with REER + trade-balance deficit)
to any country/portfolio: feed it any candidate macro variables and any
expected economic signs, and it picks the best subset on the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.logger import get_logger
from .pd_models import PD_MODEL_FAMILIES, _BasePDModel, build_pd_model

LOG = get_logger()


# ---------------------------------------------------------------------------
# Adaptive combo size policy
# ---------------------------------------------------------------------------

def adaptive_max_combo_size(n_obs: int) -> int:
    """Maximum number of macro variables in a combination given sample size.

    Conservative palier following the >=10 events-per-parameter rule
    (Harrell 2015). Each combination implies (k_vars + intercept + scale)
    ~= k_vars + 2 estimated parameters for Beta regression.

    n < 15  : 1   (very short series — typical with 12-15 yearly observations)
    n < 25  : 2
    n < 40  : 3
    n >= 40 : min(4, n // 10 - 2)
    """
    if n_obs < 15:
        return 1
    if n_obs < 25:
        return 2
    if n_obs < 40:
        return 3
    return max(2, min(4, n_obs // 10 - 2))


# ---------------------------------------------------------------------------
# Standardization helper (kept on the selector for inverse application)
# ---------------------------------------------------------------------------

@dataclass
class StandardScaler1D:
    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "StandardScaler1D":
        for c in df.columns:
            x = df[c].astype(float).values
            self.means[c] = float(np.mean(x))
            s = float(np.std(x, ddof=1))
            # Avoid degenerate columns
            self.stds[c] = s if s > 1e-12 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in df.columns:
            mu = self.means.get(c, 0.0)
            sd = self.stds.get(c, 1.0)
            out[c] = (df[c].astype(float).values - mu) / sd
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"means": dict(self.means), "stds": dict(self.stds)}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    target: str
    best_family: str
    best_features: List[str]
    best_model: _BasePDModel
    leaderboard: pd.DataFrame
    scaler: Optional[StandardScaler1D]
    candidate_variables: List[str]
    expected_signs: Dict[str, int]
    n_obs: int
    max_combo_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "best_family": self.best_family,
            "best_features": list(self.best_features),
            "model": self.best_model.to_dict(),
            "scaler": self.scaler.to_dict() if self.scaler else None,
            "n_obs": int(self.n_obs),
            "max_combo_size": int(self.max_combo_size),
            "candidate_variables": list(self.candidate_variables),
            "expected_signs": dict(self.expected_signs),
            "leaderboard_top": self.leaderboard.head(10).to_dict(orient="records"),
        }


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class VariableSelector:
    """Run the (family x variable-combination) tournament for a single PD target."""

    def __init__(self,
                 candidate_variables: Sequence[str],
                 expected_signs: Dict[str, int],
                 families: Sequence[str] = ("beta", "logit", "vasicek"),
                 standardize: bool = True,
                 max_combo_size: Optional[int] = None,
                 min_combo_size: int = 1) -> None:
        for fam in families:
            if fam not in PD_MODEL_FAMILIES:
                raise ValueError(f"Unknown PD model family: {fam}")
        self.candidate_variables = list(candidate_variables)
        self.expected_signs = dict(expected_signs)
        self.families = list(families)
        self.standardize = bool(standardize)
        self.max_combo_size = max_combo_size
        self.min_combo_size = int(min_combo_size)

    # ----------------------------------------------------------- public API

    def select(self, history: pd.DataFrame, target: str) -> SelectionResult:
        """Run the tournament and return the best model for ``target``."""
        if target not in history.columns:
            raise ValueError(f"Target column '{target}' not in history")

        # 1. Restrict to candidate vars actually present in history
        available_vars = [v for v in self.candidate_variables if v in history.columns]
        missing = sorted(set(self.candidate_variables) - set(available_vars))
        if missing:
            LOG.warning(f"[var_selector:{target}] candidate variables missing from history "
                        f"and ignored: {missing}")
        if not available_vars:
            raise ValueError(f"No candidate variables available in history for {target}")

        # 2. Extract working sample (drop rows with any NaN in target or candidates)
        working = history[[target] + available_vars].dropna()
        n_obs = len(working)
        if n_obs < 5:
            raise ValueError(f"Not enough observations for {target} ({n_obs} < 5)")

        # 3. Determine adaptive max combo size
        k_max = (self.max_combo_size if self.max_combo_size is not None
                 else adaptive_max_combo_size(n_obs))
        k_max = max(self.min_combo_size,
                    min(k_max, len(available_vars)))

        LOG.info(f"[var_selector:{target}] n_obs={n_obs} candidates={len(available_vars)} "
                 f"max_combo_size={k_max} families={self.families}")

        # 4. Standardize if requested (keeps an inverse-able scaler)
        scaler: Optional[StandardScaler1D] = None
        if self.standardize:
            scaler = StandardScaler1D().fit(working[available_vars])
            X_full = scaler.transform(working[available_vars])
        else:
            X_full = working[available_vars].astype(float).copy()
        y_full = working[target].astype(float).clip(1e-9, 1 - 1e-9)

        # 5. Run all (family x combo) fits
        rows: List[Dict[str, Any]] = []
        fitted: Dict[Tuple[str, Tuple[str, ...]], _BasePDModel] = {}

        for k in range(self.min_combo_size, k_max + 1):
            for combo in combinations(available_vars, k):
                X = X_full[list(combo)]
                for fam in self.families:
                    try:
                        model = build_pd_model(fam)
                        model.fit(X, y_full)
                    except Exception as exc:
                        LOG.warning(f"[var_selector:{target}] fit failed "
                                    f"family={fam} vars={combo}: {exc}")
                        continue
                    diag = model.diagnostics_
                    # Economic-sign score (+1 if matches expected sign, 0 else, NaN if not provided)
                    signs_ok = 0
                    signs_total = 0
                    sign_detail = {}
                    for f in combo:
                        exp = self.expected_signs.get(f, 0)
                        coef = model.params_[f]
                        if exp == 0:
                            sign_detail[f] = "n/a"
                            continue
                        signs_total += 1
                        # Note: in standardized OLS-on-link space, sign of beta has
                        # the same interpretation as on the original variable,
                        # because standardization is monotone affine.
                        if (coef > 0 and exp > 0) or (coef < 0 and exp < 0):
                            signs_ok += 1
                            sign_detail[f] = "OK"
                        else:
                            sign_detail[f] = "WRONG"
                    sens_eco_score = (signs_ok / signs_total) if signs_total > 0 else 1.0
                    fitted[(fam, combo)] = model
                    rows.append({
                        "family": fam,
                        "n_vars": k,
                        "variables": " + ".join(combo),
                        "variables_tuple": combo,
                        "r2": diag.get("r2", float("nan")),
                        "rmse": diag.get("rmse", float("nan")),
                        "aic": diag.get("aic", float("nan")),
                        "bic": diag.get("bic", float("nan")),
                        "sens_eco_score": sens_eco_score,
                        "signs_ok": signs_ok,
                        "signs_total": signs_total,
                        "sign_detail": sign_detail,
                        "n_obs": diag.get("n_obs", n_obs),
                    })

        if not rows:
            raise RuntimeError(f"No model could be fitted for target {target}")

        leaderboard = pd.DataFrame(rows)
        # 6. Rank: sens_eco desc -> r2 desc -> rmse asc
        leaderboard = leaderboard.sort_values(
            by=["sens_eco_score", "r2", "rmse"],
            ascending=[False, False, True]
        ).reset_index(drop=True)
        leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))

        best_row = leaderboard.iloc[0]
        best_family = best_row["family"]
        best_vars = list(best_row["variables_tuple"])
        best_model = fitted[(best_family, tuple(best_vars))]

        LOG.info(f"[var_selector:{target}] WINNER family={best_family} "
                 f"vars={best_vars} R2={best_row['r2']:.3f} RMSE={best_row['rmse']:.4f} "
                 f"sens_eco={best_row['signs_ok']}/{best_row['signs_total']}")

        # 7. Drop the python-tuple column for serialization friendliness
        leaderboard_serialisable = leaderboard.drop(columns=["variables_tuple"])

        return SelectionResult(
            target=target,
            best_family=best_family,
            best_features=best_vars,
            best_model=best_model,
            leaderboard=leaderboard_serialisable,
            scaler=scaler,
            candidate_variables=available_vars,
            expected_signs=self.expected_signs,
            n_obs=n_obs,
            max_combo_size=k_max,
        )
