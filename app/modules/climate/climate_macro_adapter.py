"""
climate_macro_adapter.py
========================
Translates NGFS trajectories into MacroDeltaVector objects, one per
(scenario_name, horizon_year) pair.

Two enrichment layers:

  Layer 1 — Macro delta (toujours présent)
    Calcule les Δmacro (GDP, inflation, taux, FX…) entre la trajectoire
    NGFS stressée et la baseline.  Source : macro_trajectories ou résultats
    pre-calculés (ngfs_credit / ngfs_liquidity).

  Layer 2 — Deltas cibles (optionnel, quand MultiTargetSatelliteFactory fourni)
    Les satellites calibrés projettent les variables de risque (PD, ΔLCR,
    ΔNSFR, β₁, β₂, β₃) sur les trajectoires NGFS stressées ET la baseline,
    puis calculent le delta stressé−baseline.  Ces deltas écrasent les
    approximations proxy de Layer 1 dans MacroDeltaVector.

    Appeler prepare_satellite_projections() avant compute_delta() pour
    activer la Layer 2.

Generic design — no hardcoded institution names, country codes, or magic
thresholds.  All limits are configurable via constructor arguments.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .macro_delta import MacroDeltaVector

if TYPE_CHECKING:
    from .multi_target_satellite import MultiTargetSatelliteFactory

LOG = logging.getLogger("climate.macro_adapter")

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de la Layer 1 (macro delta)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_BOUNDS: Dict[str, float] = {
    "delta_gdp_pct":             20.0,
    "delta_inflation_pct":       50.0,
    "delta_policy_rate_bp":    5000.0,
    "delta_fx_pct":             200.0,
    "delta_sovereign_spread_bp": 2000.0,
    "delta_oil_price_pct":      300.0,
    "delta_unemployment_pct":    30.0,
}

# (colonnes candidates, champ MacroDeltaVector, facteur de conversion)
_COL_MAP: List[Tuple[List[str], str, float]] = [
    (["real_gdp_growth", "gdp_growth", "GDP_growth"],
     "delta_gdp_pct", 1.0),
    (["cpi_inflation", "inflation"],
     "delta_inflation_pct", 1.0),
    (["policy_rate"],
     "delta_policy_rate_bp", 100.0),
    (["exchange_rate_lcu_usd", "exchange_rate", "real_effective_xrate"],
     "delta_fx_pct", 1.0),
    (["gov_debt_gdp", "lending_rate"],
     "delta_sovereign_spread_bp", 100.0),
    (["unemployment_rate"],
     "delta_unemployment_pct", 1.0),
    (["oil_price", "crude_oil_price", "oil_price_usd"],
     "delta_oil_price_pct", 1.0),
]

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de la Layer 2 (satellite factory)
# ─────────────────────────────────────────────────────────────────────────────

# Cible satellite → champ MacroDeltaVector
_TARGET_TO_FIELD: Dict[str, str] = {
    "pd":    "delta_pd_pp",
    "lcr":   "delta_lcr_pp",
    "nsfr":  "delta_nsfr_pp",
    "beta1": "delta_beta1",
    "beta2": "delta_beta2",
    "beta3": "delta_beta3",
}

# Facteur d'échelle : sortie satellite → unité du champ MacroDeltaVector
#   pd    : satellite en fraction (0.05 = 5%) → delta_pd_pp en pp (5.0)
#   lcr/nsfr : satellite en pp (déjà Δ)       → pas de conversion
#   beta* : satellite en unités NS absolues   → pas de conversion
_SATDELTA_SCALE: Dict[str, float] = {
    "pd":    100.0,
    "lcr":     1.0,
    "nsfr":    1.0,
    "beta1":   1.0,
    "beta2":   1.0,
    "beta3":   1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# ClimateMacroAdapter
# ─────────────────────────────────────────────────────────────────────────────

class ClimateMacroAdapter:
    """
    Traduit les trajectoires NGFS en MacroDeltaVector (Layer 1 + Layer 2).

    Parameters
    ----------
    satellite_factory : MultiTargetSatelliteFactory, optional
        Factory déjà calibrée (fit_all() appelé).  Si fournie, appeler
        prepare_satellite_projections() pour activer la Layer 2.
    plausibility_bounds : dict, optional
        Bornes d'alerte par champ MacroDeltaVector (warn-only).
    warn_only : bool
        Si True (défaut), les dépassements de bornes sont loggés mais ne
        bloquent pas le calcul.
    """

    def __init__(
        self,
        satellite_factory: Optional["MultiTargetSatelliteFactory"] = None,
        plausibility_bounds: Optional[Dict[str, float]] = None,
        warn_only: bool = True,
        gdp_pd_sensitivity: float = 0.5,
        rate_lcr_sensitivity: float = 5.0,
    ) -> None:
        self._factory              = satellite_factory
        self._bounds               = plausibility_bounds or dict(_DEFAULT_BOUNDS)
        self._warn_only            = warn_only
        self._gdp_pd_sensitivity   = gdp_pd_sensitivity
        self._rate_lcr_sensitivity = rate_lcr_sensitivity

        # Layer 2 — rempli par prepare_satellite_projections()
        # {scenario_name: {target: np.ndarray}}  — deltas stressé−baseline
        self._sat_deltas: Dict[str, Dict[str, np.ndarray]] = {}
        # {scenario_name: List[int]}
        self._sat_years:  Dict[str, List[int]] = {}

    # ── Layer 2 : branchement factory ────────────────────────────────────────

    def prepare_satellite_projections(
        self,
        ngfs_dfs: Dict[str, pd.DataFrame],
        baseline_name: str = "Baseline",
    ) -> None:
        """
        Projette la factory calibrée sur chaque scénario NGFS et stocke les
        deltas (stressé − baseline) pour enrichir les MacroDeltaVector.

        Doit être appelé après fit_all() et avant compute_delta().

        Parameters
        ----------
        ngfs_dfs : dict {scenario_name: DataFrame}
            DataFrame par scénario, indexé par année (int),
            colonnes ⊇ CLIMATE_MACRO_SOCLE.
        baseline_name : str
            Nom du scénario de référence dans ngfs_dfs (défaut "Baseline").
        """
        if self._factory is None:
            LOG.warning(
                "prepare_satellite_projections: satellite_factory non fournie — "
                "Layer 2 désactivée."
            )
            return

        if baseline_name not in ngfs_dfs:
            LOG.warning(
                "prepare_satellite_projections: scénario baseline '%s' absent "
                "de ngfs_dfs — Layer 2 désactivée.",
                baseline_name,
            )
            return

        # Projection baseline (référence commune)
        baseline_df    = ngfs_dfs[baseline_name]
        baseline_years = sorted(baseline_df.index.astype(int).tolist())
        proj_baseline  = self._factory.project(
            baseline_df, scenario_alias=baseline_name
        )
        LOG.info(
            "Satellite Layer 2: projection baseline '%s' — %d cibles, %d années",
            baseline_name, len(proj_baseline), len(baseline_years),
        )

        # Projection stressée pour chaque scénario ≠ baseline
        for scen_name, scen_df in ngfs_dfs.items():
            if scen_name == baseline_name:
                continue

            years = sorted(scen_df.index.astype(int).tolist())
            proj_stressed = self._factory.project(
                scen_df, scenario_alias=scen_name
            )

            # Calcul des deltas : stressé − baseline (interpolé sur l'index baseline)
            deltas: Dict[str, np.ndarray] = {}
            for target, arr_stressed in proj_stressed.items():
                arr_base = proj_baseline.get(target)
                if arr_base is None:
                    # Satellite baseline absent → delta = projection stressée brute
                    deltas[target] = arr_stressed.copy()
                    LOG.debug(
                        "Layer 2 [%s/%s]: baseline absent → delta = proj brute",
                        scen_name, target,
                    )
                    continue

                # Aligner sur l'index le plus court
                n = min(len(arr_stressed), len(arr_base))
                delta_arr = arr_stressed[:n] - arr_base[:n]
                deltas[target] = delta_arr

                LOG.debug(
                    "Layer 2 [%s/%s]: Δmax=%.4f Δmean=%.4f (n=%d pts)",
                    scen_name, target,
                    float(np.max(np.abs(delta_arr))),
                    float(np.mean(delta_arr)),
                    n,
                )

            self._sat_deltas[scen_name] = deltas
            self._sat_years[scen_name]  = years

        active = list(self._sat_deltas.keys())
        LOG.info(
            "Layer 2 prête: %d scénarios stressés — %s",
            len(active), active,
        )

        # ── Mise à jour B3.3 (Monotonie scénarios) dans les rapports post-estimation
        # Maintenant que les projections brutes par scénario sont disponibles,
        # on peut tester baseline ≤ adverse ≤ severe pour chaque satellite.
        # Les projections sont indexées par alias normalisé ("baseline", "adverse", etc.)
        if self._factory is not None and hasattr(self._factory, "update_monotony_test"):
            _scenario_projs: Dict[str, Dict[str, np.ndarray]] = {}
            _scenario_projs[baseline_name.lower()] = proj_baseline
            for _scen_name, _scen_df in ngfs_dfs.items():
                if _scen_name == baseline_name:
                    continue
                _proj = self._factory.project(_scen_df, scenario_alias=_scen_name)
                _scenario_projs[_scen_name.lower()] = _proj
            try:
                self._factory.update_monotony_test(_scenario_projs)
                LOG.info(
                    "Layer 2: test B3.3 (monotonie) mis à jour pour %d satellites",
                    len(self._factory._results),
                )
            except Exception as _me:
                LOG.warning(
                    "prepare_satellite_projections: update_monotony_test échoué "
                    "(ignoré): %s", _me,
                )

    def has_satellite_layer(self, scenario_name: str) -> bool:
        """True si la Layer 2 contient des projections pour ce scénario."""
        return scenario_name in self._sat_deltas

    # ── API publique ──────────────────────────────────────────────────────────

    def compute_delta(
        self,
        ngfs_projections: Dict[str, Any],
        scenario_name: str,
        horizon_year: int,
        baseline_name: str = "Baseline",
    ) -> MacroDeltaVector:
        """
        Calcule le MacroDeltaVector pour un (scénario, année).

        Layer 1 (macro delta) — toujours exécutée :
          1. macro_trajectories  (trajectoires brutes NGFS — source la plus précise)
          2. ngfs_credit / ngfs_liquidity  (résultats pré-calculés — fallback)

        Layer 2 (deltas cibles) — si prepare_satellite_projections() a été appelé :
          enrichit le vecteur avec delta_pd_pp, delta_lcr_pp, delta_nsfr_pp,
          delta_beta1/2/3 issus des satellites calibrés.
        """
        macro_traj = ngfs_projections.get("macro_trajectories")
        if macro_traj:
            vec = self._from_macro_trajectories(
                macro_traj, scenario_name, horizon_year, baseline_name
            )
        else:
            vec = self._from_computed_results(
                ngfs_projections, scenario_name, horizon_year
            )

        # Layer 2 — enrichissement satellite
        self._enrich_from_satellite(vec, scenario_name, horizon_year)

        self._check_plausibility(vec)
        return vec

    def compute_all_deltas(
        self,
        ngfs_projections: Dict[str, Any],
        scenario_names: List[str],
        horizon_years: List[int],
        baseline_name: str = "Baseline",
    ) -> Dict[str, Dict[int, MacroDeltaVector]]:
        """Retourne {scenario_name: {year: MacroDeltaVector}} pour toutes les combinaisons."""
        result: Dict[str, Dict[int, MacroDeltaVector]] = {}
        for scen in scenario_names:
            result[scen] = {}
            for yr in horizon_years:
                result[scen][yr] = self.compute_delta(
                    ngfs_projections, scen, yr, baseline_name
                )
        return result

    # ── Layer 2 — enrichissement ──────────────────────────────────────────────

    def _enrich_from_satellite(
        self,
        vec: MacroDeltaVector,
        scenario_name: str,
        horizon_year: int,
    ) -> None:
        """
        Injecte les deltas cibles issus du satellite dans le MacroDeltaVector.

        Pour chaque cible calibrée (pd, lcr, nsfr, beta1/2/3) :
          - Recherche l'index de l'année la plus proche dans _sat_years
          - Lit le delta pré-calculé (stressé − baseline)
          - Applique le facteur d'échelle (_SATDELTA_SCALE)
          - Écrit le champ correspondant sur le vecteur (setattr)

        Pas d'exception si la cible ou l'année est absente — dégradation
        silencieuse (le champ reste à 0.0, la Layer 1 conserve son effet).
        """
        deltas = self._sat_deltas.get(scenario_name)
        if not deltas:
            return

        years = self._sat_years.get(scenario_name, [])
        if not years:
            return

        # Index de l'année (exact ou nearest)
        if horizon_year in years:
            idx = years.index(horizon_year)
        else:
            idx = min(range(len(years)), key=lambda i: abs(years[i] - horizon_year))
            LOG.debug(
                "Layer 2 [%s/%d]: année exacte absente → nearest=%d",
                scenario_name, horizon_year, years[idx],
            )

        enriched: List[str] = []
        for target, arr in deltas.items():
            field = _TARGET_TO_FIELD.get(target)
            if field is None:
                continue
            if idx >= len(arr):
                continue
            scale = _SATDELTA_SCALE.get(target, 1.0)
            val   = float(arr[idx]) * scale
            setattr(vec, field, val)
            enriched.append(f"{field}={val:.4f}")

        if enriched:
            LOG.info(
                "Layer 2 [%s/%d]: enrichi — %s",
                scenario_name, horizon_year, "  ".join(enriched),
            )

    # ── Layer 1 — helpers ─────────────────────────────────────────────────────

    def _from_macro_trajectories(
        self,
        macro_traj: Dict[str, Any],
        scenario_name: str,
        horizon_year: int,
        baseline_name: str,
    ) -> MacroDeltaVector:
        """Construit le MacroDeltaVector Layer 1 depuis les trajectoires macro brutes."""
        baseline_row = self._extract_row(macro_traj, baseline_name, horizon_year)
        stressed_row = self._extract_row(macro_traj, scenario_name, horizon_year)

        if not baseline_row and not stressed_row:
            LOG.warning(
                "ClimateMacroAdapter: pas de données macro pour scénario=%r "
                "année=%d — vecteur zéro.",
                scenario_name, horizon_year,
            )
            return self._zero_delta(scenario_name, horizon_year)

        fields: Dict[str, float] = {}
        for cols, field_name, factor in _COL_MAP:
            for col in cols:
                bl_val = baseline_row.get(col)
                st_val = stressed_row.get(col)
                if bl_val is not None and st_val is not None:
                    try:
                        fields[field_name] = (float(st_val) - float(bl_val)) * factor
                    except (TypeError, ValueError):
                        pass
                    break  # première colonne gagnante

        return MacroDeltaVector(
            scenario_name      = scenario_name,
            horizon_year       = horizon_year,
            source_scenario_id = f"{scenario_name}:{horizon_year}",
            **fields,
        )

    def _from_computed_results(
        self,
        ngfs_projections: Dict[str, Any],
        scenario_name: str,
        horizon_year: int,
    ) -> MacroDeltaVector:
        """
        Approxime le MacroDeltaVector Layer 1 depuis les résultats pré-calculés
        (ngfs_credit / ngfs_liquidity).

        Ces dicts contiennent des ratios stressés (PD, LCR, NSFR), pas les
        variables macro brutes — les deltas produits sont des proxies grossiers.
        Ils sont remplacés par la Layer 2 si la factory est active.
        """
        alias  = self._resolve_alias(ngfs_projections, scenario_name)
        fields: Dict[str, float] = {}

        # ── Credit : proxy GDP depuis ΔPDC ───────────────────────────────────
        ngfs_credit = ngfs_projections.get("ngfs_credit", {})
        if ngfs_credit:
            scens  = ngfs_credit.get("scenarios", {})
            bl_scen = scens.get("baseline", {})
            st_scen = scens.get(alias, {})
            bl_pd  = _get_value_at_year(
                bl_scen.get("years", []), bl_scen.get("pd", []), horizon_year)
            st_pd  = _get_value_at_year(
                st_scen.get("years", []), st_scen.get("pd", []), horizon_year)
            if bl_pd is not None and st_pd is not None and bl_pd > 0:
                fields["delta_gdp_pct"] = -(st_pd - bl_pd) * 100 * self._gdp_pd_sensitivity

        # ── Liquidity : proxy taux depuis ΔLCR ───────────────────────────────
        ngfs_liq = ngfs_projections.get("ngfs_liquidity", {})
        if ngfs_liq:
            scens   = ngfs_liq.get("scenarios", {})
            bl_scen = scens.get("baseline", {})
            st_scen = scens.get(alias, {})
            bl_lcr  = _get_value_at_year(
                bl_scen.get("years", []), bl_scen.get("lcr", []), horizon_year)
            st_lcr  = _get_value_at_year(
                st_scen.get("years", []), st_scen.get("lcr", []), horizon_year)
            if bl_lcr is not None and st_lcr is not None and bl_lcr > 0:
                fields["delta_policy_rate_bp"] = -(st_lcr - bl_lcr) * self._rate_lcr_sensitivity

        return MacroDeltaVector(
            scenario_name      = scenario_name,
            horizon_year       = horizon_year,
            source_scenario_id = f"{scenario_name}:{horizon_year}:approx",
            **fields,
        )

    # ── utilitaires ───────────────────────────────────────────────────────────

    def _resolve_alias(
        self,
        ngfs_projections: Dict[str, Any],
        scenario_name: str,
    ) -> str:
        adverse_name  = ngfs_projections.get("adverse_scenario", "")
        severe_name   = ngfs_projections.get("severe_scenario", "")
        baseline_name = ngfs_projections.get("baseline_scenario", "Baseline")
        if scenario_name == baseline_name:
            return "baseline"
        if scenario_name == adverse_name:
            return "adverse"
        if scenario_name == severe_name:
            return "severe"
        name_lower = scenario_name.lower()
        if any(kw in name_lower for kw in ("sev", "below", "fragment", "delayed")):
            return "severe"
        return "adverse"

    @staticmethod
    def _extract_row(
        macro_traj: Dict[str, Any],
        scenario_name: str,
        year: int,
    ) -> Dict[str, Any]:
        scen_data = macro_traj.get(scenario_name, {})
        row = scen_data.get(year) or scen_data.get(str(year)) or {}
        return dict(row)

    @staticmethod
    def _zero_delta(scenario_name: str, horizon_year: int) -> MacroDeltaVector:
        return MacroDeltaVector(
            scenario_name      = scenario_name,
            horizon_year       = horizon_year,
            source_scenario_id = f"{scenario_name}:{horizon_year}:zero",
        )

    def _check_plausibility(self, vec: MacroDeltaVector) -> None:
        """Log si un delta macro dépasse la borne de plausibilité (ne bloque pas)."""
        for field_name, bound in self._bounds.items():
            val = getattr(vec, field_name, 0.0)
            if abs(val) > bound:
                LOG.warning(
                    "ClimateMacroAdapter: delta suspect — "
                    "%s=%.2f dépasse la borne ±%.1f  "
                    "(scénario=%r année=%d)",
                    field_name, val, bound,
                    vec.scenario_name, vec.horizon_year,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Helper module-level
# ─────────────────────────────────────────────────────────────────────────────

def _get_value_at_year(
    years: List[Any],
    values: List[Any],
    target_year: int,
) -> Optional[float]:
    """
    Retourne la valeur pour target_year depuis des listes parallèles.
    Fallback sur l'année la plus proche si l'exacte est absente.
    """
    pairs: List[tuple] = []
    for yr, val in zip(years, values):
        try:
            pairs.append((int(yr), float(val)))
        except (TypeError, ValueError):
            pass
    if not pairs:
        return None
    for yr, val in pairs:
        if yr == target_year:
            return val
    _, nearest_val = min(pairs, key=lambda p: abs(p[0] - target_year))
    return nearest_val
