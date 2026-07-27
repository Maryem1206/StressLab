"""
mapping.py
==========
Map NGFS variable names to historical-data column names.

Strategy (hybrid)
-----------------
1. If a YAML override file is provided, it wins for matched keys.
2. Otherwise, the auto-mapper:
   - normalizes both names (lowercase, alphanumeric only),
   - applies a small synonym dictionary (gdp <- output, real_gdp;
     inflation <- cpi, hicp; unemployment <- jobless; etc.),
   - matches on substring, then on token-set Jaccard similarity.
3. NGFS variables with no match are DROPPED (per the user's instruction)
   and reported in the diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd

from .utils import EngineConfig, get_logger, normalize_label

log = get_logger(__name__)

try:
    import yaml  # optional
except ImportError:
    yaml = None


# Synonyms grouped by canonical concept. Keys are concepts; values are
# alternative spellings/aliases.
_SYNONYMS: Dict[str, List[str]] = {
    "gdp": ["gdp", "realgdp", "grossdomesticproduct", "output", "gdpgrowth"],
    "inflation": ["inflation", "cpi", "hicp", "consumerpriceindex", "priceindex"],
    "unemployment": ["unemployment", "jobless", "unemploymentrate"],
    "policyrate": ["policyrate", "interventionrate", "centralbankrate",
                   "shortrate", "interestrate", "policyinterestrate"],
    "longrate": ["longrate", "longterminterestrate", "10yyield",
                 "tenyearyield", "bondyield"],
    "exchangerate": ["exchangerate", "fx", "fxrate", "effectiveexchangerate"],
    "equity": ["equity", "stockindex", "stockprice", "equityprice",
               "equityindex"],
    "houseprice": ["houseprice", "housingprice", "realestate"],
    "consumption": ["consumption", "privateconsumption", "householdconsumption"],
    "investment": ["investment", "fixedinvestment", "capitalformation"],
    "domesticdemand": ["domesticdemand", "internaldemand"],
    "exports": ["exports"],
    "imports": ["imports"],
    "carbonprice": ["carbonprice", "co2price", "emissionsprice"],
    "oilprice": ["oilprice", "brent", "wti", "crude"],
    "coalprice": ["coalprice"],
    "gasprice": ["gasprice", "naturalgasprice"],
}


@dataclass
class MappingResult:
    matches: Dict[str, str] = field(default_factory=dict)   # ngfs_var -> hist_col
    unmatched: List[str] = field(default_factory=list)
    method_per_match: Dict[str, str] = field(default_factory=dict)


def _concept_of(name: str) -> Optional[str]:
    """Return the canonical concept key (e.g. 'gdp') matching this name, or None."""
    n = normalize_label(name)
    for concept, aliases in _SYNONYMS.items():
        for a in aliases:
            if a in n:
                return concept
    return None


def _load_yaml_overrides(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    if yaml is None:
        log.warning("PyYAML not installed; ignoring mapping override file.")
        return {}
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            log.warning("Mapping YAML is not a dict; ignoring.")
            return {}
        log.info("Loaded %d mapping overrides from %s.", len(data), path)
        return {str(k): str(v) for k, v in data.items()}
    except FileNotFoundError:
        log.warning("Mapping YAML not found: %s", path)
        return {}


def load_severity_ranks(path: Optional[str]) -> Dict[str, int]:
    """
    Charge la table de sévérité NGFS depuis mapping.yaml.

    Lit la section 'ngfs_scenario_severity' du fichier YAML et retourne
    un dict {nom_scénario: rang_int}. Générique : applicable à n'importe
    quel jeu de scénarios NGFS en modifiant le fichier de config.

    Retourne un dict vide si la section est absente ou si YAML n'est pas
    installé — le ScenarioSelectorConfig utilisera alors ses valeurs par défaut.
    """
    if not path:
        return {}
    if yaml is None:
        log.warning("PyYAML not installed; severity ranks not loaded.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        section = data.get("ngfs_scenario_severity", {})
        if not isinstance(section, dict):
            log.warning("'ngfs_scenario_severity' in %s is not a dict; ignored.", path)
            return {}
        ranks = {}
        for scenario_name, rank_val in section.items():
            try:
                ranks[str(scenario_name)] = int(rank_val)
            except (TypeError, ValueError):
                log.warning("Invalid rank '%s' for scenario '%s'; skipped.",
                            rank_val, scenario_name)
        log.info("Loaded %d NGFS severity ranks from %s.", len(ranks), path)
        return ranks
    except FileNotFoundError:
        log.debug("mapping.yaml not found at '%s'; using default severity ranks.", path)
        return {}
    except Exception as e:
        log.warning("Error loading severity ranks from %s: %s", path, e)
        return {}


def map_variables(ngfs_vars: List[str],
                  hist_cols: List[str],
                  cfg: EngineConfig) -> MappingResult:
    """Build a mapping NGFS variable -> historical column."""
    log.info("Mapping %d NGFS variables to %d historical columns.",
             len(ngfs_vars), len(hist_cols))

    overrides = _load_yaml_overrides(cfg.mapping_yaml_path)

    res = MappingResult()
    hist_set = set(hist_cols)

    # Pre-index historical columns by concept
    hist_by_concept: Dict[str, List[str]] = {}
    for c in hist_cols:
        cc = _concept_of(c)
        if cc:
            hist_by_concept.setdefault(cc, []).append(c)

    for v in ngfs_vars:
        # 1) explicit override
        if v in overrides:
            target = overrides[v]
            if target in hist_set:
                res.matches[v] = target
                res.method_per_match[v] = "override"
                continue
            else:
                log.warning("Override target '%s' for '%s' not in historical "
                            "columns — falling back.", target, v)

        # 2) concept match
        concept = _concept_of(v)
        if concept and concept in hist_by_concept:
            target = hist_by_concept[concept][0]   # first hit (deterministic)
            res.matches[v] = target
            res.method_per_match[v] = f"concept:{concept}"
            continue

        # 3) substring fallback (last resort)
        nv = normalize_label(v)
        candidates = [c for c in hist_cols if nv and (
            normalize_label(c) in nv or nv in normalize_label(c))]
        if candidates:
            target = candidates[0]
            res.matches[v] = target
            res.method_per_match[v] = "substring"
            continue

        res.unmatched.append(v)

    if res.unmatched:
        log.warning("Mapping: %d NGFS variables dropped (no historical proxy): %s",
                    len(res.unmatched), res.unmatched)
    log.info("Mapping: %d/%d NGFS variables matched.",
             len(res.matches), len(ngfs_vars))
    return res
