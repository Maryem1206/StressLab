"""
scenario_family_filter.py
=========================
Filtre les scénarios CT (GEM-E3 IIASA) pour ne garder qu'un représentant
par famille de scénarios.

Problème résolu
---------------
GEM-E3 publie des scénarios avec :
  1. Des runs stochastiques : DIRE_run1, HWTP_run1, SWUC_run1, ...
     → doublons du même scénario, à exclure
  2. Des variantes géographiques : DAPS_AFR_R, DAPS_ASIA, DAPS_EUR, ...
     → même type de choc, régions différentes
     → garder uniquement la variante la plus pertinente pour le pays analysé

Pipeline
--------
1. Exclure tout scénario dont le nom contient "_run" (runs stochastiques)
2. Grouper les scénarios restants par famille (préfixe commun)
3. Pour chaque famille avec N variantes → sélectionner la plus pertinente
   selon la région WB du pays (auto-dérivée depuis cfg.country)
4. Retourner 1 représentant par famille → candidats pour le scenario_selector

Exemple (Égypte, région WB = MNA)
----------------------------------
Entrée  : DAPS_AFR_R, DAPS_AFR_R_run1, DAPS_ASIA, DAPS_EUR, DAPS_NAM,
          DIRE, DIRE_run1, HWTP, HWTP_run1, SWUC, SWUC_run1
Sortie  : DAPS_AFR_R, DIRE, HWTP, SWUC   ← 4 scénarios, 1 par famille

Standardisation
---------------
Le module est entièrement générique :
- Aucun nom de scénario hardcodé
- La sélection de variante DAPS est auto-dérivée depuis cfg.country
- Fonctionne pour n'importe quelle structure de noms de scénarios GEM-E3

Note : ce module est UNIQUEMENT appelé en mode CT (is_ct=True).
       Le mode LT (NGFS NiGEM) n'est pas concerné.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from .utils import EngineConfig

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constantes génériques
# ──────────────────────────────────────────────────────────────────────

# Pattern pour détecter les runs stochastiques
# Couvre : _run1, _run2, _run_1, _RUN1, etc.
_RUN_PATTERN = re.compile(r"_run\d*$", re.IGNORECASE)

# Mapping : suffixe géographique → priorité par région WB
# Clé = code région WB, Valeur = liste de suffixes par ordre de priorité
_WB_REGION_TO_GEO_SUFFIXES: Dict[str, List[str]] = {
    "MNA": ["AFR_R", "AFR", "AFRICA"],              # Middle East & North Africa
    "SSA": ["AFR_R", "AFR", "AFRICA"],              # Sub-Saharan Africa
    "ECS": ["EUR", "EU", "EUROPE"],                 # Europe & Central Asia
    "LCN": ["SAM", "SA", "LAT", "LATAM", "LATIN"], # Latin America & Caribbean
    "EAP": ["ASIA", "AS", "PACIFIC", "PAC"],        # East Asia & Pacific
    "SAS": ["ASIA", "AS", "SOUTH_AS"],              # South Asia
    "NAC": ["NAM", "NA", "NORTH_AM", "NORTHAM"],    # North America
}

# Fallback si la région WB n'est pas dans le mapping
_DEFAULT_GEO_PRIORITY: List[str] = [
    "AFR_R", "AFR", "EUR", "ASIA", "NAM", "SAM", "OCE"
]


# ──────────────────────────────────────────────────────────────────────
# Fonctions internes
# ──────────────────────────────────────────────────────────────────────

def _is_run(scenario_name: str) -> bool:
    """Retourne True si le scénario est un run stochastique (_run, _run1, etc.)."""
    return bool(_RUN_PATTERN.search(scenario_name))


def _get_family(scenario_name: str) -> str:
    """
    Extrait le nom de famille d'un scénario GEM-E3.

    Règle : le nom de famille est le premier token séparé par "_".
    Exemples :
        DAPS_AFR_R  → "DAPS"
        DIRE        → "DIRE"
        HWTP_run1   → "HWTP"
        SWUC        → "SWUC"
        NET_ZERO    → "NET"   (si une telle famille existe)
    """
    return scenario_name.split("_")[0].upper()


def _get_geo_suffix(scenario_name: str) -> Optional[str]:
    """
    Extrait le suffixe géographique d'un scénario (tout ce qui suit la famille).

    Exemples :
        DAPS_AFR_R  → "AFR_R"
        DAPS_ASIA   → "ASIA"
        DAPS_EUR    → "EUR"
        DIRE        → None  (pas de suffixe géo)
    """
    parts = scenario_name.split("_", 1)
    if len(parts) < 2:
        return None
    suffix = parts[1].upper()
    # Retirer le suffixe _run si présent
    suffix = _RUN_PATTERN.sub("", suffix).strip("_")
    return suffix if suffix else None


def _select_best_variant(
    variants: List[str],
    wb_region: Optional[str],
) -> str:
    """
    Sélectionne la variante la plus pertinente pour une région WB donnée.

    Algorithme :
    1. Récupérer la liste de priorités géographiques pour la région WB
    2. Chercher la première variante dont le suffixe géo correspond
    3. Si aucune correspondance → prendre la première variante alphabétiquement

    Parameters
    ----------
    variants : liste des variantes d'une même famille (sans les runs)
    wb_region : code région WB (MNA, ECS, LCN, EAP, SAS, NAC, SSA)

    Returns
    -------
    Nom du scénario représentant sélectionné
    """
    if len(variants) == 1:
        return variants[0]

    # Récupérer les priorités géographiques pour cette région
    geo_priorities = _WB_REGION_TO_GEO_SUFFIXES.get(
        wb_region or "", _DEFAULT_GEO_PRIORITY
    )

    # Construire un mapping suffixe → scénario
    suffix_to_scn: Dict[str, str] = {}
    for scn in variants:
        geo = _get_geo_suffix(scn)
        if geo:
            suffix_to_scn[geo] = scn

    # Chercher le premier suffixe prioritaire présent dans les variants
    for priority_suffix in geo_priorities:
        if priority_suffix in suffix_to_scn:
            return suffix_to_scn[priority_suffix]

    # Aucune correspondance → première variante alphabétiquement
    return sorted(variants)[0]


def _get_wb_region_for_country(country: str) -> Optional[str]:
    """
    Retourne le code région WB pour un pays donné.
    Utilise le resolver existant pour éviter la duplication de logique.
    """
    try:
        from .variable_resolver import get_wb_region_code
        return get_wb_region_code(country)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# Fonction principale
# ──────────────────────────────────────────────────────────────────────

def filter_scenario_families(
    scenarios: List[str],
    baseline_name: str,
    cfg: EngineConfig,
) -> List[str]:
    """
    Filtre les scénarios CT pour ne garder qu'un représentant par famille.

    Étapes :
    1. Exclure le Baseline et les runs stochastiques (_run, _run1, ...)
    2. Grouper par famille (préfixe avant "_")
    3. Pour chaque famille avec plusieurs variantes, sélectionner la plus
       pertinente selon la région WB du pays configuré
    4. Retourner la liste filtrée

    Parameters
    ----------
    scenarios : liste complète des scénarios disponibles dans le fichier CT
    baseline_name : nom du scénario Baseline (à exclure du filtrage)
    cfg : EngineConfig (utilise cfg.country pour la région WB)

    Returns
    -------
    Liste filtrée : 1 représentant par famille, sans runs
    """
    # ── Résoudre la région WB du pays ────────────────────────────────
    wb_region = _get_wb_region_for_country(cfg.country)
    log.info(
        "ScenarioFamilyFilter: country='%s' → WB region='%s'.",
        cfg.country, wb_region or "unknown",
    )

    # ── Étape 1 : exclure baseline + runs ────────────────────────────
    candidates = [
        s for s in scenarios
        if s != baseline_name and not _is_run(s)
    ]

    n_runs_excluded = len([s for s in scenarios
                           if s != baseline_name and _is_run(s)])
    log.info(
        "ScenarioFamilyFilter: %d scénarios avant filtrage, "
        "%d runs exclus → %d candidats.",
        len(scenarios) - 1,   # -1 pour baseline
        n_runs_excluded,
        len(candidates),
    )

    if not candidates:
        log.warning("ScenarioFamilyFilter: aucun candidat après exclusion des runs.")
        return []

    # ── Étape 2 : grouper par famille ────────────────────────────────
    families: Dict[str, List[str]] = {}
    for scn in candidates:
        family = _get_family(scn)
        families.setdefault(family, []).append(scn)

    log.info(
        "ScenarioFamilyFilter: %d familles détectées: %s.",
        len(families),
        {f: len(v) for f, v in families.items()},
    )

    # ── Étape 3 : sélectionner un représentant par famille ───────────
    representatives: List[str] = []
    for family, variants in sorted(families.items()):
        if len(variants) == 1:
            chosen = variants[0]
            log.info(
                "  Famille '%s': 1 variante → '%s' ✓", family, chosen
            )
        else:
            chosen = _select_best_variant(variants, wb_region)
            log.info(
                "  Famille '%s': %d variantes %s → '%s' sélectionné "
                "(région WB='%s') ✓",
                family, len(variants), variants, chosen, wb_region or "?",
            )
        representatives.append(chosen)

    log.info(
        "ScenarioFamilyFilter: %d familles → %d représentants: %s.",
        len(families), len(representatives), representatives,
    )

    return representatives
