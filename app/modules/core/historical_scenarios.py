"""
historical_scenarios.py
=======================
Catalogue de crises historiques et calcul automatique des chocs.

Méthodologie (conforme BCBS 2009, EBA 2023) :
==============================================
  1. Le catalogue CRISIS_CATALOG définit les crises disponibles avec
     leurs fenêtres temporelles (t_pre_start..t_pre_end = période pré-crise,
     t_peak = année du pic de stress).

  2. compute_historical_shocks() calcule, pour chaque variable macro :
     Δvar = var(t_peak) − moyenne(var[t_pre_start..t_pre_end])
     C'est la définition standard du FMI pour les stress tests historiques.

  3. build_adverse_severe() applique la décomposition de sévérité :
     Adverse = 50% × Δvar    (stress modéré)
     Severe  = 100% × Δvar   (stress intégral)

  4. extract_pd_crisis_profile() extrait le profil temporel normalisé du PD
     pendant la pire crise observée dans les données uploadées.

Standardisation :
=================
  Aucun paramètre country-specific n'est hardcodé dans la logique.
  Les crises sont définies par {nom, pays, fenêtre temporelle}.
  Le module fonctionne identiquement pour n'importe quel pays.
  Il suffit d'ajouter une entrée au CRISIS_CATALOG.

Academic references :
  • BCBS (2009) "Principles for sound stress testing practices"
  • EBA (2023) "Methodological note — EU-wide stress test"
  • IMF (2014) "Stress Testing — Principles, Concepts and Frameworks"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("historical_scenarios")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CATALOGUE DES CRISES                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class CrisisDefinition:
    """
    Définition d'une crise historique.

    t_pre_start..t_pre_end : fenêtre pré-crise (pour la moyenne de référence)
    t_peak                 : année du pic de stress (pour mesurer le Δ)
    t_recovery             : année de début de récupération (optionnel)
    country                : ISO2 du pays ("global" si crise mondiale)
    description            : description lisible pour le reporting
    """
    name: str
    t_pre_start: int
    t_pre_end: int
    t_peak: int
    t_recovery: Optional[int] = None
    country: str = "global"
    description: str = ""


# ── Catalogue extensible ──────────────────────────────────────────────────
# Pour ajouter une crise : une seule ligne dans CRISIS_CATALOG.
# Les chocs seront calculés automatiquement depuis les données API.

CRISIS_CATALOG: Dict[str, CrisisDefinition] = {

    "egypt_2016": CrisisDefinition(
        name="egypt_2016",
        t_pre_start=2013,
        t_pre_end=2015,
        t_peak=2017,
        t_recovery=2019,
        country="EG",
        description="Dévaluation EGP nov. 2016 + choc inflationniste 2017",
    ),

    "gulf_war_1990": CrisisDefinition(
        name="gulf_war_1990",
        t_pre_start=1988,
        t_pre_end=1990,
        t_peak=1991,
        t_recovery=1993,
        country="global",
        description="Guerre du Golfe 1990-1991 — choc pétrolier + récession USA",
    ),

    "gfc_2008": CrisisDefinition(
        name="gfc_2008",
        t_pre_start=2005,
        t_pre_end=2007,
        t_peak=2009,
        t_recovery=2011,
        country="global",
        description="Global Financial Crisis 2008-2009",
    ),

    "covid_2020": CrisisDefinition(
        name="covid_2020",
        t_pre_start=2017,
        t_pre_end=2019,
        t_peak=2020,
        t_recovery=2021,
        country="global",
        description="Pandémie COVID-19 — confinements + choc de demande",
    ),

    "arab_spring_2011": CrisisDefinition(
        name="arab_spring_2011",
        t_pre_start=2008,
        t_pre_end=2010,
        t_peak=2011,
        t_recovery=2013,
        country="EG",
        description="Révolution égyptienne 2011 — instabilité politique",
    ),

    "taper_tantrum_2013": CrisisDefinition(
        name="taper_tantrum_2013",
        t_pre_start=2010,
        t_pre_end=2012,
        t_peak=2013,
        t_recovery=2015,
        country="global",
        description="Fed taper tantrum — fuite de capitaux EM",
    ),

    "russia_ukraine_2022": CrisisDefinition(
        name="russia_ukraine_2022",
        t_pre_start=2019,
        t_pre_end=2021,
        t_peak=2022,
        t_recovery=2024,
        country="global",
        description="Guerre Russie-Ukraine — choc énergie + inflation mondiale",
    ),
}


def list_available_crises() -> List[Dict[str, Any]]:
    """Retourne le catalogue sous forme lisible pour le frontend/config."""
    return [
        {
            "name": c.name,
            "description": c.description,
            "period": f"{c.t_pre_start}-{c.t_peak}",
            "country": c.country,
        }
        for c in CRISIS_CATALOG.values()
    ]


def get_crisis(name: str) -> CrisisDefinition:
    """Récupère une crise par son nom. Raise KeyError si introuvable."""
    if name not in CRISIS_CATALOG:
        available = ", ".join(CRISIS_CATALOG.keys())
        raise KeyError(
            f"Crise '{name}' introuvable. Disponibles : {available}")
    return CRISIS_CATALOG[name]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CALCUL DES CHOCS HISTORIQUES                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def compute_historical_shocks(
    macro_full: pd.DataFrame,
    crisis: CrisisDefinition,
) -> Dict[str, float]:
    """
    Calcule les chocs historiques Δvar pour chaque variable macro.

    Formule (standard FMI) :
        Δvar = var(t_peak) − moyenne(var[t_pre_start..t_pre_end])

    Parameters
    ----------
    macro_full : pd.DataFrame
        DataFrame macro complet (historique + projections).
        Index = années (int), colonnes = variables macro.
    crisis : CrisisDefinition
        Définition de la crise (fenêtre pré-crise + pic).

    Returns
    -------
    Dict[str, float]
        {variable_name: delta} pour chaque variable avec données suffisantes.
        delta est dans l'unité brute de la variable (%, USD, index, etc.)
    """
    deltas: Dict[str, float] = {}

    pre_years = list(range(crisis.t_pre_start, crisis.t_pre_end + 1))
    t_peak = crisis.t_peak

    for col in macro_full.columns:
        # Données pré-crise
        pre_data = macro_full.loc[
            macro_full.index.isin(pre_years), col
        ].dropna()

        # Données au pic
        if t_peak not in macro_full.index:
            continue
        peak_val = macro_full.loc[t_peak, col]

        if pd.isna(peak_val) or len(pre_data) == 0:
            continue

        pre_mean = float(pre_data.mean())
        delta = float(peak_val) - pre_mean

        deltas[col] = delta

        LOG.debug(
            "  %-30s pre_mean=%8.2f  peak=%8.2f  Δ=%+8.2f",
            col, pre_mean, float(peak_val), delta,
        )

    LOG.info(
        "compute_historical_shocks [%s]: %d variables, "
        "window [%d-%d] → peak %d",
        crisis.name, len(deltas),
        crisis.t_pre_start, crisis.t_pre_end, crisis.t_peak,
    )

    return deltas


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DÉCOMPOSITION ADVERSE / SEVERE                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

ADVERSE_SEVERITY: float = 0.50
SEVERE_SEVERITY: float = 1.00


def build_adverse_severe(
    historical_deltas: Dict[str, float],
    adverse_pct: float = ADVERSE_SEVERITY,
    severe_pct: float = SEVERE_SEVERITY,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Décompose les chocs historiques en scénarios adverse et severe.

    Parameters
    ----------
    historical_deltas : Dict[str, float]
        Sortie de compute_historical_shocks(). Δvar par variable.
    adverse_pct : float
        Fraction du choc pour adverse (default 0.50 = 50%).
    severe_pct : float
        Fraction du choc pour severe (default 1.00 = 100%).

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float]]
        (adverse_deltas, severe_deltas)
    """
    adverse = {var: delta * adverse_pct for var, delta in historical_deltas.items()}
    severe = {var: delta * severe_pct for var, delta in historical_deltas.items()}

    LOG.info(
        "build_adverse_severe: %d variables. "
        "Adverse=%.0f%%, Severe=%.0f%%",
        len(historical_deltas), adverse_pct * 100, severe_pct * 100,
    )

    return adverse, severe


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PROFIL TEMPOREL DE CRISE (forme normalisée)                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def extract_pd_crisis_profile(
    pd_series: pd.Series,
    n_horizon: int,
) -> np.ndarray:
    """
    Extrait le profil temporel de stress à partir de la pire crise
    observée dans la série PD historique.

    Contrainte réglementaire (EBA 2023, BCBS 2009) :
        Le stress s'intensifie sur l'horizon — il ne se résorbe pas.
        Le profil est donc **monotoniquement croissant** par construction.

    Méthode :
        1. Mesurer la vitesse de montée du PD durant la phase ascendante
           de la pire crise (du trough au peak).
        2. Convertir en profil normalisé croissant sur n_horizon.
        3. Forcer la monotonicité via cumulative max.

    Parameters
    ----------
    pd_series : pd.Series   Série PD historique (index = années).
    n_horizon : int         Longueur de l'horizon de projection.

    Returns
    -------
    np.ndarray  Profil monotone croissant ∈ [0, 1], longueur n_horizon.
    """
    vals = pd_series.sort_index().values.astype(float)
    n = len(vals)

    if n < 3:
        return _monotonic_ramp(n_horizon)

    # ── Trouver la phase ascendante la plus violente ──────────────────
    # On cherche le couple (trough_idx, peak_idx) qui maximise
    # PD[peak] - PD[trough] avec peak > trough (montée, pas descente)
    best_rise = 0.0
    best_trough = 0
    best_peak = n - 1

    for i in range(n):
        for j in range(i + 1, n):
            rise = vals[j] - vals[i]
            if rise > best_rise:
                best_rise = rise
                best_trough = i
                best_peak = j

    if best_rise < 1e-9:
        LOG.warning("No rising phase found in PD history. Using standard ramp.")
        return _monotonic_ramp(n_horizon)

    # ── Extraire la phase montante et la re-sampler sur n_horizon ─────
    rise_segment = vals[best_trough : best_peak + 1]
    rise_len = len(rise_segment)

    # Normaliser entre 0 et 1
    r_min, r_max = rise_segment.min(), rise_segment.max()
    rise_norm = (rise_segment - r_min) / (r_max - r_min + 1e-12)

    # Re-sampler sur n_horizon points via interpolation linéaire
    x_orig = np.linspace(0, 1, rise_len)
    x_new = np.linspace(0, 1, n_horizon)
    profile = np.interp(x_new, x_orig, rise_norm)

    # ── Forcer la monotonicité croissante (cumulative max) ────────────
    profile = np.maximum.accumulate(profile)

    # ── S'assurer que le profil part > 0 et finit à 1.0 ──────────────
    profile = np.clip(profile, 0.0, 1.0)
    if profile[-1] < 0.99:
        profile = profile / max(profile[-1], 1e-9)
    profile[-1] = 1.0  # peak at the end of the horizon

    years_idx = pd_series.sort_index().index
    LOG.info(
        "extract_pd_crisis_profile: rising phase %d→%d (%d years), "
        "ΔPD=%.4f (%.2f%% → %.2f%%), resampled to %d-year monotonic profile",
        int(years_idx[best_trough]), int(years_idx[best_peak]),
        rise_len, best_rise,
        vals[best_trough] * 100, vals[best_peak] * 100,
        n_horizon,
    )

    return profile


def _monotonic_ramp(n: int) -> np.ndarray:
    """
    Profil standard croissant concave (stress s'accélère puis se stabilise).
    Conforme EBA : montée progressive vers le pic en fin d'horizon.
    """
    t = np.linspace(0, 1, n)
    return t ** 0.7  # concave: accélération en début, plateau en fin


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONSTRUCTION DES TRAJECTOIRES PD FINALES                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def build_pd_trajectories(
    pd_baseline: np.ndarray,
    amplitude: float,
    profile: np.ndarray,
    adverse_pct: float = ADVERSE_SEVERITY,
    severe_pct: float = SEVERE_SEVERITY,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit les 3 trajectoires PD par décomposition de sévérité.

    PD_baseline(t)  = pd_baseline(t)                          [inchangé]
    PD_adverse(t)   = pd_baseline(t) + adverse_pct × amplitude × profile(t)
    PD_severe(t)    = pd_baseline(t) + severe_pct  × amplitude × profile(t)

    Properties garanties :
        • Severe ≥ Adverse ≥ Baseline ∀t  (par construction)
        • Forme de la déviation = profil historique empirique
        • Amplitude calibrée par le satellite model

    Parameters
    ----------
    pd_baseline : np.ndarray   PD baseline projeté par le satellite model
    amplitude : float          max_t(PD_stressed(t)) - PD_baseline(t=0)
    profile : np.ndarray       profil normalisé [0,1] de longueur n_horizon
    adverse_pct : float        fraction de sévérité adverse (default 0.50)
    severe_pct : float         fraction de sévérité severe (default 1.00)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (pd_baseline, pd_adverse, pd_severe) — all clipped to [1e-5, 0.99]
    """
    pd_adverse = pd_baseline + adverse_pct * amplitude * profile
    pd_severe = pd_baseline + severe_pct * amplitude * profile

    pd_baseline = np.clip(pd_baseline, 1e-5, 0.99)
    pd_adverse = np.clip(pd_adverse, 1e-5, 0.99)
    pd_severe = np.clip(pd_severe, 1e-5, 0.99)

    LOG.info(
        "build_pd_trajectories: amplitude=%.4f (%.2f pp), "
        "profile peak=%.2f",
        amplitude, amplitude * 100, profile.max(),
    )

    return pd_baseline, pd_adverse, pd_severe
