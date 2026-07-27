"""
output_types.py
===============
Définition du contrat RiskOutput pour le module liquidité.

En production, remplacer par :
    from core.output_types import RiskOutput

Ce fichier standalone permet au module de tourner de façon autonome
sans dépendre de la plateforme complète.

Contrat plateforme :
    risk_type   = "liquidity"
    scenario_id = "baseline" | "adverse" | "severe"
    loss        = pic d'augmentation du NCO (unité monétaire)
    metrics     = dict avec clés standardisées (voir ci-dessous)
    time_series = pd.DataFrame colonnes : year|lcr|nsfr|hqla|nco|asf|rsf
    metadata    = dict libre
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


@dataclass
class RiskOutput:
    """Résultat standardisé d'un module de risque — contrat plateforme.

    Attributs
    ---------
    risk_type : str
        Identifiant du module. Toujours "liquidity" pour ce module.
    scenario_id : str
        Identifiant du scénario : "baseline", "adverse" ou "severe".
    loss : float
        Métrique de perte principale. Pour la liquidité : pic d'augmentation
        du NCO (Net Cash Outflow) par rapport au baseline, en unité monétaire.
        Valeur positive = stress, 0.0 pour le baseline.
    metrics : Dict[str, float]
        Métriques détaillées. Clés standardisées pour le module liquidité :
            lcr_baseline    — LCR année de référence (%)
            lcr_stressed    — LCR minimum sur l'horizon stressé (%)
            nsfr_baseline   — NSFR année de référence (%)
            nsfr_stressed   — NSFR minimum sur l'horizon stressé (%)
            breach_year     — Première année LCR<100 ou NSFR<100 (int ou None)
            hqla_delta      — Variation HQLA baseline→stressed (unité monétaire)
            nco_delta       — Variation NCO baseline→stressed (unité monétaire)
    time_series : pd.DataFrame
        Trajectoire annuelle. Colonnes : year, lcr, nsfr, hqla, nco, asf, rsf.
        Index entier 0..N-1, colonne "year" contient les années.
    metadata : dict
        Informations libres : banque, pays, modèles utilisés, paramètres, etc.
    """

    risk_type   : str
    scenario_id : str
    loss        : float
    metrics     : Dict[str, object]
    time_series : pd.DataFrame
    metadata    : Dict[str, object] = field(default_factory=dict)

    # ── Accès rapide aux métriques clés ──────────────────────────────────────

    @property
    def lcr_baseline(self) -> float:
        return float(self.metrics.get("lcr_baseline", float("nan")))

    @property
    def lcr_stressed(self) -> float:
        return float(self.metrics.get("lcr_stressed", float("nan")))

    @property
    def nsfr_baseline(self) -> float:
        return float(self.metrics.get("nsfr_baseline", float("nan")))

    @property
    def nsfr_stressed(self) -> float:
        return float(self.metrics.get("nsfr_stressed", float("nan")))

    @property
    def breach_year(self) -> Optional[int]:
        v = self.metrics.get("breach_year")
        return int(v) if v is not None else None

    @property
    def is_breach(self) -> bool:
        """True si LCR ou NSFR passe sous 100% sur l'horizon projeté."""
        return self.breach_year is not None

    # ── Affichage ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        breach = f"breach={self.breach_year}" if self.is_breach else "no breach"
        return (
            f"RiskOutput(risk_type='{self.risk_type}', "
            f"scenario='{self.scenario_id}', "
            f"lcr={self.lcr_stressed:.1f}%, "
            f"nsfr={self.nsfr_stressed:.1f}%, "
            f"{breach})"
        )

    def summary(self) -> str:
        """Résumé lisible sur plusieurs lignes."""
        lines = [
            f"{'─'*55}",
            f"  RiskOutput — {self.risk_type.upper()} | {self.scenario_id.upper()}",
            f"{'─'*55}",
            f"  LCR  : baseline={self.lcr_baseline:.1f}%  →  stressed={self.lcr_stressed:.1f}%",
            f"  NSFR : baseline={self.nsfr_baseline:.1f}%  →  stressed={self.nsfr_stressed:.1f}%",
            f"  NCO delta  : {self.metrics.get('nco_delta', 0):+.2f}",
            f"  HQLA delta : {self.metrics.get('hqla_delta', 0):+.2f}",
            f"  Breach year: {self.breach_year if self.is_breach else '—'}",
            f"  Loss (peak NCO increase): {self.loss:.2f}",
            f"{'─'*55}",
        ]
        return "\n".join(lines)
