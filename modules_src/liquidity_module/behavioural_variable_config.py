"""
behavioural_variable_config.py
==============================
Source de vérité unique pour les bornes comportementales du module liquidité.

Ce fichier est l'unique source de vérité pour les bornes comportementales
du module liquidité. Ne pas dupliquer ailleurs.
Note : satellite_calibrator.py et validation_engine.py gardent leurs propres
copies existantes pour l'instant — dette connue, factorisation post-soutenance.
"""

from __future__ import annotations

# Registre des 5 variables comportementales satellites.
# bounds : (min, max) domaine réglementaire / économique admissible.
# min_obs : observations historiques minimales pour calibration.
# conflict_resolution : méthode appliquée en cas de conflit ADF/KPSS.
SATELLITE_VARIABLE_CONFIG: dict = {
    "run_off_retail": {
        "bounds": (0.0, 1.0),
        "min_obs": 5,
        "conflict_resolution": "AR(1)",
    },
    "run_off_corporate": {
        "bounds": (0.0, 1.0),
        "min_obs": 5,
        "conflict_resolution": "AR(1)",
    },
    "haircut_add": {
        "bounds": (0.0, 0.50),
        "min_obs": 5,
        "conflict_resolution": "AR(1)",
    },
    "asf_factor_retail": {
        "bounds": (0.70, 1.00),
        "min_obs": 5,
        "conflict_resolution": "AR(1)",
    },
    "asf_factor_corporate": {
        "bounds": (0.20, 0.80),
        "min_obs": 5,
        "conflict_resolution": "AR(1)",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# DELTA_CAPS — plafond dur sur la variation annuelle (|Δ| an sur an)
# ─────────────────────────────────────────────────────────────────────────────
# Règle de calibration : cap = max(|Δ annuel|) historique × 2
#   ("extreme but plausible" — autorise un choc deux fois plus violent que le
#   pire mouvement annuel jamais observé sur l'historique de la banque, sans
#   autoriser une divergence de plusieurs dizaines de points de % en un an
#   comme produisait la régression satellite non plafonnée sous choc NGFS).
#
# Source des données : uploads/liquidity_input_1.xlsx, colonnes bilan
#   historiques 2010-2024 (15 années), lues via
#   LiquidityDataLoader._read_time_series(). max(|Δ|) calculé comme
#   max(|série[t] - série[t-1]|) sur toute la période disponible.
#
# Traçabilité (pour reproduire le calcul) :
#   run_off_retail        : max historique brut = 0.0550 (année 2020) → cap = 0.11
#   run_off_corporate     : max historique brut = 0.1400 (année 2022) → cap = 0.28
#   haircut_add           : max historique brut = 0.0800 (année 2022) → cap = 0.16
#   asf_factor_retail     : max historique brut = 0.1200 (année 2020) → cap = 0.24
#   asf_factor_corporate  : max historique brut = 0.1600 (année 2016) → cap = 0.32
#
# Appliqué dans LiquidityStressEngine.compute_stress() : le delta contraint en
# signe (constrained_delta) est plafonné à ±DELTA_CAPS[var] AVANT la
# décroissance long-terme (0.5^(t/half_life)) — la décroissance s'applique
# donc sur le delta déjà borné, jamais sur le delta brut de la régression.
DELTA_CAPS: dict = {
    "run_off_retail":       0.11,
    "run_off_corporate":    0.28,
    "haircut_add":          0.16,
    "asf_factor_retail":    0.24,
    "asf_factor_corporate": 0.32,
}
