"""
utils.py
========
Cross-cutting utilities for the Macro Variable Selection Engine.

Contents
--------
- get_logger : standard logger factory
- EngineConfig : single configuration object passed through the pipeline
- ECONOMIC_PRIORS : single source of truth for economic-sign expectations
- expected_sign : look up expected sign for a (variable, risk_type) pair
- normalize_label : fuzzy-matching helper

Sign convention for ECONOMIC_PRIORS
-----------------------------------
   +1  : increase in macro variable is expected to INCREASE the target
         (e.g., unemployment up -> PD up)
   -1  : increase in macro variable is expected to DECREASE the target
         (e.g., GDP up -> PD down)
    0  : ambiguous / context-dependent — never penalize the sign
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Optional
from dataclasses import dataclass, field
from typing import List, Callable


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def get_logger(name: str = "macro_engine", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(h)
        logger.setLevel(level)
        # NOTE: propagate is intentionally left True (default) so these
        # records also reach the Flask app's root handlers (console +
        # logs/app.log — see app/__init__.py). This module used to set
        # propagate=False, which silently dropped every log line from
        # multi_scenario.py/data_loader.py/geo_selector.py/etc. from
        # app.log — the entire Stage 1/2 tournament and NGFS-file loading
        # ran with zero visible timestamps there, making the two multi-
        # minute Phase 1 gaps in app.log look like unexplained stalls.
    return logger


# -----------------------------------------------------------------------------
# Economic priors (extensible at runtime)
# -----------------------------------------------------------------------------
ECONOMIC_PRIORS: Dict[str, Dict[str, int]] = {
    "credit": {
        "gdp_growth": -1, "real_gdp": -1, "gdp": -1,
        "industrial_production": -1, "consumption": -1, "investment": -1,
        "domestic_demand": -1, "exports": -1, "imports": 0,
        "house_price": -1, "equity": -1, "stock_index": -1,
        "unemployment": +1,
        "inflation": 0, "cpi": 0,
        "policy_rate": +1, "intervention_rate": +1,
        "short_rate": +1, "long_rate": +1,
        "interest_rate": +1, "yield": +1, "credit_spread": +1,
        "oil_price": 0,
        "carbon_price": +1, "co2_price": +1,
        "emissions": 0, "energy_price": +1,
        "coal_price": +1, "gas_price": +1,
        "exchange_rate": 0, "fx": 0, "effective_exchange_rate": 0,
    },
    "market": {
        "vix": +1, "volatility": +1,
        "equity": -1, "stock_index": -1,
        "credit_spread": +1, "policy_rate": 0, "yield": 0,
        "gdp": -1, "oil_price": 0, "carbon_price": +1,
        "exchange_rate": 0,
    },

    # Module Marche -- beta1, beta2, beta3 (Nelson-Siegel) -- ajoute le 2026-06-26
    # Source API : FRED (EGYPTAINTRATE) / World Bank (WB) / IMF WEO
    # Variables : policy_rate, inflation, gdp_growth, fiscal_deficit,
    #             public_debt, fx_reserves
    # beta1 = niveau de la courbe (level)
    # beta2 = pente de la courbe (slope)
    # beta3 = courbure (curvature) -- neutre pour toutes les variables retenues
    "market_beta1": {
        # beta1 (niveau) -- impact sur le niveau moyen de la courbe des taux souverains
        "policy_rate":    +1,  # transmission directe politique monetaire -> niveau courbe
        "inflation":      +1,  # anticipations inflation -> prime nominale (effet Fisher)
        "gdp_growth":     -1,  # croissance -> fuite vers actifs risques -> pression baissiere sur taux souverains
        "fiscal_deficit": +1,  # deficit -> prime de risque fiscal -> pression haussiere sur taux longs
        "public_debt":    +1,  # dette elevee -> prime de soutenabilite -> taux souverains plus eleves
        "fx_reserves":    -1,  # reserves hausse -> signal solidite macroeconomique -> taux souverains bas
    },
    "market_beta2": {
        # beta2 (spread d'inversion) -- NS convention: beta2 = y(tau->0) - y(tau->inf)
        #   = taux_court - taux_long  (positif si courbe inversee, negatif si courbe normale)
        # Regle : si macro -> y(court) monte PLUS que y(long) -> beta2 augmente -> +1
        #         si macro -> y(long)  monte PLUS que y(court) -> beta2 baisse   -> -1
        "policy_rate":    +1,  # CBE hausse -> T91j monte fortement -> beta2 augmente
        "inflation":      +1,  # inflation -> taux courts reagissent en premier -> beta2 augmente
        "gdp_growth":     -1,  # croissance -> investissement -> demande taux longs -> y(long) monte -> beta2 baisse
        "fiscal_deficit": -1,  # deficit -> emission obligataire longue -> y(long) monte -> beta2 baisse
        "public_debt":    -1,  # prime soutenabilite sur les maturites longues -> beta2 baisse
        "fx_reserves":    -1,  # confiance -> pression baissiere sur taux courts -> beta2 baisse
    },
    "market_beta3": {
        # beta3 (courbure) -- neutre pour toutes les variables macro retenues
        "policy_rate":     0,
        "inflation":       0,
        "gdp_growth":      0,
        "fiscal_deficit":  0,
        "public_debt":     0,
        "fx_reserves":     0,
    },

    # Liquidity target convention: HIGHER LCR/NSFR = SAFER → signs are vs target.
    "liquidity": {
        "gdp": +1, "unemployment": -1,
        "policy_rate": -1, "credit_spread": -1,
        "deposit": +1, "interbank_rate": -1,
    },
    "climate": {
        "carbon_price": +1, "co2_price": +1,
        "emissions": 0, "temperature": +1,
        "physical_damage": +1, "energy_price": +1,
        "gdp": -1,
    },
    "operational": {
        "gdp": -1, "unemployment": +1, "inflation": 0,
    },
}


def expected_sign(variable_name: str, risk_type: str) -> int:
    """Return +1, -1 or 0. 0 = no prior / ambiguous."""
    priors = ECONOMIC_PRIORS.get(risk_type.lower(), {})
    name_l = re.sub(r"[^a-z0-9]+", "_", variable_name.lower()).strip("_")
    for pattern in sorted(priors.keys(), key=len, reverse=True):
        if pattern in name_l:
            return priors[pattern]
    return 0


def normalize_label(label: str) -> str:
    """Lowercase + strip non-alphanumerics. Used for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class EngineConfig:
    """Single configuration passed through the entire pipeline."""

    # --- geography ---
    country: str = ""
    region_map: Dict[str, str] = field(default_factory=dict)

    # --- NGFS scenario / risk channel ---
    scenario: str = "Net Zero 2050"
    # In NGFS Phase 5, the actual unstressed reference is the 'Baseline' scenario
    # (which has NO risk-channel suffix in variable names). 'Current Policies' is
    # itself a stressed scenario and has only physical channels in this file.
    baseline_scenario: str = "Baseline"
    risk_channel: str = "combined"                # 'combined' STRICT (option a)

    # --- target ---
    target_variable: str = "PD"
    risk_type: str = "credit"
    target_col: str = "target"   # nom de la colonne PD dans hist_df
    # --- file paths ---
    ngfs_path: str = ""
    historical_path: str = ""
    mapping_yaml_path: Optional[str] = None
    auto_accept_mapping: bool = False
    output_path: str = "engine_output.json"

    # --- NGFS column names (Phase 5 official layout) ---
    ngfs_sheet: str = "data"
    ngfs_col_model: str = "Model"
    ngfs_col_scenario: str = "Scenario"
    ngfs_col_region: str = "Region"
    ngfs_col_variable: str = "Variable"
    ngfs_col_unit: str = "Unit"

    # --- historical file layout (user-provided wide CSV/XLSX) ---
    hist_year_col: str = "year"

    # --- Stage 1 filters ---
    s1_keep_top_variance_pct: float = 0.70
    s1_min_shock_magnitude: float = 0.0
    s1_apply_economic_whitelist: bool = True

    # --- Stage 2 filters ---
    min_abs_correlation: float = 0.20
    vif_threshold: float = 5.0
    pvalue_threshold: float = 0.10
    max_combo_size: int = 3
    min_combo_size: int = 2               # combo minimum 2 variables (pas de modèle univarié)
    max_candidate_vars: int = 8

    # --- ranking weights ---
    w_economic: float = 0.50
    w_bic: float = 0.25
    w_significance: float = 0.15
    w_fit: float = 0.10
    # --- logit + sélection modèle ---
    force_logit_target: bool = True       # logit(y) si y ∈ (0,1)
    auto_select_model: bool = False       # True = rang #1 auto, False = l'utilisateur choisit
    auto_accept_scenarios: bool = False   # True = accepter sélection auto scénarios sans prompt
    # --- output ---
    top_n_models: int = 5

    # --- misc ---
    random_state: int = 42

    # --- CT (short-term) mode ---
    ngfs_mode: str = "LT"                    # "LT" | "CT"
    ngfs_ct_path: Optional[str] = None       # GEM-E3 file path

    # --- Regional data for CT (évite le rebasing pays) ---
    # use_regional_data_ct=True : utiliser données WB régionales au lieu pays
    # ct_region_code=None       : auto-dérivé depuis le pays (voir variable_resolver)
    # ct_region_code="MNA"      : forcer une région spécifique
    use_regional_data_ct: bool = False        # False = comportement par défaut (données pays)
    ct_region_code: Optional[str] = None     # None = auto-dérivé depuis cfg.country

    # --- Projection anchoring ---
    # projection_start_year=None : auto-dérivé = max(années hist_df) + 1
    # projection_start_year=2025 : forcer une année spécifique
    projection_start_year: Optional[int] = None

    # --- Forced scenario pair (set by web UI after user selection) ---
    forced_adverse_scenario: Optional[str] = None
    forced_severe_scenario:  Optional[str] = None
    forced_model_rank: Optional[int] = None     # 0-based rank from model table

    # --- Model stress validator (4 hard filters + soft scoring) ---
    pd_min_plausible: float = 0.001          # F1: PD min plausible (0.1%)
    pd_max_plausible: float = 0.30           # F1: PD max plausible (30%)
    pd_saturation_tol: float = 0.001         # F4: tolérance aux bornes 0/1
    min_sign_consistency: float = 1.0        # F2: % signes économiques corrects (1.0 = 100%)
    min_stress_delta: float = 0.001          # F3: Δ minimum baseline → stress
    min_stress_scenarios: int = 2            # F3: nb min de scénarios stressants
    enable_stress_validator: bool = True     # activer/désactiver le filtre stress

    # --- F0: qualité statistique minimale du modèle satellite ---
    min_r2: float = 0.10                     # F0a: R² minimum (0 = désactivé)
    min_pvalue_any: float = 0.20             # F0b: au moins 1 variable avec p < seuil

    # --- Stage 1: divergence inter-scénarios (Option B) ---
    s1_min_scenario_divergence: float = 0.0  # 0 = désactivé ; > 0 = seuil CV inter-scénarios

    # --- Baseline coherence checks ---
    baseline_max_annual_pd_change: float = 0.15   # C2: variation max annuelle PD Baseline (ex. 15%)
    baseline_max_pd_value: float = 0.40            # C1: PD Baseline max acceptable en projection
    baseline_min_pd: float = 0.001                 # C0: PD moyenne minimale — en dessous = dégénéré

    # --- Cross-risk unanimous variable selection (climate orchestration only) ---
    # When set by climate/wrapper.py, restricts Stage 2's candidate historical
    # columns to variables independently accepted by credit (PD) AND at least
    # one liquidity target AND at least one market beta factor — so the same
    # macro shock story drives all three risk types, instead of each module
    # picking its own best-fit variables in isolation. None = no restriction
    # (default; unaffected for standalone, non-climate credit runs).
    restrict_candidate_vars: Optional[List[str]] = None

@dataclass
class VariableTransform:
    """
    Décrit une transformation appliquée à une variable historique.
    Doit être rejouée sur la variable NGFS correspondante
    lors de la projection.

    Catégories :
        'stationarize' → DOIT être appliquée aux NGFS
        'normalize'    → NE DOIT PAS être appliquée aux NGFS
        'lag'          → décalage temporel sur NGFS

    Exemples :
        VariableTransform('Real_GDP', 'log_diff', 'stationarize')
        VariableTransform('policy_rate', 'diff', 'stationarize')
        VariableTransform('Real_GDP', 'lag', 'lag', params={'n': 1})
    """
    variable: str
    method: str          # 'diff' | 'log_diff' | 'pct_change' | 'log' | 'lag'
    category: str        # 'stationarize' | 'normalize' | 'lag'
    params: dict = field(default_factory=dict)


@dataclass
class TransformRegistry:
    """
    Registre de toutes les transformations appliquées pendant Stage 2.
    Passé à la couche de projection pour rejouer sur NGFS.
    """
    transforms: List[VariableTransform] = field(default_factory=list)

    def register(self, t: VariableTransform):
        self.transforms.append(t)

    def must_replay_on_ngfs(self) -> List[VariableTransform]:
        """Retourne uniquement les transformations à rejouer sur NGFS."""
        return [t for t in self.transforms
                if t.category in ('stationarize', 'lag')]