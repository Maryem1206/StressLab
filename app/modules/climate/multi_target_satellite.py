"""
multi_target_satellite.py
=========================
MultiTargetSatelliteFactory — calibre un satellite macro→risque pour chaque
variable cible (PD, ΔLCR, ΔNSFR, β₁, β₂, β₃) sur le même socle de 6 variables
macro WEO enrichi de transformations économétriques, avec validation post-
estimation complète pour TOUTES les cibles.

Pour LCR et NSFR on modélise la VARIATION du ratio (Δ en pp) par rapport au
baseline NGFS, pas le niveau absolu.  Le niveau absolu complet est produit par
ngfs_liquidity_engine.py (pipeline composantes via LiquidityStressEngine).

Modèles supportés :
  • "OLS" — y_t = α + Σ γ_j X_j,t + ε_t
            (PD, ΔLCR, ΔNSFR — données annuelles)
  • "AR1" — y_t = α + ρ y_{t-1} + Σ γ_j X_j,t + ε_t
            (β₁, β₂, β₃ NS — séries avec persistance marquée)

Espace de features (CLIMATE_MACRO_SOCLE + TRANSFORMED_FEATURES = 11 candidats) :
  Variables originales  : real_gdp_growth, cpi_inflation, policy_rate,
                          unemployment_rate, exchange_rate_lcu_usd, gov_debt_gdp
  Lags t-1             : real_gdp_growth__lag1, unemployment_rate__lag1
                          → capturent le délai de transmission macro → risque
  Premières différences : policy_rate__delta, cpi_inflation__delta
                          → chocs de taux/inflation > niveaux absolus (PD + betas NS)
  Log                  : exchange_rate_lcu_usd__log
                          → linéarise les effets de change (scale nonlinéaire)

Sélection des variables (4 critères séquentiels) :
  1. Disponibilité  : ≥ MIN_OBS observations communes avec la cible
  2. Corrélation    : |corr| ≥ min_abs_corr (défaut 0.10)
  3. Significativité: p-value < pvalue_threshold (OLS univarié HC1, défaut 0.20)
  4. Signe éco.     : γ estimé ∈ sens attendu par _SIGN_PRIORS (rejet hard si faux)
  5. VIF            : < vif_threshold après sélection (défaut 10.0, filtre itératif)
  6. Harrell cap    : max_k = floor(n_obs / harrell_factor) (défaut 10)

Validation post-estimation (8 tests, 3 batteries) — TOUTES les cibles :
  Battery 1 (universelle) : R²adj+RMSE, Breusch-Godfrey LM, CUSUM, Theil U, RMSE, MAE
  Battery 2 (dispatch)    : RESET Ramsey (OLS) | AR(1) Stationnarité |ρ| (AR1)
  Battery 3 (climat)      : Monotonie scénarios baseline≤adverse≤severe
  Résultat agrégé         : leaderboard() → pd.DataFrame toutes cibles × tous tests

Projection NGFS (sans refit — fonctions pures) :
  OLS : y_t* = α + Σ γ_j · ngfs_j,t
  AR1 : projection récursive, initialisée sur la dernière valeur observée

Cohérence des transformations en projection :
  Les lags et deltas sont recalculés sur la trajectoire NGFS avec gestion
  de la frontière historique (self._last_hist_row) pour t=0 :
    lag1[t=0]   = dernière valeur historique observée
    delta[t=0]  = NGFS[t=0] − dernière valeur historique

Références :
  Harrell F.E. (2015). Regression Modeling Strategies. Springer.
  Diebold & Li (2006). J. Econometrics 130(2):337–364.
  BCBS (2013). Basel III: LCR and liquidity risk monitoring tools (BCBS 238).
  EBA (2023). Methodological Note — EU-wide Stress Test 2023.
  Breusch (1978). Australian Economic Papers 17:334–355.
  Godfrey (1978). Econometrica 46(6):1293–1301.
  Brown, Durbin & Evans (1975). J. R. Stat. Soc. B 37(2):149–192.
  Ramsey (1969). J. R. Stat. Soc. B 31(2):350–371.
  Theil (1966). Applied Economic Forecasting. North-Holland.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

LOG = logging.getLogger("climate.multi_target_satellite")

MIN_OBS = 5  # observations communes minimales pour tenter une calibration

# ─────────────────────────────────────────────────────────────────────────────
# Socle commun — 6 variables macro WEO partagées entre tous les modules
# ─────────────────────────────────────────────────────────────────────────────

CLIMATE_MACRO_SOCLE: List[str] = [
    "real_gdp_growth",        # Croissance PIB réel (% a/a)
    "cpi_inflation",          # Inflation CPI (% a/a)
    "policy_rate",            # Taux directeur banque centrale (%)
    "unemployment_rate",      # Taux de chômage (%)
    "exchange_rate_lcu_usd",  # Taux de change LCU/USD
    "gov_debt_gdp",           # Dette publique / PIB (%)
]

# ─────────────────────────────────────────────────────────────────────────────
# Transformations économétriques du socle
#
# Chaque transformation cible un mécanisme de transmission spécifique :
#
#   __lag1  — délai de transmission (le risque crédit de 2024 reflète le PIB 2023)
#   __delta — choc plutôt que niveau (une hausse brusque de 200bp est plus stressante
#             qu'un régime de taux stable à 5%)
#   __log   — linéarisation d'échelle (un change à 15 LCU/USD vs 1500 LCU/USD :
#             l'impact relatif est similaire → log capte ça, pas le niveau)
#
# Cohérence projection : boundary gérée dans _build_ngfs_feature_matrix() via
# _last_hist_row — les lags/deltas à t=0 utilisent la dernière donnée historique.
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORMED_FEATURES: List[str] = [
    "real_gdp_growth__lag1",       # lag(1) du PIB réel — délai transmission crédit
    "unemployment_rate__lag1",     # lag(1) du chômage  — délai transmission liquidité
    "policy_rate__delta",          # Δtaux directeur    — choc monétaire (betas NS + PD)
    "cpi_inflation__delta",        # Δinflation         — surprise inflationniste (β₁ Fisher)
    "exchange_rate_lcu_usd__log",  # log(FX)            — linéarisation scale change
]

# Cibles AR(1)+macro — les autres utilisent OLS
_AR1_TARGETS = frozenset({"beta1", "beta2", "beta3"})

# ─────────────────────────────────────────────────────────────────────────────
# Échelles RMSE par cible — used in B1.5/B1.6 (PostEstimationValidator)
#
# rmse_scale est le multiplicateur sur le seuil PASS (0.10 × scale).
# Il est ajusté à la plage typique de chaque variable :
#   PD      [0,1] fraction   → scale=1.0  → seuil PASS < 0.10 (10pp de PD)
#   LCR     supposé ratio    → scale=1.0  → seuil PASS < 0.10
#   NSFR    supposé ratio    → scale=1.0
#   beta1   range ~0–20%     → scale=10.0 → seuil PASS < 1.0 (100bp)
#   beta2   range ~±5%       → scale=5.0  → seuil PASS < 0.50 (50bp)
#   beta3   range ~±3%       → scale=3.0  → seuil PASS < 0.30 (30bp)
#
# Si LCR/NSFR sont en pourcentage (ex. 120 pour 120%), multiplier scale par 100.
# ─────────────────────────────────────────────────────────────────────────────
_TARGET_RMSE_SCALE: Dict[str, float] = {
    "pd":    1.0,
    "lcr":   1.0,
    "nsfr":  1.0,
    "beta1": 10.0,
    "beta2": 5.0,
    "beta3": 3.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Signes économiques attendus par (variable du socle × cible)
#
# Convention identique à ECONOMIC_PRIORS (utils.py) :
#   +1  hausse macro → hausse cible
#   -1  hausse macro → baisse cible
#    0  ambigu → signe non pénalisé
#
# Les variantes transformées (__lag1, __delta, __log) héritent du même signe
# que la variable source : le mécanisme économique est identique, seule la
# forme temporelle change.
#
# Sources :
#   pd      → EBA (2023) Methodological Note ; Drehmann (2008)
#   lcr     → BCBS 238 : LCR plus élevé = plus sain
#   nsfr    → BCBS 295 : NSFR plus élevé = financement plus stable
#   beta1/2/3 → Diebold-Li (2006) ; ECONOMIC_PRIORS["market_beta*"] (utils.py)
# ─────────────────────────────────────────────────────────────────────────────

# Mappage colonne (originale ou transformée) → clé dans _SIGN_PRIORS
_SOCLE_TO_PRIOR_KEY: Dict[str, str] = {
    # Variables originales
    "real_gdp_growth":             "gdp_growth",
    "cpi_inflation":               "inflation",
    "policy_rate":                 "policy_rate",
    "unemployment_rate":           "unemployment",
    "exchange_rate_lcu_usd":       "exchange_rate",
    "gov_debt_gdp":                "public_debt",
    # Variants transformées — même prior économique que la variable source
    "real_gdp_growth__lag1":       "gdp_growth",
    "unemployment_rate__lag1":     "unemployment",
    "policy_rate__delta":          "policy_rate",
    "cpi_inflation__delta":        "inflation",
    "exchange_rate_lcu_usd__log":  "exchange_rate",
}

_SIGN_PRIORS: Dict[str, Dict[str, int]] = {
    # ── Crédit : PD (probabilité de défaut) ──────────────────────────────────
    "pd": {
        "gdp_growth":    -1,
        "inflation":      0,
        "policy_rate":   +1,
        "unemployment":  +1,
        "exchange_rate": +1,
        "public_debt":   +1,
    },
    # ── Liquidité : ΔLCR ─────────────────────────────────────────────────────
    "lcr": {
        "gdp_growth":    +1,
        "inflation":      0,
        "policy_rate":   -1,
        "unemployment":  -1,
        "exchange_rate": -1,
        "public_debt":   -1,
    },
    # ── Liquidité : ΔNSFR ────────────────────────────────────────────────────
    "nsfr": {
        "gdp_growth":    +1,
        "inflation":      0,
        "policy_rate":   -1,
        "unemployment":  -1,
        "exchange_rate": -1,
        "public_debt":   -1,
    },
    # ── Marché : β₁ (niveau courbe des taux) ─────────────────────────────────
    "beta1": {
        "gdp_growth":    -1,
        "inflation":     +1,
        "policy_rate":   +1,
        "unemployment":   0,
        "exchange_rate":  0,
        "public_debt":   +1,
    },
    # ── Marché : β₂ (pente) ──────────────────────────────────────────────────
    "beta2": {
        "gdp_growth":    -1,
        "inflation":     +1,
        "policy_rate":   +1,
        "unemployment":   0,
        "exchange_rate":  0,
        "public_debt":   -1,
    },
    # ── Marché : β₃ (courbure) ───────────────────────────────────────────────
    "beta3": {
        "gdp_growth":     0,
        "inflation":      0,
        "policy_rate":    0,
        "unemployment":   0,
        "exchange_rate":  0,
        "public_debt":    0,
    },
}

_TARGET_TO_RISK_TYPE: Dict[str, str] = {
    "pd":    "credit",
    "lcr":   "liquidity",
    "nsfr":  "liquidity",
    "beta1": "market_beta1",
    "beta2": "market_beta2",
    "beta3": "market_beta3",
}


def _expected_sign(target: str, socle_col: str) -> int:
    """Retourne le signe économique attendu (+1 / -1 / 0) pour (cible, variable).

    Fonctionne pour les colonnes originales ET les variantes transformées
    (__lag1, __delta, __log) via _SOCLE_TO_PRIOR_KEY.
    """
    prior_key = _SOCLE_TO_PRIOR_KEY.get(socle_col, socle_col)
    local = _SIGN_PRIORS.get(target, {})
    if prior_key in local:
        return local[prior_key]
    try:
        _src = str(
            Path(__file__).resolve().parent.parent.parent.parent
            / "modules_src" / "climate_module"
        )
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from macro_selection_engine.utils import expected_sign
        risk_type = _TARGET_TO_RISK_TYPE.get(target, "credit")
        return expected_sign(prior_key, risk_type)
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Import conditionnel du validateur post-estimation
# ─────────────────────────────────────────────────────────────────────────────

try:
    from ..core.post_estimation_validator import (
        PostEstimationValidator,
        RiskOutput as _RiskOutput,
        ValidationReport as _ValidationReport,
    )
    _VALIDATOR_AVAILABLE = True
except ImportError:
    try:
        import importlib, os
        _core_path = str(Path(__file__).resolve().parent.parent / "core")
        if _core_path not in sys.path:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from core.post_estimation_validator import (
            PostEstimationValidator,
            RiskOutput as _RiskOutput,
            ValidationReport as _ValidationReport,
        )
        _VALIDATOR_AVAILABLE = True
    except ImportError:
        _VALIDATOR_AVAILABLE = False
        PostEstimationValidator = None   # type: ignore
        _RiskOutput = None               # type: ignore
        _ValidationReport = None         # type: ignore

if not _VALIDATOR_AVAILABLE:
    LOG.warning(
        "multi_target_satellite: post_estimation_validator non importable — "
        "validation post-estimation désactivée (leaderboard vide)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Types de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SatelliteResult:
    """Résultat de calibration d'un satellite macro → variable de risque.

    Les champs *_is (in-sample) et fitted_sm sont remplis automatiquement
    et alimentent PostEstimationValidator dans leaderboard().
    """

    target:        str                # "pd", "lcr", "nsfr", "beta1", "beta2", "beta3"
    family:        str                # "OLS" | "AR1"
    intercept:     float              # α
    rho:           float              # ρ AR(1) ; 0.0 pour OLS
    gammas:        Dict[str, float]   # {var_name: γ} — coefficients figés (jamais refit)
    selected_vars: List[str]
    sigma:         float              # std résidus in-sample
    r2:            float
    n_obs:         int
    max_k:         int                # Harrell cap effectif
    last_value:    float              # dernière valeur observée (init projection AR1)
    warnings:      List[str] = field(default_factory=list)
    # ── Données in-sample pour validation post-estimation ────────────────────
    y_true_is:     Optional[np.ndarray] = field(default=None, repr=False)
    y_pred_is:     Optional[np.ndarray] = field(default=None, repr=False)
    residuals_is:  Optional[np.ndarray] = field(default=None, repr=False)
    fitted_sm:     Optional[Any]        = field(default=None, repr=False)

    def predict_ols(self, macro_row: Dict[str, float]) -> float:
        """α + Σ γ_j X_j pour une ligne macro (originale ou transformée)."""
        val = self.intercept
        for var, gamma in self.gammas.items():
            val += gamma * float(macro_row.get(var, 0.0))
        return val

    def predict_ar1_next(self, prev_value: float, macro_row: Dict[str, float]) -> float:
        """α + ρ y_{t-1} + Σ γ_j X_j,t."""
        val = self.intercept + self.rho * float(prev_value)
        for var, gamma in self.gammas.items():
            val += gamma * float(macro_row.get(var, 0.0))
        return val


@dataclass
class FactoryReport:
    """Résumé de la calibration multi-cibles, incluant les tests post-estimation."""
    n_requested:       int = 0
    n_fitted:          int = 0
    details:           Dict[str, dict] = field(default_factory=dict)
    validation_reports: Dict[str, dict] = field(default_factory=dict)
    # details[target] = {family, selected_vars, r2, n_obs, max_k, warnings}
    # validation_reports[target] = ValidationReport.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Factory principale
# ─────────────────────────────────────────────────────────────────────────────

class MultiTargetSatelliteFactory:
    """
    Calibre un satellite macro→risque par variable cible + validation post-
    estimation complète pour PD, LCR, NSFR, β₁, β₂, β₃.

    L'espace de features comprend le CLIMATE_MACRO_SOCLE (6 variables WEO)
    enrichi des TRANSFORMED_FEATURES (5 transformations) soit 11 candidats au
    total.  La sélection en 4 critères (corrélation, p-value, signe éco, VIF +
    Harrell) choisit le sous-ensemble optimal pour chaque cible.

    Toutes les cibles sont calibrées sur leurs niveaux historiques observés :
    PD (fraction), LCR/NSFR (%), betas NS (unités absolues).
    Les deltas ΔLCR, ΔNSFR, ΔPD, Δβ sont calculés dans ClimateMacroAdapter
    par soustraction (proj_stressed − proj_baseline).

    Post-estimation :
    Après calibration de chaque satellite, 8 tests statistiques (3 batteries)
    sont exécutés via PostEstimationValidator :
      Battery 1 : R²adj, Breusch-Godfrey LM, CUSUM, Theil U, RMSE, MAE
      Battery 2 : RESET Ramsey (OLS) ou AR(1) Stationnarité |ρ| (AR1)
      Battery 3 : Monotonie scénarios (N/A à ce stade — peuplée en projection)
    Appeler leaderboard() après fit_all() pour obtenir le tableau récapitulatif
    de TOUS les tests pour TOUTES les cibles.

    Parameters
    ----------
    macro_df : pd.DataFrame
        DataFrame WEO indexé par année (int), colonnes ⊇ CLIMATE_MACRO_SOCLE.
    min_abs_corr, pvalue_threshold, vif_threshold, harrell_factor : voir module docstring.
    """

    def __init__(
        self,
        macro_df: pd.DataFrame,
        min_abs_corr: float = 0.10,
        pvalue_threshold: float = 0.20,
        vif_threshold: float = 10.0,
        harrell_factor: int = 10,
    ) -> None:
        self.min_abs_corr     = float(min_abs_corr)
        self.pvalue_threshold = float(pvalue_threshold)
        self.vif_threshold    = float(vif_threshold)
        self.harrell_factor   = max(1, int(harrell_factor))

        available_socle = [c for c in CLIMATE_MACRO_SOCLE if c in macro_df.columns]
        if not available_socle:
            LOG.warning(
                "MultiTargetSatelliteFactory: aucune colonne du socle trouvée "
                "(colonnes reçues: %s)", list(macro_df.columns)
            )

        raw = macro_df[available_socle].copy()
        raw.index = raw.index.astype(int)
        raw.sort_index(inplace=True)

        # Dernière ligne historique — frontière de projection pour lags/deltas
        self._last_hist_row: pd.Series = raw.iloc[-1].copy()

        # Matrice de features augmentée (socle + transformations)
        augmented = self._build_feature_matrix(raw)
        all_expected = CLIMATE_MACRO_SOCLE + TRANSFORMED_FEATURES
        available_all = [c for c in all_expected if c in augmented.columns]
        self._macro_df = augmented[available_all].copy()

        self._results: Dict[str, SatelliteResult]  = {}
        self._val_reports: Dict[str, Any]           = {}   # ValidationReport objects
        self._report = FactoryReport()

        LOG.info(
            "MultiTargetSatelliteFactory: %d/%d vars socle, "
            "%d features totales (socle + transformées): %s",
            len(available_socle), len(CLIMATE_MACRO_SOCLE),
            len(available_all), available_all,
        )

    # ── API publique ──────────────────────────────────────────────────────────

    def fit_all(
        self,
        targets: Dict[str, pd.Series],
    ) -> Dict[str, SatelliteResult]:
        """
        Calibre un satellite pour chaque (nom, série historique) dans `targets`
        puis exécute les tests post-estimation pour chaque satellite calibré.

        Parameters
        ----------
        targets : dict {target_name: pd.Series indexée par année (int)}
            Noms reconnus : "pd", "lcr", "nsfr", "beta1", "beta2", "beta3"

        Returns
        -------
        dict {target_name: SatelliteResult}
        """
        self._report.n_requested = len(targets)

        for name, series in targets.items():
            family = "AR1" if name in _AR1_TARGETS else "OLS"
            try:
                result = self._fit_one(name, series, family)
                self._results[name] = result
                self._report.n_fitted += 1
                self._report.details[name] = {
                    "family":        result.family,
                    "selected_vars": result.selected_vars,
                    "r2":            round(result.r2, 4),
                    "n_obs":         result.n_obs,
                    "max_k":         result.max_k,
                    "warnings":      result.warnings,
                }
                LOG.info(
                    "Satellite [%s/%s]: vars=%s  R²=%.3f  n=%d  max_k=%d",
                    name, family, result.selected_vars,
                    result.r2, result.n_obs, result.max_k,
                )
                # ── Tests post-estimation ──────────────────────────────────
                val_report = self._run_post_estimation(result)
                if val_report is not None:
                    self._val_reports[name] = val_report
                    self._report.validation_reports[name] = val_report.to_dict()
                    LOG.info(
                        "Post-estimation [%s]: %s — PASS:%d WARN:%d FAIL:%d",
                        name, val_report.verdict,
                        val_report.n_pass, val_report.n_warn, val_report.n_fail,
                    )

            except Exception as exc:
                LOG.warning("Satellite [%s]: calibration échouée — %s", name, exc)
                self._report.details[name] = {"error": str(exc)}

        LOG.info(
            "MultiTargetSatelliteFactory: %d/%d satellites calibrés | "
            "%d rapports post-estimation disponibles",
            self._report.n_fitted, self._report.n_requested,
            len(self._val_reports),
        )
        return dict(self._results)

    def project(
        self,
        ngfs_df: pd.DataFrame,
        scenario_alias: str = "adverse",
    ) -> Dict[str, np.ndarray]:
        """
        Projette chaque satellite calibré sur les trajectoires NGFS (zéro refit).

        Les mêmes transformations appliquées lors de la calibration sont
        recalculées sur la trajectoire NGFS via _build_ngfs_feature_matrix(),
        avec gestion correcte de la frontière historique (t=0).
        """
        if not self._results:
            LOG.warning("project() appelé avant fit_all() — retourne dict vide")
            return {}

        ngfs_augmented = self._build_ngfs_feature_matrix(ngfs_df)
        years = sorted(ngfs_augmented.index.astype(int).tolist())
        out: Dict[str, np.ndarray] = {}

        for name, sat in self._results.items():
            try:
                projected = self._project_one(sat, ngfs_augmented, years)
                out[name] = projected
                LOG.info(
                    "Projection [%s/%s] scén=%s: %d années | "
                    "mean=%.4f  peak=%.4f",
                    name, sat.family, scenario_alias, len(projected),
                    float(projected.mean()), float(projected.max()),
                )
            except Exception as exc:
                LOG.warning(
                    "Projection satellite [%s/%s] échouée: %s",
                    name, scenario_alias, exc,
                )
        return out

    def update_monotony_test(
        self,
        scenario_projections: Dict[str, Dict[str, np.ndarray]],
    ) -> None:
        """
        Met à jour le test B3.3 (Monotonie scénarios) pour chaque satellite
        une fois que les projections NGFS sont disponibles.

        Appelée par ClimateMacroAdapter après prepare_satellite_projections().

        Parameters
        ----------
        scenario_projections : dict {scenario_alias: {target: np.ndarray}}
            Projections par scénario et par cible.
            Les alias "baseline", "adverse", "severe" sont attendus.
        """
        if not _VALIDATOR_AVAILABLE:
            return

        baseline = scenario_projections.get("baseline") or scenario_projections.get("Baseline", {})
        adverse  = scenario_projections.get("adverse")  or scenario_projections.get("Adverse", {})
        severe   = scenario_projections.get("severe")   or scenario_projections.get("Severe", {})

        for name, sat in self._results.items():
            if name not in self._val_reports:
                continue

            scen_res = {}
            if name in baseline:
                scen_res["baseline"] = np.asarray(baseline[name], dtype=float)
            if name in adverse:
                scen_res["adverse"] = np.asarray(adverse[name], dtype=float)
            if name in severe:
                scen_res["severe"] = np.asarray(severe[name], dtype=float)

            if len(scen_res) < 2:
                continue  # pas assez de scénarios pour tester la monotonie

            # Reconstruire RiskOutput avec scenario_results et relancer le validateur
            try:
                ro = self._build_risk_output(sat, scenario_results=scen_res)
                val = PostEstimationValidator(ro).run()
                self._val_reports[name] = val
                self._report.validation_reports[name] = val.to_dict()
                LOG.info(
                    "Post-estimation B3.3 [%s] mis à jour: %s "
                    "— PASS:%d WARN:%d FAIL:%d",
                    name, val.verdict,
                    val.n_pass, val.n_warn, val.n_fail,
                )
            except Exception as exc:
                LOG.warning(
                    "update_monotony_test [%s]: échec — %s", name, exc
                )

    def leaderboard(self) -> pd.DataFrame:
        """
        Construit le tableau de bord récapitulatif de TOUS les tests post-
        estimation pour l'ensemble des satellites calibrés.

        Chaque ligne représente un (satellite × test).
        Les colonnes incluent le verdict global du satellite sur chaque ligne
        pour permettre un tri/filtrage immédiat.

        Returns
        -------
        pd.DataFrame — vide si aucun rapport disponible.
        Colonnes :
            Satellite | Famille | n_obs | R² | Batterie | Test ID | Test
            Statistique | p-value | Seuil | Statut | Message | Verdict
        Trié par : Satellite (ordre calibration), Batterie, Test ID.
        """
        if not self._val_reports:
            LOG.info("leaderboard(): aucun rapport post-estimation disponible.")
            return pd.DataFrame()

        rows = []
        # Conserver l'ordre de calibration (insertion order dict Python 3.7+)
        for name in self._results:
            sat = self._results[name]
            val = self._val_reports.get(name)
            if val is None:
                continue
            r2_fmt   = f"{sat.r2:.4f}"
            verdict  = val.verdict
            n_obs    = val.n_obs

            for tr in val.results:
                rows.append({
                    "Satellite":   name,
                    "Famille":     sat.family,
                    "n_obs":       n_obs,
                    "R²":          r2_fmt,
                    "Batterie":    tr.battery,
                    "Test ID":     tr.test_id,
                    "Test":        tr.test_name,
                    "Statistique": (
                        f"{tr.statistic:.6f}" if tr.statistic is not None else "—"
                    ),
                    "p-value":     (
                        f"{tr.p_value:.4f}" if tr.p_value is not None else "—"
                    ),
                    "Seuil":       tr.threshold,
                    "Statut":      tr.status,
                    "Message":     tr.message,
                    "Verdict":     verdict,
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Tri : ordre de calibration préservé par merge sort stable
        df = df.sort_values(
            ["Satellite", "Batterie", "Test ID"],
            kind="stable",
        ).reset_index(drop=True)
        return df

    def leaderboard_summary(self) -> pd.DataFrame:
        """
        Version condensée du leaderboard : une ligne par satellite avec le
        décompte PASS/WARN/FAIL et le verdict global.

        Returns
        -------
        pd.DataFrame — colonnes :
            Satellite | Famille | n_obs | R² | n_tests | PASS | WARN | FAIL | Verdict
        Trié par : Verdict (REJECTED → VALIDATED_WITH_WARNINGS → VALIDATED),
                   puis R² décroissant.
        """
        if not self._val_reports:
            return pd.DataFrame()

        rows = []
        _VERDICT_ORDER = {"REJECTED": 0, "VALIDATED_WITH_WARNINGS": 1, "VALIDATED": 2}

        for name in self._results:
            sat = self._results[name]
            val = self._val_reports.get(name)
            if val is None:
                continue
            n_tested = val.n_pass + val.n_warn + val.n_fail
            rows.append({
                "Satellite": name,
                "Famille":   sat.family,
                "n_obs":     val.n_obs,
                "R²":        round(sat.r2, 4),
                "n_tests":   n_tested,
                "PASS":      val.n_pass,
                "WARN":      val.n_warn,
                "FAIL":      val.n_fail,
                "Verdict":   val.verdict,
                "_v_order":  _VERDICT_ORDER.get(val.verdict, 9),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values(
            ["_v_order", "R²"],
            ascending=[True, False],
        ).drop(columns=["_v_order"]).reset_index(drop=True)
        return df

    @property
    def report(self) -> FactoryReport:
        return self._report

    def get(self, target: str) -> Optional[SatelliteResult]:
        return self._results.get(target)

    def get_validation_report(self, target: str) -> Optional[Any]:
        """Retourne le ValidationReport pour une cible donnée (ou None)."""
        return self._val_reports.get(target)

    # ── Construction des features ─────────────────────────────────────────────

    @staticmethod
    def _build_feature_matrix(
        macro_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Augmente le DataFrame macro historique avec TRANSFORMED_FEATURES.

        Appelée une seule fois dans __init__() sur les données historiques.
        Les lags et deltas sont calculés par pandas .shift()/.diff() sur
        l'index trié — NaN à la première ligne filtré par obs communes.
        """
        df = macro_df.copy()

        # Lags t-1 — délai de transmission macro → risque
        for col, lag_col in [
            ("real_gdp_growth",   "real_gdp_growth__lag1"),
            ("unemployment_rate", "unemployment_rate__lag1"),
        ]:
            if col in df.columns:
                df[lag_col] = df[col].shift(1)

        # Premières différences — chocs > niveaux absolus
        for col, delta_col in [
            ("policy_rate",   "policy_rate__delta"),
            ("cpi_inflation", "cpi_inflation__delta"),
        ]:
            if col in df.columns:
                df[delta_col] = df[col].diff()

        # Log — linéarisation scale effet de change
        if "exchange_rate_lcu_usd" in df.columns:
            df["exchange_rate_lcu_usd__log"] = np.log(
                df["exchange_rate_lcu_usd"].clip(lower=1e-6)
            )

        return df

    def _build_ngfs_feature_matrix(
        self,
        ngfs_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Augmente le DataFrame NGFS projeté avec les mêmes transformations,
        en gérant correctement la frontière historique (self._last_hist_row).

        Frontière historique :
          lag1[t=0]   = last_hist_row[col]
          delta[t=0]  = NGFS[t=0] − last_hist_row[col]
        Pour t ≥ 1 : lags/deltas calculés depuis la trajectoire NGFS.
        """
        df = ngfs_df.copy()
        df.index = df.index.astype(int)
        df.sort_index(inplace=True)
        years = df.index.tolist()
        hist = self._last_hist_row

        # Lags t-1
        for col, lag_col in [
            ("real_gdp_growth",   "real_gdp_growth__lag1"),
            ("unemployment_rate", "unemployment_rate__lag1"),
        ]:
            if col not in df.columns:
                continue
            lag_vals: Dict[int, float] = {}
            for i, yr in enumerate(years):
                if i == 0:
                    lag_vals[yr] = float(hist.get(col, np.nan))
                else:
                    prev_yr = years[i - 1]
                    v = df.at[prev_yr, col] if prev_yr in df.index else np.nan
                    lag_vals[yr] = float(v)
            df[lag_col] = pd.Series(lag_vals, dtype=float)

        # Premières différences
        for col, delta_col in [
            ("policy_rate",   "policy_rate__delta"),
            ("cpi_inflation", "cpi_inflation__delta"),
        ]:
            if col not in df.columns:
                continue
            delta_vals: Dict[int, float] = {}
            for i, yr in enumerate(years):
                curr = float(df.at[yr, col]) if yr in df.index else np.nan
                if i == 0:
                    hist_val = float(hist.get(col, np.nan))
                    delta_vals[yr] = curr - hist_val
                else:
                    prev_yr  = years[i - 1]
                    prev_val = float(df.at[prev_yr, col]) if prev_yr in df.index else np.nan
                    delta_vals[yr] = curr - prev_val
            df[delta_col] = pd.Series(delta_vals, dtype=float)

        # Log
        if "exchange_rate_lcu_usd" in df.columns:
            df["exchange_rate_lcu_usd__log"] = np.log(
                df["exchange_rate_lcu_usd"].clip(lower=1e-6)
            )

        return df

    # ── Calibration interne ───────────────────────────────────────────────────

    def _fit_one(
        self,
        name: str,
        target_series: pd.Series,
        family: str,
    ) -> SatelliteResult:
        """Calibre un satellite pour une cible."""
        warns: List[str] = []

        target = target_series.dropna().copy()
        target.index = target.index.astype(int)

        common_idx = target.index.intersection(self._macro_df.index)
        if len(common_idx) < MIN_OBS:
            raise ValueError(
                f"{len(common_idx)} observations communes cible×macro (min {MIN_OBS})"
            )

        y      = target.reindex(common_idx).astype(float)
        X_full = self._macro_df.reindex(common_idx).copy()

        n_obs = len(y)
        max_k = max(1, int(np.floor(n_obs / self.harrell_factor)))

        LOG.info(
            "Satellite [%s/%s]: n_obs=%d → Harrell max_k=%d "
            "sur %d features candidates",
            name, family, n_obs, max_k, X_full.shape[1],
        )

        selected = self._select_variables(name, y, X_full, max_k, family, warns)

        if family == "AR1":
            (intercept, rho, gammas, resid, r2,
             y_true_arr, y_pred_arr, sm_result) = self._fit_ar1(y, X_full, selected)
        else:
            (intercept, gammas, resid, r2,
             y_true_arr, y_pred_arr, sm_result) = self._fit_ols(y, X_full, selected)
            rho = 0.0

        sigma      = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
        last_value = float(y.iloc[-1])

        return SatelliteResult(
            target        = name,
            family        = family,
            intercept     = intercept,
            rho           = rho,
            gammas        = gammas,
            selected_vars = list(gammas.keys()),
            sigma         = sigma,
            r2            = r2,
            n_obs         = n_obs,
            max_k         = max_k,
            last_value    = last_value,
            warnings      = warns,
            y_true_is     = y_true_arr,
            y_pred_is     = y_pred_arr,
            residuals_is  = np.asarray(resid, dtype=float),
            fitted_sm     = sm_result,
        )

    def _select_variables(
        self,
        name: str,
        y: pd.Series,
        X: pd.DataFrame,
        max_k: int,
        family: str,
        warns: List[str],
    ) -> List[str]:
        """
        Sélection en 4 critères séquentiels sur l'espace de features augmenté.
        Critère #3 (signe éco) fonctionne pour originaux ET transformés
        via _SOCLE_TO_PRIOR_KEY.
        """
        if family == "AR1" and len(y) > 2:
            y_sel = y.diff().dropna()
            X_sel = X.reindex(y_sel.index)
        else:
            y_sel = y
            X_sel = X

        candidates: List[Tuple[str, float]] = []

        for col in X_sel.columns:
            x      = X_sel[col].dropna()
            common = y_sel.index.intersection(x.index)
            if len(common) < MIN_OBS:
                LOG.debug("Satellite [%s] var=%s SKIP obs<MIN", name, col)
                continue

            yc = y_sel.reindex(common).values.astype(float)
            xc = x.reindex(common).values.astype(float)

            # Critère 1 : corrélation absolue
            try:
                corr = float(np.corrcoef(yc, xc)[0, 1])
            except Exception:
                continue
            if np.isnan(corr) or abs(corr) < self.min_abs_corr:
                LOG.debug(
                    "Satellite [%s] var=%s REJETÉ corr=%.3f < %.2f",
                    name, col,
                    abs(corr) if not np.isnan(corr) else 0.0,
                    self.min_abs_corr,
                )
                continue

            # Critère 2 : p-value OLS univarié HC1
            try:
                X_uni     = sm.add_constant(xc.reshape(-1, 1), has_constant="add")
                res_uni   = sm.OLS(yc, X_uni).fit(cov_type="HC1")
                pval      = float(res_uni.pvalues[1])
                gamma_est = float(res_uni.params[1])
            except Exception:
                continue
            if pval >= self.pvalue_threshold:
                LOG.debug(
                    "Satellite [%s] var=%s REJETÉ p=%.4f ≥ %.2f",
                    name, col, pval, self.pvalue_threshold,
                )
                continue

            # Critère 3 : signe économique (fonctionne pour variantes transformées)
            exp_sign = _expected_sign(name, col)
            if exp_sign != 0:
                sign_ok = (
                    (gamma_est > 0 and exp_sign > 0)
                    or (gamma_est < 0 and exp_sign < 0)
                )
                if not sign_ok:
                    prior_key = _SOCLE_TO_PRIOR_KEY.get(col, col)
                    msg = (
                        f"Satellite [{name}] var={col}: signe incohérent — "
                        f"γ={'+'if gamma_est>0 else ''}{gamma_est:.4f} "
                        f"attendu {'+'if exp_sign>0 else '-'}1 "
                        f"(_SIGN_PRIORS['{name}']['{prior_key}'])"
                    )
                    LOG.warning(msg)
                    warns.append(msg)
                    continue

            candidates.append((col, abs(corr)))
            LOG.debug(
                "Satellite [%s] var=%s RETENU corr=%.3f p=%.4f γ=%+.4f sign_ok=%s",
                name, col, abs(corr), pval, gamma_est,
                "N/A(prior=0)" if exp_sign == 0 else "OK",
            )

        if not candidates:
            LOG.info(
                "Satellite [%s]: aucune variable retenue → modèle pur intercept/AR(1)",
                name,
            )
            return []

        # Harrell cap
        candidates.sort(key=lambda t: -t[1])
        selected = [c[0] for c in candidates[:max_k]]

        # VIF post-sélection — élimine aussi les paires (original + transformé) trop colinéaires
        if len(selected) > 1:
            selected = self._apply_vif_filter(name, selected, X.reindex(y.index), warns)

        LOG.info(
            "Satellite [%s]: %d var(s) retenue(s) / %d candidates: %s",
            name, len(selected), len(candidates), selected,
        )
        return selected

    def _apply_vif_filter(
        self,
        name: str,
        selected: List[str],
        X: pd.DataFrame,
        warns: List[str],
    ) -> List[str]:
        """Retire itérativement la variable avec le VIF le plus élevé > seuil."""
        remaining = list(selected)
        while len(remaining) > 1:
            sub = X[remaining].dropna()
            if len(sub) < MIN_OBS:
                break
            vifs      = _compute_vif(sub)
            worst_var = max(vifs, key=vifs.get)
            worst_vif = vifs[worst_var]
            if worst_vif <= self.vif_threshold:
                break
            msg = (
                f"Satellite [{name}] VIF: '{worst_var}' retiré "
                f"(VIF={worst_vif:.1f} > {self.vif_threshold:.1f})"
            )
            LOG.warning(msg)
            warns.append(msg)
            remaining.remove(worst_var)
        return remaining

    # ── Estimation OLS ────────────────────────────────────────────────────────

    def _fit_ols(
        self,
        y: pd.Series,
        X_full: pd.DataFrame,
        selected: List[str],
    ) -> Tuple[float, Dict[str, float], np.ndarray, float, np.ndarray, np.ndarray, Any]:
        """OLS final. Retourne (α, gammas, résidus, R², y_true, y_pred, sm_result)."""
        if not selected:
            alpha   = float(y.mean())
            y_arr   = y.values.astype(float)
            y_pred  = np.full_like(y_arr, alpha)
            return alpha, {}, (y_arr - alpha), 0.0, y_arr, y_pred, None

        sub    = X_full[selected].copy()
        common = y.index.intersection(sub.dropna().index)
        yc     = y.reindex(common).values.astype(float)
        Xc     = sub.reindex(common).values
        X_mat  = sm.add_constant(Xc, has_constant="add")

        try:
            res    = sm.OLS(yc, X_mat).fit(cov_type="HC1")
            alpha  = float(res.params[0])
            gammas = {v: float(res.params[i + 1]) for i, v in enumerate(selected)}
            y_pred = np.asarray(res.fittedvalues, dtype=float)
            return alpha, gammas, res.resid, float(res.rsquared), yc, y_pred, res
        except Exception as exc:
            raise RuntimeError(f"OLS final échoué pour '{selected}': {exc}") from exc

    # ── Estimation AR(1)+macro ────────────────────────────────────────────────

    def _fit_ar1(
        self,
        y: pd.Series,
        X_full: pd.DataFrame,
        selected: List[str],
    ) -> Tuple[float, float, Dict[str, float], np.ndarray, float, np.ndarray, np.ndarray, Any]:
        """AR(1)+macro. Retourne (α, ρ, gammas, résidus, R², y_true, y_pred, sm_result)."""
        _RHO_MAX = 0.99

        y_lag       = y.shift(1).dropna()
        common_base = y.index.intersection(y_lag.index)

        if not selected:
            yc    = y.reindex(common_base).values.astype(float)
            lag_c = y_lag.reindex(common_base).values.astype(float)
            X_mat = sm.add_constant(lag_c.reshape(-1, 1), has_constant="add")
            try:
                res    = sm.OLS(yc, X_mat).fit(cov_type="HC1")
                rho    = max(-_RHO_MAX, min(_RHO_MAX, float(res.params[1])))
                y_pred = np.asarray(res.fittedvalues, dtype=float)
                return (float(res.params[0]), rho, {}, res.resid,
                        float(res.rsquared), yc, y_pred, res)
            except Exception as exc:
                raise RuntimeError(f"AR(1) pur échoué: {exc}") from exc

        common = common_base.intersection(X_full[selected].dropna().index)
        if len(common) < MIN_OBS:
            raise ValueError(
                f"Observations insuffisantes AR(1)+macro ({len(common)} < {MIN_OBS})"
            )

        yc    = y.reindex(common).values.astype(float)
        lag_c = y_lag.reindex(common).values.astype(float)
        Xc    = X_full[selected].reindex(common).values
        X_mat = sm.add_constant(np.column_stack([lag_c, Xc]), has_constant="add")

        try:
            res    = sm.OLS(yc, X_mat).fit(cov_type="HC1")
            alpha  = float(res.params[0])
            rho    = max(-_RHO_MAX, min(_RHO_MAX, float(res.params[1])))
            gammas = {v: float(res.params[i + 2]) for i, v in enumerate(selected)}
            gammas = {v: g for v, g in gammas.items() if abs(g) >= 1e-8}
            y_pred = np.asarray(res.fittedvalues, dtype=float)
            return alpha, rho, gammas, res.resid, float(res.rsquared), yc, y_pred, res
        except Exception as exc:
            raise RuntimeError(f"AR(1)+macro échoué: {exc}") from exc

    # ── Projection sans refit ─────────────────────────────────────────────────

    def _project_one(
        self,
        sat: SatelliteResult,
        ngfs_df: pd.DataFrame,
        years: List[int],
    ) -> np.ndarray:
        """
        Projette un satellite sur les années NGFS. Fonction pure (zéro refit).

        ngfs_df doit être le DataFrame AUGMENTÉ (résultat de
        _build_ngfs_feature_matrix) pour que les features transformées
        (__lag1, __delta, __log) soient correctement disponibles.
        """
        projected = np.empty(len(years), dtype=float)

        if sat.family == "AR1":
            prev = sat.last_value
            for i, yr in enumerate(years):
                row          = _extract_macro_row(ngfs_df, yr, sat.selected_vars)
                val          = sat.predict_ar1_next(prev, row)
                projected[i] = val
                prev         = val
        else:
            for i, yr in enumerate(years):
                row          = _extract_macro_row(ngfs_df, yr, sat.selected_vars)
                projected[i] = sat.predict_ols(row)

        return projected

    # ── Validation post-estimation ────────────────────────────────────────────

    def _build_risk_output(
        self,
        sat: SatelliteResult,
        scenario_results: Optional[Dict[str, np.ndarray]] = None,
    ) -> Any:
        """
        Construit un RiskOutput pour PostEstimationValidator à partir d'un
        SatelliteResult calibré.

        Le module est "climate" pour toutes les cibles.
        Le model_type est mappé selon la famille du satellite :
          OLS → "ols_log"
          AR1 → "ar1_macro"
        scenario_results est None par défaut → B3.3 retourne N/A à la
        calibration et est mis à jour par update_monotony_test() en projection.
        """
        if not _VALIDATOR_AVAILABLE or sat.y_true_is is None:
            return None

        # rmse_scale : scale par défaut de la table, affiné dynamiquement si
        # la plage observée est très différente du défaut
        base_scale = _TARGET_RMSE_SCALE.get(sat.target, 1.0)
        if sat.y_true_is is not None and len(sat.y_true_is) > 1:
            y_range = float(np.max(sat.y_true_is) - np.min(sat.y_true_is))
            # Si la plage dépasse 10× la valeur par défaut attendue, ajuster
            if y_range > base_scale * 10.0:
                base_scale = y_range / 2.0

        return _RiskOutput(
            module         = "climate",
            model_type     = "AR1" if sat.family == "AR1" else "OLS",
            y_true         = sat.y_true_is,
            y_pred         = sat.y_pred_is,
            residuals      = sat.residuals_is,
            n_obs          = sat.n_obs,
            k_params       = len(sat.selected_vars),
            fitted_model   = sat.fitted_sm,
            rho            = sat.rho if sat.family == "AR1" else None,
            scenario_results = scenario_results,
            rmse_scale     = base_scale,
        )

    def _run_post_estimation(
        self,
        sat: SatelliteResult,
    ) -> Optional[Any]:
        """
        Exécute les 8 tests post-estimation (3 batteries) via PostEstimationValidator
        pour un satellite calibré.

        Battery 1 : R²adj+RMSE, Breusch-Godfrey LM, CUSUM, Theil U, RMSE, MAE
        Battery 2 : RESET Ramsey (OLS) | AR(1) Stationnarité |ρ| (AR1)
        Battery 3 : Monotonie scénarios — N/A ici, peuplée par update_monotony_test()

        Retourne None si le validateur n'est pas importable ou si les données
        in-sample sont manquantes (dégradation silencieuse).
        """
        if not _VALIDATOR_AVAILABLE:
            return None

        ro = self._build_risk_output(sat)
        if ro is None:
            return None

        try:
            return PostEstimationValidator(ro).run()
        except Exception as exc:
            LOG.warning(
                "_run_post_estimation [%s]: PostEstimationValidator échoué — %s",
                sat.target, exc,
            )
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_vif(X: pd.DataFrame) -> Dict[str, float]:
    """Calcule le VIF pour chaque colonne de X."""
    cols = list(X.columns)
    if len(cols) < 2:
        return {cols[0]: 1.0} if cols else {}

    vifs: Dict[str, float] = {}
    for col in cols:
        y_v   = X[col].values.astype(float)
        X_v   = X[[c for c in cols if c != col]].values
        X_mat = sm.add_constant(X_v, has_constant="add")
        try:
            r2  = float(sm.OLS(y_v, X_mat).fit().rsquared)
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 - 1e-10 else float("inf")
        except Exception:
            vif = float("inf")
        vifs[col] = vif
    return vifs


def _extract_macro_row(
    ngfs_df: pd.DataFrame,
    year: int,
    selected_vars: List[str],
) -> Dict[str, float]:
    """
    Extrait une ligne macro NGFS pour une année.

    Fonctionne pour les colonnes originales ET les colonnes transformées
    (__lag1, __delta, __log) car ngfs_df est le DataFrame augmenté produit
    par _build_ngfs_feature_matrix().

    Variable absente ou NaN → 0.0.
    Année hors index → année la plus proche disponible.
    """
    if not selected_vars:
        return {}

    if year in ngfs_df.index:
        src_year = year
    elif not ngfs_df.empty:
        src_year = min(ngfs_df.index, key=lambda y: abs(y - year))
    else:
        return {var: 0.0 for var in selected_vars}

    row: Dict[str, float] = {}
    for var in selected_vars:
        if var in ngfs_df.columns:
            val = ngfs_df.at[src_year, var]
            row[var] = 0.0 if (val != val or val is None) else float(val)
        else:
            row[var] = 0.0
    return row
