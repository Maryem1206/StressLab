"""
liquidity_stress_engine.py
==========================
Moteur de stress test liquidité — Plateforme multi-risques
Calcule LCR (BCBS 238) et NSFR (BCBS 295) sous 3 scénarios :
    - historique   : 1 crise → baseline / adverse / severe
    - paramétrique : chocs manuels sur les 6 variables macro
    - ngfs         : trajectoires climatiques projetées

RSF : 6 catégories granulaires BCBS 295 lues depuis rsf_factors.
Macro : fetchée automatiquement via API (WorldBank → IMF WEO → BIS).
         JAMAIS lue depuis le fichier Excel utilisateur.

Compatible avec SatelliteCalibrator.calibrate_all() / SatelliteResult.predict().
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

LOG = logging.getLogger("liquidity_engine")
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .output_types import RiskOutput
from .behavioural_variable_config import DELTA_CAPS
from .satellite_calibrator import SatelliteCalibrator, SatelliteResult

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES RÉGLEMENTAIRES FIXES (BCBS 238 / BCBS 295)
# ─────────────────────────────────────────────────────────────────────────────
L2A_HAIRCUT_REG   = 0.15   # BCBS 238 §42
L2A_CAP_RATIO     = 40/60  # BCBS 238 §43
WHOLESALE_OUTFLOW = 0.25   # BCBS 238 §105
OFFBS_OUTFLOW     = 0.05   # BCBS 238 §136
INFLOW_CAP        = 0.75   # BCBS 238 §154
# Conservative default — BCBS 295 §35-37 assigns 0% ASF to wholesale funding
# from financial institutions with residual maturity < 6 months. Applied
# uniformly in absence of maturity breakdown data. Not calibrated from any
# specific bank or panel.
WHOLESALE_ASF     = 0.0    # BCBS 295 §35-37 — financement wholesale <6 mois

BREACH_THRESHOLD  = 100.0

# Noms de colonnes macro — MAJUSCULES, compatibles avec SatelliteCalibrator
MACRO_VARS = [
    "GDP_growth", "unemployment_rate", "cpi_inflation",
    "policy_rate", "exchange_rate",
]

# Satellites à prédire
SATELLITE_VARS = [
    "run_off_retail", "run_off_corporate", "haircut_add",
    "asf_factor_retail", "asf_factor_corporate",
]

try:
    from modules_src.climate_module.macro_selection_engine.baseline_projector import (
        project_baseline,
        summarize_fallbacks,
    )
    from .behavioural_variable_config import SATELLITE_VARIABLE_CONFIG
    _PROJ_OK = True
except ImportError as e:
    LOG.error(
        "Projection baseline indisponible (%s) — module liquidité "
        "dégradé : ancrage plat conservé ET clipping de domaine "
        "post-delta désactivé.", e
    )
    _PROJ_OK = False
    SATELLITE_VARIABLE_CONFIG = {}

# ── Classification des items bilanciel par comportement sous stress ────────────
# Passifs de financement : se contractent (déposants fuient, lignes se tarissent)
_LIABILITY_COLS: frozenset = frozenset({
    "retail_deposits", "corporate_deposits", "wholesale_funding",
    "off_bs_commitments", "L1", "L2a", "inflows_contractual", "asf_other",
})
# Actifs : augmentent (tirages sur lignes, accumulation de NPL, rollover)
_ASSET_COLS: frozenset = frozenset({
    "mortgage_retail_loans", "corporate_loans", "npls_other_assets",
    "cash_central_bank", "sovereign_securities", "short_term_interbank",
    "loans", "securities", "other_assets",
})


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS — paramètres d'entrée
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LiquidityInputs:
    """Données bilancielles + macro (API). Pas de macro dans l'Excel."""
    time_series: pd.DataFrame
    rsf_weights: Dict[str, float]
    macro_history: pd.DataFrame


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────
class LiquidityDataLoader:

    REQUIRED_BILAN = [
        "run_off_retail", "run_off_corporate", "haircut_add",
        "asf_factor_retail", "asf_factor_corporate",
        "retail_deposits", "corporate_deposits", "wholesale_funding",
        "off_bs_commitments", "asf_other",
        "L1", "L2a",
        "inflows_contractual",
    ]

    RSF_ASSET_COLS = [
        "cash_central_bank", "sovereign_securities", "short_term_interbank",
        "mortgage_retail_loans", "corporate_loans", "npls_other_assets",
    ]

    @classmethod
    def load(cls, path, macro_df: pd.DataFrame) -> LiquidityInputs:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        ts = cls._read_time_series(path)
        rsf_weights = cls._read_rsf_factors(path)
        cls._validate_macro(macro_df, ts)

        return LiquidityInputs(
            time_series=ts,
            rsf_weights=rsf_weights,
            macro_history=macro_df,
        )

    # BCBS 295 default RSF weights — granulaires, utilisés quand rsf_factors existe
    _DEFAULT_RSF_WEIGHTS: Dict[str, float] = {
        "cash_central_bank":     0.00,   # BCBS 295 §45(a)
        "sovereign_securities":  0.05,   # BCBS 295 §45(b)
        "short_term_interbank":  0.15,   # BCBS 295 §45(c/d)
        "mortgage_retail_loans": 0.65,   # BCBS 295 §46(b)
        "corporate_loans":       0.85,   # BCBS 295 §46(d)
        "npls_other_assets":     1.00,   # BCBS 295 §46(e)
    }

    # Poids RSF sur colonnes agrégées — utilisés quand loans/securities/other_assets
    # sont présents sans les 6 colonnes granulaires.
    # ATTENTION : le poids "loans" dépend de la composition du portefeuille.
    # Utiliser _RSF_AGG_BY_PORTFOLIO (sélectionné via portfolio_type) plutôt que
    # ce dict hardcodé, qui est conservé pour rétrocompatibilité uniquement.
    _DEFAULT_RSF_AGG_WEIGHTS: Dict[str, float] = {
        "loans":        0.85,   # BCBS 295 §46(d) — corporate > 1an (conservative)
        "securities":   0.50,   # BCBS 295 §45(d) — titre mixte
        "other_assets": 1.00,   # BCBS 295 §46(e) — actifs non liquides
    }

    # Poids RSF agrégés par type de portfolio (BCBS 295 §45-46)
    # Sélectionné via le paramètre portfolio_type du moteur.
    # Le chemin granulaire (6 colonnes RSF distinctes) reste prioritaire.
    _RSF_AGG_BY_PORTFOLIO: Dict[str, Dict[str, float]] = {
        # Banque à dominante retail — prêts hypothécaires, dépôts retail
        # loans weight = 0.65 (BCBS 295 §46b — prêts retail/hypothécaires)
        "retail_heavy": {
            "loans":        0.65,
            "securities":   0.10,   # T-bills court terme, bons du Trésor
            "other_assets": 1.00,
        },
        # Portefeuille mixte 60% retail / 40% corporate (défaut prudent)
        "mixed": {
            "loans":        0.75,   # Moyenne pondérée §46(b) + §46(d)
            # APPROXIMATION : aucun §BCBS 295 ne prescrit 25% pour un mix T-bills +
            # senior unsecured. Les poids granulaires exacts sont §45(b)=5% et
            # §45(c/d)=15%. Cette valeur est une convention interne non sourcée.
            "securities":   0.25,
            "other_assets": 1.00,   # BCBS 295 §46(e)
        },
        # Banque corporate/wholesale — prêts entreprises, financement marché
        # loans weight = 0.85 (BCBS 295 §46d — corporate > 1 an)
        "corporate_heavy": {
            "loans":        0.85,
            "securities":   0.50,   # Senior non-secured (BCBS 295 §45d)
            "other_assets": 1.00,
        },
    }

    # Mapping: aggregate column → list of (rsf_col, fraction)
    # INOPÉRANT depuis Phase 1 fix : _read_time_series() ne synthétise plus de
    # colonnes granulaires à partir des agrégats. Ce mécanisme rendait la Priorité 1
    # de _calc_rsf() toujours vraie (colonnes synthétiques ≡ colonnes réelles),
    # ce qui bloquait la Priorité 2 (_RSF_AGG_BY_PORTFOLIO). Conservé pour référence.
    _RSF_AGGREGATE_MAP = {
        "other_assets": [("cash_central_bank",     1.00)],
        "securities":   [("sovereign_securities",  0.60),
                         ("short_term_interbank",  0.40)],
        "loans":        [("mortgage_retail_loans", 0.50),
                         ("corporate_loans",       0.35),
                         ("npls_other_assets",     0.15)],
    }

    # Accepted sheet names for time_series data
    _TS_SHEET_NAMES = ["time_series", "liquidity_input", "data", "bilan"]

    @classmethod
    def _read_time_series(cls, path: Path) -> pd.DataFrame:
        xl = pd.ExcelFile(path)
        # Try each candidate sheet name
        candidate_sheets = [s for s in cls._TS_SHEET_NAMES if s in xl.sheet_names]
        # Also try any sheet if none of the candidates match
        if not candidate_sheets:
            candidate_sheets = xl.sheet_names

        for sheet in candidate_sheets:
            for header_row in [0, 1, 2]:
                try:
                    df = pd.read_excel(path, sheet_name=sheet,
                                       header=header_row)
                    df.columns = [str(c).strip() for c in df.columns]
                    if "year" not in df.columns:
                        continue
                    df = df.dropna(subset=["year"])
                    df = df[pd.to_numeric(df["year"], errors="coerce").notna()]
                    df["year"] = df["year"].astype(int)
                    df = df.reset_index(drop=True)

                    missing_bilan = [c for c in cls.REQUIRED_BILAN
                                     if c not in df.columns]
                    # Accepte soit les 6 colonnes granulaires BCBS 295 (Priorité 1
                    # dans _calc_rsf), soit les 3 agrégats loans/securities/
                    # other_assets (Priorité 2 via _RSF_AGG_BY_PORTFOLIO).
                    # La synthèse _RSF_AGGREGATE_MAP a été supprimée ici pour que
                    # _calc_rsf() choisisse correctement le chemin selon la donnée
                    # réellement fournie par la banque.
                    _has_granular = all(c in df.columns
                                        for c in cls.RSF_ASSET_COLS)
                    _has_agg = all(c in df.columns
                                   for c in ("loans", "securities", "other_assets"))
                    if not missing_bilan and (_has_granular or _has_agg):
                        # Detect pre-computed LCR / NSFR columns.
                        # The engine recomputes ratios via BCBS formulas, but the
                        # bank's dataset may have its own values computed with
                        # different assumptions (e.g., broader L1 definition,
                        # no L2a cap). If found, these are stored as _lcr_obs /
                        # _nsfr_obs and used to anchor the historical series and
                        # projection baseline so charts match the source data.
                        for _target, _kw in [("_lcr_obs", "lcr"),
                                             ("_nsfr_obs", "nsfr")]:
                            if _target in df.columns:
                                continue
                            _skip = set(cls.REQUIRED_BILAN) | set(cls.RSF_ASSET_COLS)
                            for _col in df.columns:
                                if _col in _skip:
                                    continue
                                if _kw not in _col.lower():
                                    continue
                                _vals = pd.to_numeric(df[_col], errors="coerce")
                                # Accept only columns whose values look like a
                                # percentage ratio (> 20 %) rather than a rate.
                                if _vals.notna().sum() >= 1 and _vals.max() > 20:
                                    df[_target] = _vals
                                    LOG.info(
                                        "Pre-computed ratio column '%s' → '%s' "
                                        "(max=%.1f%%) — will anchor series.",
                                        _col, _target, float(_vals.max()),
                                    )
                                    break
                        return df
                except Exception:
                    continue

        raise ValueError(
            f"Impossible de lire les données depuis {path}. "
            f"Colonnes bilan requises : {cls.REQUIRED_BILAN}. "
            f"Colonnes RSF requises : soit les 6 colonnes granulaires BCBS 295 "
            f"{cls.RSF_ASSET_COLS}, soit les 3 agrégats "
            f"['loans', 'securities', 'other_assets']."
        )

    @classmethod
    def _read_rsf_factors(cls, path: Path) -> Dict[str, float]:
        # RSF factors are uniform across all banks (BCBS 295 §45-46) — always hardcoded.
        # Per-bank rsf_factors sheets are ignored: they introduce inconsistency with
        # the regulatory standard and were the root cause of NSFR computation gaps.
        return dict(cls._DEFAULT_RSF_WEIGHTS)

    @classmethod
    def _validate_macro(cls, macro_df: pd.DataFrame, ts: pd.DataFrame):
        missing = [c for c in MACRO_VARS if c not in macro_df.columns]
        if missing:
            raise ValueError(f"Colonnes macro manquantes : {missing}")


# ─────────────────────────────────────────────────────────────────────────────
# MOTEUR LCR / NSFR
# ─────────────────────────────────────────────────────────────────────────────
class LiquidityStressEngine:

    def __init__(
        self,
        inputs: LiquidityInputs,
        portfolio_type: str = "mixed",
        bs_stress_delta_liab: Optional[Dict[str, float]] = None,
        bs_stress_delta_asset: Optional[Dict[str, float]] = None,
        growth_decay_half_life: float = 5.0,
    ):
        """
        Parameters
        ----------
        inputs             : LiquidityInputs (bilan + macro)
        portfolio_type     : "retail_heavy" | "mixed" | "corporate_heavy"
                             Sélectionne le poids RSF correct pour la colonne
                             agrégée "loans" (BCBS 295 §46b/d).
                             Ignoré si les 6 colonnes granulaires sont présentes.
        bs_stress_delta_liab  : dict {scenario_id: delta_pp} — variation de taux
                             de croissance des passifs sous stress.
                             Défaut : baseline=0, adverse=-1%, severe=-3%.
        bs_stress_delta_asset : dict {scenario_id: delta_pp} — variation de taux
                             de croissance des actifs sous stress.
                             Défaut : baseline=0, adverse=+1%, severe=+3%.
        """
        self.inputs = inputs
        self.portfolio_type = portfolio_type
        self._calibrator = SatelliteCalibrator()
        self._sat_results: Optional[Dict[str, SatelliteResult]] = None
        # CAGR historiques par colonne bilan (calculé dans calibrate())
        self._bs_cagr: Dict[str, float] = {}
        # Stress deltas sur les taux de croissance bilan par niveau de sévérité
        self._bs_stress_liab: Dict[str, float] = bs_stress_delta_liab or {
            "baseline": 0.00,
            "adverse":  -0.01,   # −1 pp : contraction modérée des dépôts
            "severe":   -0.03,   # −3 pp : fuite des dépôts en crise
        }
        self._bs_stress_asset: Dict[str, float] = bs_stress_delta_asset or {
            "baseline": 0.00,
            "adverse":  +0.01,   # +1 pp : tirages sur lignes de crédit (adverse)
            "severe":   +0.03,   # +3 pp : tirages massifs + accumulation NPL
        }
        # Demi-vie (années) du choc de stress dans _project_bilan : au-delà
        # d'un horizon de quelques années, un taux de croissance constant
        # composé (1+g)^t diverge de façon irréaliste (ex. horizon NGFS de
        # 25 ans) — seul le choc (stress_liab/stress_asset) décroît vers 0 ;
        # le CAGR historique de chaque poste de bilan reste inchangé.
        self._growth_half_life = float(growth_decay_half_life)

    # ── CALIBRATION ──────────────────────────────────────────────────────────
    def calibrate(
        self,
        forced_sat_ranks: Optional[Dict[str, int]] = None,
    ) -> "LiquidityStressEngine":
        """Calibre les 5 satellites via SatelliteCalibrator.calibrate_all()."""
        ts = self.inputs.time_series
        mac = self.inputs.macro_history

        # Aligner sur les années communes (les deux indexés par year)
        if "year" in mac.columns:
            mac = mac.set_index("year")
        ts_idx = ts.set_index("year")

        # Diagnostic — un DataFrame avec des années dupliquées fait que
        # .loc[common_years] renvoie plus de lignes que d'années communes,
        # désalignant silencieusement bilan/macro (voir incident shapes
        # (15,) vs (30,)). Log clair pour confirmer lequel des deux avant
        # de corriger, plutôt que de deviner.
        _ts_dup  = int(ts_idx.index.duplicated().sum())
        _mac_dup = int(mac.index.duplicated().sum())
        LOG.info(
            "LiquidityStressEngine.calibrate: bilan=%d lignes (%d année(s) "
            "dupliquée(s)) | macro=%d lignes (%d année(s) dupliquée(s))",
            len(ts_idx), _ts_dup, len(mac), _mac_dup,
        )
        if _ts_dup:
            _dup_years_ts = sorted(set(ts_idx.index[ts_idx.index.duplicated()]))
            LOG.warning(
                "LiquidityStressEngine.calibrate: années dupliquées dans le "
                "BILAN : %s", _dup_years_ts,
            )
        if _mac_dup:
            _dup_years_mac = sorted(set(mac.index[mac.index.duplicated()]))
            LOG.warning(
                "LiquidityStressEngine.calibrate: années dupliquées dans la "
                "MACRO : %s", _dup_years_mac,
            )

        common_years = sorted(set(ts_idx.index) & set(mac.index))
        ts_aligned = ts_idx.loc[common_years]
        mac_aligned = mac.loc[common_years]

        self._sat_results = self._calibrator.calibrate_all(
            ts_aligned, mac_aligned, forced_ranks=forced_sat_ranks or {},
        )

        # ── Calcul des CAGR historiques par item bilan ───────────────────────
        # Utilisés pour projeter le bilan dynamiquement dans compute_stress().
        # Même logique que le module crédit pour l'EAD.
        self._bs_cagr = self._compute_bs_cagr(ts_idx.sort_index())
        LOG.info(
            "Balance sheet CAGRs computed for %d items (portfolio_type='%s').",
            len(self._bs_cagr), self.portfolio_type,
        )
        return self

    # ── BALANCE SHEET CAGR ──────────────────────────────────────────────────
    @staticmethod
    def _compute_bs_cagr(ts: pd.DataFrame) -> Dict[str, float]:
        """
        Calcule le CAGR historique de chaque colonne numérique du bilan.

        Formule : g = (val_last / val_first)^(1/(n-1)) - 1
        Si n < 2 ou val_first ≈ 0 → g = 0 (pas de tendance mesurable).
        """
        cagr: Dict[str, float] = {}
        for col in ts.columns:
            try:
                series = pd.to_numeric(ts[col], errors="coerce").dropna()
                n = len(series)
                if n < 2:
                    cagr[col] = 0.0
                    continue
                first = float(series.iloc[0])
                last  = float(series.iloc[-1])
                if abs(first) < 1e-6:
                    cagr[col] = 0.0
                else:
                    g = float((last / first) ** (1.0 / (n - 1)) - 1.0)
                    # Plafonner à ±30% / an pour éviter les extrapolations absurdes
                    cagr[col] = float(np.clip(g, -0.30, 0.30))
            except Exception:
                cagr[col] = 0.0
        return cagr

    def _project_bilan(
        self,
        bilan_anchor: pd.Series,
        t: int,
        stress_liab: float,
        stress_asset: float,
    ) -> pd.Series:
        """
        Projette le bilan à l'année t (offset depuis l'ancrage).

        Pour chaque item, pour k=1..t :
          g_eff(k) = g_hist + g_choc × 0.5^(k/half_life)
          bilan_t[col] = bilan_anchor[col] × Π_{k=1}^{t} (1 + g_eff(k))

        g_hist (CAGR historique de l'item, tendance structurelle) reste
        constant ; g_choc (stress_liab/stress_asset, choc lié au scénario)
        décroît vers 0 avec une demi-vie de self._growth_half_life années.
        Un taux constant composé (1+g)^t diverge de façon irréaliste sur un
        horizon NGFS long (25 ans+) — le choc est traité comme transitoire,
        la tendance structurelle continue seule au-delà de quelques années.
        stress_liab=stress_asset=0 (scénario baseline) reproduit exactement
        l'ancien (1+g_hist)^t — aucun changement pour la baseline.

        Passifs (dépôts, wholesale) : stress_liab est négatif sous stress
                                      → la base de dépôts se contracte.
        Actifs (prêts, titres)       : stress_asset est positif sous stress
                                      → les tirages et NPL augmentent.
        Items inconnus               : g_hist seul (croissance naturelle).

        Cohérent avec le module crédit (EAD CAGR + stress EAD growth,
        voir capital_engine.py::_decayed_growth_curve).
        """
        if not self._bs_cagr:
            return bilan_anchor  # calibrate() non appelé — pas de projection
        bilan_t = bilan_anchor.copy().astype(float)
        half_life = self._growth_half_life
        decay_path = [0.5 ** (k / half_life) for k in range(1, t + 1)]
        for col in bilan_anchor.index:
            g_hist    = self._bs_cagr.get(col, 0.0)
            val_0     = float(bilan_anchor[col])
            if abs(val_0) < 1e-9:
                continue
            if col in _LIABILITY_COLS:
                g_choc = stress_liab
            elif col in _ASSET_COLS:
                g_choc = stress_asset
            else:
                g_choc = 0.0
            factor = 1.0
            for dk in decay_path:
                factor *= (1.0 + g_hist + g_choc * dk)
            bilan_t[col] = val_0 * factor
        return bilan_t

    # ── PREDICT SATELLITES ───────────────────────────────────────────────────
    def _predict_satellites(self, macro_row_df: pd.DataFrame) -> pd.Series:
        """
        Prédit les 5 variables satellites pour un vecteur macro (1 ligne).
        Retourne un Series avec les 5 colonnes satellite.
        """
        preds = {}
        for var_name, sat_result in self._sat_results.items():
            values = sat_result.predict(macro_row_df)
            preds[var_name] = float(values[0]) if len(values) > 0 else 0.0
        return pd.Series(preds)

    def _predict_satellites_all(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit les 5 satellites pour TOUTES les lignes de macro_df d'un coup.

        Pourquoi ce n'est pas _predict_satellites() appelé en boucle :
        Les satellites peuvent utiliser des features de type __lag1 ou __growth
        calculées par _make_feature_df().  Sur UNE seule ligne, ces features
        produisent NaN (pas de ligne précédente) → la closure les remplace par la
        moyenne d'entraînement (constante) → raw_delta = 0 entre baseline et sévère.
        En passant TOUTES les années d'un coup (+ contexte historique), les lags et
        taux de croissance sont calculés correctement à partir de t−1.

        Retourne un DataFrame(index=macro_df.index, colonnes=SATELLITE_VARS).
        """
        n = len(macro_df)
        preds: dict = {}
        for var_name, sat_result in self._sat_results.items():
            if var_name not in SATELLITE_VARS:
                continue
            values = sat_result.predict(macro_df)
            preds[var_name] = [float(v) for v in values] if len(values) == n else [0.0] * n
        return pd.DataFrame(
            {v: preds.get(v, [0.0] * n) for v in SATELLITE_VARS},
            index=macro_df.index,
        )

    # ── COMPUTE STRESS ───────────────────────────────────────────────────────
    def compute_stress(
        self,
        scenarios: Dict[str, pd.DataFrame],
        enforce_sign: bool = True,
    ) -> Dict[str, RiskOutput]:
        """
        Calcule LCR/NSFR pour chaque scénario.

        Parameters
        ----------
        scenarios : dict {scenario_id: DataFrame_macro_projetée}
                    Chaque DataFrame est indexé par year et contient MACRO_VARS.
        enforce_sign : bool, default True
                    Si True (mode stress test traditionnel), les deltas satellites
                    dont le signe est contraire à l'attente économique sont mis à
                    zéro (run-off ne peut que monter, ASF ne peut que baisser).
                    Mettre à False pour les scénarios NGFS climatiques où un
                    scénario "adverse" (ex. Delayed Transition) peut afficher une
                    macro court-terme meilleure que le baseline Net Zero, rendant
                    les deltas négatifs pour les run-off économiquement corrects.
        """
        self._check_calibrated()

        ts = self.inputs.time_series.set_index("year")
        rsf = self.inputs.rsf_weights
        last_year = ts.index.max()
        bilan_anchor = ts.loc[last_year]

        # Transparency flag: track which RSF computation path is actually used.
        # "granular_real"        — all 6 BCBS-295 columns present in the bank's data.
        # "portfolio_default_mixed" — only aggregate columns; _RSF_AGG_BY_PORTFOLIO
        #                            weights apply (other_assets=100%, securities=25%,
        #                            loans=75%). Propagated to RiskOutput.metadata and
        #                            PlatformResult.metadata for export traceability.
        _rsf_is_granular = all(c in ts.columns for c in rsf.keys())
        _rsf_path = "granular_real" if _rsf_is_granular else "portfolio_default_mixed"

        # ── Ancrage sur les valeurs réelles observées (dernière année bilan) ──
        # Évite le biais de prédiction du modèle satellite en mode absolu.
        # Les projections utilisent : sats_projetées = sats_réelles + Δ_prédit
        # où Δ = predict(macro_stressée) - predict(macro_baseline)
        # Pour le scénario baseline : Δ = 0 → continuité parfaite avec l'historique.
        sats_anchor = bilan_anchor[SATELLITE_VARS].astype(float)

        # Contraintes de signe économique sur les deltas (BCBS / littérature) :
        #   run-off rates et haircut : ne peuvent que monter sous stress (Δ ≥ 0)
        #   facteurs ASF            : ne peuvent que baisser sous stress  (Δ ≤ 0)
        _DELTA_SIGN = {
            "run_off_retail":      +1,
            "run_off_corporate":   +1,
            "haircut_add":         +1,
            "asf_factor_retail":   -1,
            "asf_factor_corporate":-1,
        }

        mac = self.inputs.macro_history
        if "year" in mac.columns:
            mac = mac.set_index("year")
        mac_baseline_row = mac.loc[[mac.index.max()]]
        sats_model_base = self._predict_satellites(mac_baseline_row)

        # ── Poids RSF agrégés selon portfolio_type (Bug 6 fix) ───────────────
        # Priorité : table par portfolio_type > défaut conservateur 0.85
        # Le chemin granulaire (6 colonnes RSF) reste prioritaire dans _calc_rsf.
        agg_weights = LiquidityDataLoader._RSF_AGG_BY_PORTFOLIO.get(
            self.portfolio_type,
            LiquidityDataLoader._DEFAULT_RSF_AGG_WEIGHTS,
        )
        LOG.info(
            "RSF aggregate weights (portfolio_type='%s'): %s",
            self.portfolio_type, agg_weights,
        )

        # Satellites + bilan baseline = valeurs réelles (Δ = 0, bilan ancré)
        lcr_base, nsfr_base, hqla_b, nco_b = self._calc_ratios(
            sats_anchor, bilan_anchor, rsf, agg_weights
        )

        # ── Anchoring on observed LCR/NSFR ──────────────────────────────────
        # If the dataset contains pre-computed ratio columns (_lcr_obs /
        # _nsfr_obs), the engine's formula may differ from the bank's own
        # calculation (e.g., the bank applies a broader L1 definition or does
        # not cap L2a as strictly as BCBS 238 §43).
        # To keep the chart anchored on the observed values while still using
        # the formula to derive the STRESS DELTA, all projected ratios are
        # multiplied by:  scale = observed_last / formula_base
        # This way the baseline projection starts at the correct observed level
        # and the stress moves it by the same relative amount as the formula.
        _lcr_obs  = None
        _nsfr_obs = None
        if "_lcr_obs" in bilan_anchor.index:
            _v = pd.to_numeric(bilan_anchor["_lcr_obs"], errors="coerce")
            if pd.notna(_v) and float(_v) > 0:
                _lcr_obs = float(_v)
        if "_nsfr_obs" in bilan_anchor.index:
            _v = pd.to_numeric(bilan_anchor["_nsfr_obs"], errors="coerce")
            if pd.notna(_v) and float(_v) > 0:
                _nsfr_obs = float(_v)

        lcr_scale  = (_lcr_obs  / max(lcr_base,  1e-6)) if _lcr_obs  else 1.0
        nsfr_scale = (_nsfr_obs / max(nsfr_base, 1e-6)) if _nsfr_obs else 1.0

        if _lcr_obs:
            LOG.info(
                "LCR anchor: formula_base=%.2f%% obs=%.2f%% → scale=%.4f",
                lcr_base, _lcr_obs, lcr_scale,
            )
        if _nsfr_obs:
            LOG.info(
                "NSFR anchor: formula_base=%.2f%% obs=%.2f%% → scale=%.4f",
                nsfr_base, _nsfr_obs, nsfr_scale,
            )

        # Effective baseline (what the chart will show as "baseline")
        lcr_base_eff  = lcr_base  * lcr_scale
        nsfr_base_eff = nsfr_base * nsfr_scale

        # ── Projection baseline des satellites — calculée UNE SEULE FOIS ────
        # Valide pour tous les paths (A, B/C, D, E) : _predict_satellites()
        # et base_sats sont indépendants du path macro.
        _all_years: List[int] = sorted(set().union(*[
            (set(int(y) for y in sd["year"]) if "year" in sd.columns
             else set(int(y) for y in sd.index))
            for sd in scenarios.values()
        ]))
        _sat_baseline_proj: Dict[int, pd.Series] = {}
        if _PROJ_OK and _all_years:
            try:
                _bp = project_baseline(
                    hist_df=ts.reset_index(),
                    selected_hist_vars=SATELLITE_VARS,
                    horizon_years=_all_years,
                    allow_var=False,
                    variable_config=SATELLITE_VARIABLE_CONFIG,
                )
                for _yr in _all_years:
                    _yr_vals = {}
                    for _sv in SATELLITE_VARS:
                        if _sv in _bp.variable_projections:
                            _yr_vals[_sv] = (
                                _bp.variable_projections[_sv]
                                .projected_levels
                                .get(_yr, float(sats_anchor[_sv]))
                            )
                        else:
                            _yr_vals[_sv] = float(sats_anchor[_sv])
                    _sat_baseline_proj[_yr] = pd.Series(_yr_vals)
                LOG.info(
                    "Projection baseline satellites :\n%s",
                    summarize_fallbacks(_bp),
                )
            except Exception as _e:
                LOG.warning(
                    "project_baseline() satellites a échoué (%s) "
                    "— fallback sur sats_anchor.", _e
                )


        # ── Pré-calcul multi-lignes des prédictions satellite ────────────────
        # POURQUOI multi-lignes (et non per-année) :
        # Les satellites peuvent utiliser des features __lag1 / __growth calculées
        # par _make_feature_df().  Sur UNE seule ligne (scen_df.loc[[yr]]), ces
        # transforms produisent NaN → remplacés par la moyenne d'entraînement
        # (constante) → raw_delta = sats(sévère[yr]) - sats(baseline[yr]) = 0.
        # En passant TOUTES les années + 2 lignes de contexte historique, les
        # features de croissance/lag sont calculées correctement (row t utilise t-1).
        #
        # raw_delta = sats(scénario_stressé[yr]) - sats(scénario_baseline[yr])
        # Ce différentiel capture UNIQUEMENT le choc de crise (sans double-comptage
        # de la dérive macro 2024→2028, qui s'annule dans la soustraction).

        # Contexte historique pour les transforms lag/growth (2 dernières années)
        _mac_full = self.inputs.macro_history
        if "year" in _mac_full.columns:
            _mac_full = _mac_full.set_index("year")
        _hist_ctx = _mac_full.tail(2)

        # Normaliser le baseline scenario DataFrame
        _bl_raw = scenarios.get("baseline")
        if _bl_raw is not None:
            if "year" in _bl_raw.columns:
                _bl_raw = _bl_raw.set_index("year")
            elif _bl_raw.index.name != "year":
                _bl_raw.index.name = "year"

        # Prédictions baseline (yr → Series) via contexte historique
        _bl_sats_all: Optional[pd.DataFrame] = None
        if _bl_raw is not None:
            try:
                _bl_ctx_df = pd.concat([_hist_ctx, _bl_raw])
                _bl_sats_all = self._predict_satellites_all(_bl_ctx_df)
                # Garder seulement les années scénario (pas le contexte historique)
                _bl_sats_all = _bl_sats_all.loc[
                    _bl_sats_all.index.isin(_bl_raw.index)
                ]
            except Exception as _bl_err:
                LOG.warning("Pré-calcul satellites baseline échoué (%s) — fallback per-year", _bl_err)

        results = {}
        for scen_id, scen_df in scenarios.items():
            if "year" in scen_df.columns:
                scen_df = scen_df.set_index("year")
            elif scen_df.index.name != "year":
                scen_df.index.name = "year"

            years = sorted(scen_df.index.astype(int).tolist())

            # Pré-calcul multi-lignes pour ce scénario (lag/growth corrects)
            _scen_sats_all: Optional[pd.DataFrame] = None
            try:
                _scen_ctx_df = pd.concat([_hist_ctx, scen_df])
                _scen_sats_all = self._predict_satellites_all(_scen_ctx_df)
                _scen_sats_all = _scen_sats_all.loc[
                    _scen_sats_all.index.isin(scen_df.index)
                ]
            except Exception as _sc_err:
                LOG.warning("Pré-calcul satellites scénario '%s' échoué (%s) — fallback per-year",
                            scen_id, _sc_err)

            # Climate scenarios (Path E / NGFS) do not trigger a direct
            # balance-sheet funding shock — climate risk transmits via the
            # behavioral satellite channel only (run_off/ASF/haircut under
            # NiGEM-stressed macro). Deliberate scope decision, not an omission.
            # See thesis Chapter 2.
            # Stress deltas sur les taux de croissance bilan (Bug 5 fix)
            stress_liab  = self._bs_stress_liab.get(scen_id,  0.0)
            stress_asset = self._bs_stress_asset.get(scen_id, 0.0)

            records = []
            for i, yr in enumerate(years):
                t = i + 1  # offset depuis l'ancrage (T+1, T+2, …)

                # ── Projection dynamique du bilan (Bug 5 fix) ────────────────
                bilan_t = self._project_bilan(bilan_anchor, t, stress_liab, stress_asset)

                # Prédiction satellite stressée — multi-lignes si disponible
                if _scen_sats_all is not None and yr in _scen_sats_all.index:
                    sats_model_stressed = _scen_sats_all.loc[yr]
                else:
                    sats_model_stressed = self._predict_satellites(scen_df.loc[[yr]])

                # Référence baseline au même yr (différentiel = choc pur)
                if _bl_sats_all is not None and yr in _bl_sats_all.index:
                    _sats_ref_yr = _bl_sats_all.loc[yr]
                else:
                    _sats_ref_yr = sats_model_base   # fallback last_obs

                raw_delta = sats_model_stressed - _sats_ref_yr

                # Appliquer les contraintes de signe : un choc ne peut pas améliorer
                # un satellite dans le mauvais sens (ex. ASF qui monte sous stress)
                constrained_delta_vals = {}
                for v in SATELLITE_VARS:
                    _v = raw_delta[v]
                    # .loc[yr] may return a DataFrame row (Series with extra dim)
                    # when the index has duplicate year entries — squeeze to scalar
                    if hasattr(_v, "iloc"):
                        _v = _v.iloc[0]
                    raw = float(_v)
                    if not enforce_sign or raw * _DELTA_SIGN[v] >= 0:
                        constrained_delta_vals[v] = raw
                    else:
                        LOG.warning(
                            "Scenario '%s', year %d: satellite delta for '%s' has "
                            "wrong sign (raw_delta=%.6f, expected_sign=%+d) — "
                            "delta zeroed out. CONSEQUENCE: this shock has no effect "
                            "on '%s' for this year.",
                            scen_id, yr, v, raw, _DELTA_SIGN[v], v,
                        )
                        constrained_delta_vals[v] = 0.0
                constrained_delta = pd.Series(constrained_delta_vals)

                # ── Plafond dur sur la variation annuelle (DELTA_CAPS) ────────
                # La régression satellite peut produire, dès la 1ère année
                # projetée (T+1), un delta de plusieurs dizaines de points de %
                # — ex. run_off_retail à -0.75 en 2026, alors que le pire
                # mouvement annuel JAMAIS observé sur l'historique de cette
                # banque est de 0.055 (5.5 pp, année 2020). Un tel saut fait
                # s'effondrer run_off_retail/corporate à 0 simultanément →
                # NCO touche son plancher (1e-6) → LCR explose (HQLA/NCO×100).
                # DELTA_CAPS (behavioural_variable_config.py) plafonne CHAQUE
                # variable à ±(max historique × 2) — "extreme but plausible" :
                # un choc deux fois pire que tout ce qui a jamais été observé
                # reste autorisé, mais pas une divergence sans aucune ancre
                # empirique. Appliqué AVANT la décroissance long-horizon
                # ci-dessous (qui agit donc sur le delta déjà borné).
                for v in SATELLITE_VARS:
                    cap = DELTA_CAPS.get(v)
                    if cap is None:
                        continue
                    delta_brut = float(constrained_delta[v])
                    if abs(delta_brut) > cap:
                        delta_clipe = float(np.clip(delta_brut, -cap, cap))
                        hist_max = cap / 2.0  # cap = max historique × 2
                        magnitude_x = abs(delta_brut) / hist_max if hist_max > 0 else float("inf")
                        LOG.warning(
                            "Scenario '%s', year %d: satellite delta for '%s' "
                            "dépasse le plafond DELTA_CAPS — delta_brut=%.6f "
                            "delta_clipé=%.6f (cap=±%.4f) — %.1fx le max "
                            "historique observé (%.4f).",
                            scen_id, yr, v, delta_brut, delta_clipe, cap,
                            magnitude_x, hist_max,
                        )
                        constrained_delta[v] = delta_clipe

                # ── Décroissance du choc satellite (long horizon) ─────────────
                # raw_delta vient d'une régression linéaire calibrée sur
                # l'historique ; extrapolée sur un horizon NGFS long (25 ans+),
                # la trajectoire macro s'éloigne du domaine d'entraînement et le
                # delta prédit diverge (ex. asf_factor_retail hors domaine tous
                # les ans de 2023 à 2050 dans certains scénarios). Le clipping
                # de domaine borne déjà la VALEUR finale, mais un delta qui
                # reste à sa borne durant 25 ans reflète une extrapolation non
                # significative, pas une dynamique de scénario réelle.
                # Même logique que _project_bilan/_decayed_growth_curve : le
                # choc comportemental (run-off, ASF, haircut) est transitoire
                # et doit reconverger vers la projection baseline (AR/ARIMA,
                # non affectée par le choc) au lieu de diverger indéfiniment.
                # Appliquée SUR le delta déjà borné par DELTA_CAPS ci-dessus,
                # jamais sur le delta brut de la régression.
                _delta_decay = 0.5 ** (t / self._growth_half_life)
                constrained_delta = constrained_delta * _delta_decay

                # Satellites projetés = projection baseline + delta contraint
                # _base_sats : valeur projetée par AR/ARIMA depuis l'historique
                # (indépendante du scénario macro). Fallback sur sats_anchor si
                # project_baseline() a échoué ou si l'année est hors horizon.
                _base_sats = _sat_baseline_proj.get(yr, sats_anchor)
                sats_proj = _base_sats + constrained_delta

                # Clipping de domaine : signe seul ne suffit pas, magnitude non bornée
                for _v in SATELLITE_VARS:
                    _bounds = SATELLITE_VARIABLE_CONFIG.get(_v, {}).get("bounds")
                    if _bounds is None:
                        continue
                    _before = float(sats_proj[_v])
                    _clipped = float(np.clip(_before, _bounds[0], _bounds[1]))
                    if abs(_clipped - _before) > 1e-9:
                        LOG.warning(
                            "Scenario '%s', year %d: satellite '%s' hors domaine "
                            "après delta (%.6f) — clippé à %.6f (bounds=%s).",
                            scen_id, yr, _v, _before, _clipped, _bounds,
                        )
                        sats_proj[_v] = _clipped

                # Ratios calculés sur le bilan PROJETÉ (pas l'ancre figée)
                lcr, nsfr, hqla, nco = self._calc_ratios(
                    sats_proj, bilan_t, rsf, agg_weights
                )

                rec = {
                    "year": yr,
                    "lcr":  round(lcr  * lcr_scale,  2),
                    "nsfr": round(nsfr * nsfr_scale, 2),
                    "lcr_baseline":  round(lcr_base_eff,  2),
                    "nsfr_baseline": round(nsfr_base_eff, 2),
                    "hqla": round(hqla, 3),
                    "nco":  round(nco, 3),
                    "asf":  round(self._calc_asf(sats_proj, bilan_t), 3),
                    "rsf":  round(self._calc_rsf(bilan_t, rsf, agg_weights), 3),
                }
                for sv in SATELLITE_VARS:
                    rec[sv] = round(float(sats_proj.get(sv, 0.0)), 6)
                records.append(rec)

            ts_proj = pd.DataFrame(records)

            # ── Métriques agrégées ────────────────────────────────────────────
            lcr_min = ts_proj["lcr"].min()
            nsfr_min = ts_proj["nsfr"].min()

            breach_rows = ts_proj[
                (ts_proj["lcr"] < BREACH_THRESHOLD) |
                (ts_proj["nsfr"] < BREACH_THRESHOLD)
            ]
            breach_year = int(breach_rows["year"].min()) if not breach_rows.empty else None

            nco_series = ts_proj["nco"].dropna()
            if len(nco_series) > 0:
                loss = float(max(nco_series.max() - nco_series.iloc[0], 0.0))
                hqla_delta = round(float(
                    ts_proj.loc[nco_series.idxmax(), "hqla"] - hqla_b
                ), 3)
                nco_delta = round(float(nco_series.max() - nco_b), 3)
            else:
                loss = 0.0
                hqla_delta = 0.0
                nco_delta = 0.0

            results[scen_id] = RiskOutput(
                risk_type="liquidity",
                scenario_id=scen_id,
                loss=round(loss, 4),
                metrics={
                    "lcr_baseline":  round(lcr_base_eff,  2),
                    "lcr_stressed":  round(lcr_min,       2),
                    "nsfr_baseline": round(nsfr_base_eff, 2),
                    "nsfr_stressed": round(nsfr_min,      2),
                    "breach_year":   breach_year,
                    "hqla_delta":    hqla_delta,
                    "nco_delta":     nco_delta,
                },
                time_series=ts_proj,
                metadata={
                    "scenario":       scen_id,
                    "years":          years,
                    "rsf_weights":    rsf,
                    "last_hist_year": int(last_year),
                    "rsf_path":       _rsf_path,
                },
            )

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # CALCULS LCR / NSFR
    # ─────────────────────────────────────────────────────────────────────────
    def _calc_ratios(self, sats, bilan, rsf, agg_weights=None):
        hqla = self._calc_hqla(sats, bilan)
        nco  = self._calc_nco(sats, bilan)
        asf  = self._calc_asf(sats, bilan)
        rsf_ = self._calc_rsf(bilan, rsf, agg_weights)
        lcr  = (hqla / nco * 100) if nco > 0 else np.inf
        nsfr = (asf / rsf_ * 100) if rsf_ > 0 else np.inf
        return lcr, nsfr, hqla, nco

    @staticmethod
    def _calc_hqla(sats, bilan):
        l1 = float(bilan["L1"])
        l2a = float(bilan["L2a"])
        haircut_add = float(sats["haircut_add"])
        l2a_net = l2a * (1 - L2A_HAIRCUT_REG - haircut_add)
        l2a_final = min(max(l2a_net, 0.0), L2A_CAP_RATIO * l1)
        return l1 + l2a_final

    @staticmethod
    def _calc_nco(sats, bilan):
        outflows = (
            float(sats["run_off_retail"])    * float(bilan["retail_deposits"])
          + float(sats["run_off_corporate"]) * float(bilan["corporate_deposits"])
          + WHOLESALE_OUTFLOW                * float(bilan["wholesale_funding"])
          + OFFBS_OUTFLOW                    * float(bilan["off_bs_commitments"])
        )
        inflows = min(
            float(bilan["inflows_contractual"]),
            INFLOW_CAP * outflows
        )
        return max(outflows - inflows, 1e-6)

    @staticmethod
    def _calc_asf(sats, bilan):
        return (
            float(sats["asf_factor_retail"])    * float(bilan["retail_deposits"])
          + float(sats["asf_factor_corporate"]) * float(bilan["corporate_deposits"])
          + WHOLESALE_ASF                       * float(bilan["wholesale_funding"])
          + float(bilan["asf_other"])
        )

    @staticmethod
    def _calc_rsf(bilan, rsf_weights, agg_weights=None):
        """
        Calcule le RSF (Required Stable Funding) — BCBS 295.

        Priorité 1 : colonnes granulaires BCBS 295
            (cash_central_bank, sovereign_securities, short_term_interbank,
             mortgage_retail_loans, corporate_loans, npls_other_assets)
            → poids exacts depuis rsf_factors ou _DEFAULT_RSF_WEIGHTS.

        Priorité 2 : colonnes agrégées (loans / securities / other_assets)
            → poids sélectionnés par portfolio_type via agg_weights.
            Bug 6 fix : 0.65 pour retail_heavy (§46b), 0.75 mixed, 0.85 corporate.

        Parameters
        ----------
        agg_weights : dict ou None
            Poids agrégés issus de _RSF_AGG_BY_PORTFOLIO[portfolio_type].
            Si None, fallback sur _DEFAULT_RSF_AGG_WEIGHTS (0.85 — conservateur).
        """
        # Priorité 1 : colonnes granulaires BCBS 295 (depuis rsf_factors ou defaults)
        # Si toutes les colonnes granulaires sont présentes, utiliser leurs poids exacts.
        granular_cols = list(rsf_weights.keys())
        if all(c in bilan.index for c in granular_cols):
            rsf_total = sum(
                rsf_weights[cat] * float(bilan[cat]) for cat in granular_cols
            )
            return max(rsf_total, 1e-6)

        # Priorité 2 : colonnes agrégées avec poids calibrés par portfolio_type
        # (Bug 6 fix : utilise agg_weights passé par compute_stress au lieu du
        # hardcodé 0.85 qui survalorisait le RSF des banques retail-heavy)
        AGG = agg_weights or LiquidityDataLoader._DEFAULT_RSF_AGG_WEIGHTS
        agg_present = [c for c in AGG if c in bilan.index]
        if agg_present:
            return max(
                sum(AGG[c] * float(bilan[c]) for c in agg_present),
                1e-6
            )

        # Fallback : colonnes granulaires partiellement présentes
        rsf_total = 0.0
        for cat, weight in rsf_weights.items():
            if cat in bilan.index:
                rsf_total += weight * float(bilan[cat])
            else:
                raise KeyError(
                    f"Colonne RSF '{cat}' absente du time_series."
                )
        return max(rsf_total, 1e-6)

    def _check_calibrated(self):
        if self._sat_results is None:
            raise RuntimeError(
                "Satellites non calibrés. Appeler engine.calibrate() d'abord."
            )

    def compute_historical_ratios(self) -> Dict[str, list]:
        """
        Calcule LCR/NSFR historiques en appliquant les formules BCBS sur les données
        réelles de chaque année (satellites observés, pas prédits).

        LCR(t) = HQLA(bilan(t), sats_réels(t)) / NCO(bilan(t), sats_réels(t)) × 100
        NSFR(t) = ASF(bilan(t), sats_réels(t)) / RSF(bilan(t)) × 100

        Les satellites réels (run_off_retail, run_off_corporate, haircut_add,
        asf_factor_retail, asf_factor_corporate) sont lus directement depuis le
        time_series — aucune prédiction macro utilisée ici.
        Le RSF utilise les colonnes agrégées (loans/securities/other_assets) si
        présentes, conformément à la formule Excel fournie par la banque.
        """
        self._check_calibrated()

        ts = self.inputs.time_series.set_index("year")
        mac = self.inputs.macro_history
        if "year" in mac.columns:
            mac = mac.set_index("year")

        rsf_w = self.inputs.rsf_weights
        common_years = sorted(set(ts.index) & set(mac.index))

        # Use the same portfolio-type RSF weights as compute_stress() so that
        # historical NSFR is computed consistently with the projected NSFR baseline.
        agg_weights = LiquidityDataLoader._RSF_AGG_BY_PORTFOLIO.get(
            self.portfolio_type,
            LiquidityDataLoader._DEFAULT_RSF_AGG_WEIGHTS,
        )

        _has_lcr_obs  = "_lcr_obs"  in ts.columns
        _has_nsfr_obs = "_nsfr_obs" in ts.columns

        years_out, lcr_out, nsfr_out = [], [], []
        for yr in common_years:
            bilan_row = ts.loc[yr]

            # Priority: use pre-computed ratio columns from the dataset when
            # available (they reflect the bank's own methodology which may
            # differ from the BCBS formula, e.g. different L1/L2a scope).
            # Fallback: recompute from balance sheet items via BCBS formulas.
            if _has_lcr_obs:
                _v = pd.to_numeric(bilan_row["_lcr_obs"], errors="coerce")
                lcr  = float(_v)  if pd.notna(_v) and float(_v) > 0 else None
            else:
                sats_actual = bilan_row[SATELLITE_VARS]
                lcr_calc, _, _, _ = self._calc_ratios(
                    sats_actual, bilan_row, rsf_w, agg_weights
                )
                lcr = round(float(lcr_calc), 2) if np.isfinite(lcr_calc) else None

            if _has_nsfr_obs:
                _v = pd.to_numeric(bilan_row["_nsfr_obs"], errors="coerce")
                nsfr = float(_v)  if pd.notna(_v) and float(_v) > 0 else None
            else:
                sats_actual = bilan_row[SATELLITE_VARS]
                _, nsfr_calc, _, _ = self._calc_ratios(
                    sats_actual, bilan_row, rsf_w, agg_weights
                )
                nsfr = round(float(nsfr_calc), 2) if np.isfinite(nsfr_calc) else None

            years_out.append(int(yr))
            lcr_out.append(round(lcr,  2) if lcr  is not None else None)
            nsfr_out.append(round(nsfr, 2) if nsfr is not None else None)

        return {"years": years_out, "lcr": lcr_out, "nsfr": nsfr_out}

    def calibration_report(self) -> pd.DataFrame:
        self._check_calibrated()
        rows = []
        for var, r in self._sat_results.items():
            rows.append({
                "satellite": var,
                "family":    r.family,
                "r2":        r.r2,
                "aic":       r.aic,
                "bic":       r.bic,
                "converged": r.converged,
                "signs_ok":  r.signs_ok,
                "n_obs":     r.n_obs,
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# run_liquidity_stress — point d'entrée principal (appelé par test_standalone)
# ─────────────────────────────────────────────────────────────────────────────
def run_liquidity_stress(
    input_path,
    macro_hist: pd.DataFrame,
    scenarios: Dict[str, pd.DataFrame],
    bank_name: str = "",
    country: str = "",
    verbose: bool = True,
    return_engine: bool = False,
    portfolio_type: str = "mixed",
    bs_stress_delta_liab: Optional[Dict[str, float]] = None,
    bs_stress_delta_asset: Optional[Dict[str, float]] = None,
    forced_sat_ranks: Optional[Dict[str, int]] = None,
):
    """
    Point d'entrée principal du module liquidité.

    Parameters
    ----------
    input_path            : chemin vers liquidity_input.xlsx
    macro_hist            : DataFrame macro historique (index ou colonne 'year')
    scenarios             : dict {scenario_id: DataFrame_macro_projetée}
    bank_name             : label banque (metadata)
    country               : label pays (metadata)
    verbose               : afficher le rapport de calibration
    return_engine         : si True, retourne (outputs, engine) au lieu de outputs seul
    portfolio_type        : "retail_heavy" | "mixed" | "corporate_heavy"
                            Sélectionne le poids RSF correct pour la colonne
                            agrégée "loans" (BCBS 295 §46b/d).
    bs_stress_delta_liab  : dict {scenario_id: delta} override pour les passifs
    bs_stress_delta_asset : dict {scenario_id: delta} override pour les actifs
    """
    # ── Normaliser macro_hist ─────────────────────────────────────────────────
    macro_df = _ensure_year_column(macro_hist)

    # ── Charger + calibrer ────────────────────────────────────────────────────
    inputs = LiquidityDataLoader.load(input_path, macro_df)
    engine = LiquidityStressEngine(
        inputs,
        portfolio_type=portfolio_type,
        bs_stress_delta_liab=bs_stress_delta_liab,
        bs_stress_delta_asset=bs_stress_delta_asset,
    )
    engine.calibrate(forced_sat_ranks=forced_sat_ranks or {})

    if verbose:
        _print_calibration_report(engine, bank_name, country)

    # ── Projeter chaque scénario ──────────────────────────────────────────────
    # Normaliser les index des scénarios
    norm_scenarios = {}
    for scen_id, scen_df in scenarios.items():
        norm_scenarios[scen_id] = _ensure_year_index(scen_df)

    results = engine.compute_stress(norm_scenarios)

    # Enrichir metadata
    for scen_id, out in results.items():
        out.metadata.update({"bank_name": bank_name, "country": country})

    if return_engine:
        return results, engine
    return results


def _ensure_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Garantit une colonne 'year' entier (accepte index ou colonne)."""
    df = df.copy()
    if df.index.name == "year" or (
        "year" not in df.columns
        and df.index.dtype in [np.int64, np.float64, int]
    ):
        df = df.reset_index()
        if "index" in df.columns and "year" not in df.columns:
            df = df.rename(columns={"index": "year"})
    if "year" not in df.columns:
        raise ValueError("Le DataFrame macro doit avoir une colonne ou un index 'year'.")
    df["year"] = df["year"].astype(int)
    return df


def _ensure_year_index(df: pd.DataFrame) -> pd.DataFrame:
    """Garantit un index nommé 'year' (accepte colonne ou index)."""
    df = df.copy()
    if "year" in df.columns:
        df = df.set_index("year")
    if df.index.name != "year":
        df.index.name = "year"
    return df


def _print_calibration_report(engine, bank_name, country):
    try:
        from .satellite_calibrator import print_calibration_report
        print_calibration_report(engine._sat_results)
    except Exception:
        # Fallback simple
        try:
            report = engine.calibration_report()
            print(f"\n  -- Calibration - {bank_name} | {country} --")
            print(f"  {'Satellite':<25} {'R2':>7} {'Conv.':>7} {'Signes':>7}")
            print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7}")
            for _, row in report.iterrows():
                print(
                    f"  {row['satellite']:<25} {row['r2']:>7.3f} "
                    f"{'OK' if row['converged'] else 'X':>7} "
                    f"{'OK' if row['signs_ok'] else '!':>7}"
                )
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION — point d'entrée simplifié
# ─────────────────────────────────────────────────────────────────────────────
def build_engine(
    excel_path,
    macro_df: pd.DataFrame,
    portfolio_type: str = "mixed",
) -> LiquidityStressEngine:
    """
    Construit et calibre un LiquidityStressEngine en une seule ligne.

    >>> macro = fetch_macro(country="EGY", start=2010, end=2024)
    >>> engine = build_engine("data/liquidity_input.xlsx", macro, portfolio_type="retail_heavy")
    >>> results = engine.compute_stress(scenarios)
    """
    macro_df = _ensure_year_column(macro_df)
    inputs = LiquidityDataLoader.load(excel_path, macro_df)
    engine = LiquidityStressEngine(inputs, portfolio_type=portfolio_type)
    engine.calibrate()
    return engine
