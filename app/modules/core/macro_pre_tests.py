"""
macro_pre_tests.py  (module partagé — app/modules/core/)
=========================================================
Pipeline de tests pré-estimation sur les variables macro WB/FMI.
Exécuté AVANT l'estimation de N'IMPORTE QUEL module (crédit, liquidité, climat)
dès le déclenchement du run, juste après le fetch macro.

6 blocs séquentiels :
    Bloc 0 — Pertinence stress-test (filtre sémantique — NEW)
    Bloc 1 — Qualité données API (trous, longueur, outliers)
    Bloc 2 — Stationnarité (ADF + KPSS croisés, différenciation auto)
    Bloc 3 — Ruptures structurelles (Zivot-Andrews)
    Bloc 4 — Distribution et mémoire (Breusch-Godfrey LM, ARCH-LM)
    Bloc 5 — Relation avec la variable cible (CCF, Granger, signe)
             [optionnel — sauté si target_series=None, ex. module liquidité]

Filtres durs (rejet automatique) :
    - Bloc 0 : variable sur la liste noire stress-test (si hard_filter_irrelevant=True)
    - Bloc 1 : Trous consécutifs > max_consecutive_na  OU  série < min_obs
    - Bloc 2 : I(1) confirmé par ADF+KPSS, non transformable
    - Bloc 5 : rejet dur UNIQUEMENT si hard_filter_granger=True ET variable absente
               de granger_override (défaut : filtre souple, voir ci-dessous)

Diagnostics (flags, pas de rejet automatique) :
    - Bloc 0 : variable dans zone grise (ni liste blanche ni liste noire)
    - Bloc 3 : rupture structurelle → flag avec date de rupture (Zivot-Andrews)
    - Bloc 5 : Granger-causalité non confirmée → flag granger_flagged=True,
               variable CONSERVÉE ; override analyste via cfg.granger_override
    - Signe incohérent avec expected_sign → flag sign_warning
    - Effet ARCH détecté → flag hac_required

Références :
    Granger & Newbold (1974) — Spurious regressions
    Kwiatkowski, Phillips, Schmidt & Shin (1992) — KPSS test
    Zivot & Andrews (1992) — Unit root with structural break
    Granger (1969) — Causality test
    Engle (1982) — ARCH-LM test
    Breusch (1978) / Godfrey (1978) — LM test for serial correlation
"""
from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("macro_pre_tests")
warnings.filterwarnings("ignore", category=Warning)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BLOC 0 — LISTE BLANCHE / LISTE NOIRE STRESS-TEST                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Variables PERTINENTES pour un stress test bancaire (mots-clés, insensible casse)
_STRESS_RELEVANT = frozenset([
    # Activité réelle
    "gdp", "growth", "recession", "output", "pib", "croissance",
    # Inflation & prix
    "inflation", "cpi", "pce", "deflator", "price", "prix",
    # Taux & monétaire
    "rate", "taux", "interest", "libor", "sofr", "euribor", "policy",
    "yield", "spread", "bps", "overnight",
    # Emploi
    "unemployment", "chomage", "employment", "labor", "labour",
    # Change & flux de capitaux
    "exchange", "fx", "currency", "devise", "rer", "neer",
    "capital_flow", "fdi", "remittance", "reserves",
    # Crédit & secteur bancaire
    "credit", "loan", "npl", "npe", "provision", "tier", "car",
    "roe", "roa", "nim", "deposit", "bank",
    # Marché financier
    "equity", "stock", "vix", "volatility", "index", "market",
    "sovereign", "cds", "bond",
    # Fiscal & dette
    "debt", "deficit", "fiscal", "government", "budget",
    "balance", "account", "current_account",
    # Commerce & énergie
    "trade", "export", "import", "oil", "commodity", "energy",
    "petroleum", "gas",
    # Immobilier
    "housing", "property", "real_estate",
    # Codes WB/IMF fréquents
    "ny.gdp", "fp.cpi", "sl.uem", "pa.nus", "fr.inr",
    "gc.dod", "ne.exp", "ne.imp", "bx.trf", "bm.klt",
    "cm.mkt", "fb.ast", "fd.res",
])

# Variables NON PERTINENTES pour un stress test bancaire
_STRESS_IRRELEVANT = frozenset([
    # Démographie
    "population", "pop_total", "pop_growth", "sp.pop",
    "fertility", "birth_rate", "death_rate", "mortality",
    "life_expectancy", "life_exp",
    # Social / santé
    "literacy", "education", "school",
    "physician", "hospital", "health_exp_per_cap",
    "undernourishment", "malnutrition",
    "water_access", "sanitation",
    # Géographie / infrastructure de base
    "land_area", "forest_area", "arable",
    "electricity_access",
    "rural_pop", "urban_pop",
    # Environnement (sauf si module climat)
    "co2_per_capita", "co2_emission",
    "greenhouse", "methane",
    # Gouvernance / politique
    "gini", "poverty_headcount", "poverty_ratio",
    "military_exp", "arms_export",
    "rule_of_law", "governance_index",
    # Agriculture
    "agricultural_land", "cereal_yield", "food_production",
    # Codes WB spécifiques non financiers
    "sp.dyn", "sh.dyn", "se.adt", "sh.h2o", "sh.xpd",
    "en.atm", "ag.lnd", "ag.prd", "sn.itk", "si.pov",
    "ms.mil", "eg.elc",
])


def _normalize_var(name: str) -> str:
    """Normalise un nom de variable pour comparaison : minuscule, séparateurs → _."""
    return re.sub(r"[\s.\-/]", "_", name.lower())


def _classify_stress_relevance(var_name: str) -> str:
    """
    Retourne 'relevant', 'irrelevant', ou 'unknown'.
    Vérifie si un mot-clé de la liste blanche/noire est contenu dans le nom normalisé.
    """
    norm = _normalize_var(var_name)
    for kw in _STRESS_IRRELEVANT:
        if kw in norm:
            return "irrelevant"
    for kw in _STRESS_RELEVANT:
        if kw in norm:
            return "relevant"
    return "unknown"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class PreTestConfig:
    """Configuration — tous les seuils sont paramétrables, aucun hardcodé."""
    # Bloc 0 — pertinence
    hard_filter_irrelevant: bool  = True   # True → rejeter les vars liste noire
    flag_unknown:           bool  = True   # True → flag les vars zone grise

    # Bloc 1 — qualité
    min_obs:              int   = 10
    max_consecutive_na:   int   = 2
    outlier_mad_factor:   float = 4.0

    # Bloc 2 — stationnarité
    adf_pvalue:           float = 0.05
    kpss_pvalue:          float = 0.05
    auto_difference:      bool  = True

    # Bloc 3 — ruptures
    za_pvalue:            float = 0.05

    # Bloc 4 — distribution
    bg_lags:              int   = 4
    arch_lags:            int   = 4

    # Bloc 5 — relation cible (sauté si target_series=None)
    max_granger_lag:      int        = 4
    granger_pvalue:       float      = 0.10
    ccf_max_lag:          int        = 6
    # Filtre souple Granger : False (défaut) → flag seulement, pas de rejet automatique.
    # True → rejet dur (comportement pré-v2). Override possible variable par variable.
    hard_filter_granger:  bool       = False
    granger_override:     List[str]  = field(default_factory=list)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SORTIE                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class VarTestResult:
    """Résultat des tests pour une variable individuelle."""
    variable:           str
    accepted:           bool           = True
    rejection_reasons:  List[str]      = field(default_factory=list)
    warnings:           List[str]      = field(default_factory=list)
    transformed_as:     Optional[str]  = None

    # Bloc 0
    stress_relevance:   Optional[str]  = None   # 'relevant', 'irrelevant', 'unknown'

    # Bloc 1
    n_obs:              int            = 0
    max_consecutive_na: int            = 0
    n_outliers:         int            = 0

    # Bloc 2
    adf_pvalue:         Optional[float] = None
    kpss_pvalue:        Optional[float] = None
    is_stationary:      Optional[bool]  = None
    integration_order:  Optional[str]   = None

    # Bloc 3
    za_stationary:      Optional[bool]  = None
    za_break_year:      Optional[int]   = None

    # Bloc 4
    breusch_godfrey_p:  Optional[float] = None
    has_autocorrelation: Optional[bool] = None
    arch_lm_pvalue:     Optional[float] = None
    has_arch:           Optional[bool]  = None

    # Bloc 5
    best_ccf_lag:       Optional[int]   = None
    best_ccf_value:     Optional[float] = None
    granger_best_p:     Optional[float] = None
    granger_best_lag:   Optional[int]   = None
    granger_causal:     Optional[bool]  = None
    granger_flagged:    bool            = False   # True → Granger échoue mais var acceptée (filtre souple)
    granger_overridden: bool            = False   # True → inclusion manuelle explicite (override analyste)
    sign_observed:      Optional[int]   = None
    sign_expected:      Optional[int]   = None
    sign_coherent:      Optional[bool]  = None


@dataclass
class MacroPreTestResult:
    """Sortie globale du pipeline de pré-tests."""
    accepted:     List[str]                     = field(default_factory=list)
    rejected:     Dict[str, List[str]]          = field(default_factory=dict)
    transformed:  Dict[str, str]                = field(default_factory=dict)
    var_results:  Dict[str, VarTestResult]      = field(default_factory=dict)
    report:       Dict[str, Any]                = field(default_factory=dict)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BLOC 0 — PERTINENCE STRESS-TEST                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _bloc0_relevance(
    var_name: str, cfg: PreTestConfig, result: VarTestResult,
) -> bool:
    """
    Filtre sémantique : vérifie si la variable a un sens dans un stress test bancaire.

    - 'irrelevant' (liste noire) + hard_filter_irrelevant=True → rejet dur
    - 'unknown' (zone grise)                                   → flag seulement
    - 'relevant' (liste blanche)                               → OK

    Exemples de variables rejetées : population, life_expectancy, literacy_rate,
    co2_per_capita, mortality_rate, agricultural_land, gini, military_expenditure.
    """
    relevance = _classify_stress_relevance(var_name)
    result.stress_relevance = relevance

    if relevance == "irrelevant":
        msg = (
            f"Variable '{var_name}' classifiée hors périmètre stress-test bancaire "
            f"(démographique / sociale / géographique). "
            f"Exemples de vars non pertinentes : population, life_expectancy, "
            f"mortality_rate, literacy_rate, co2_per_capita, agricultural_land."
        )
        if cfg.hard_filter_irrelevant:
            result.rejection_reasons.append(msg)
            return False
        else:
            result.warnings.append(msg + " [filtre souple : non rejetée]")

    elif relevance == "unknown" and cfg.flag_unknown:
        result.warnings.append(
            f"Variable '{var_name}' en zone grise stress-test — "
            f"ni clairement financière/macro ni clairement non pertinente. "
            f"Vérifier manuellement la cohérence économique."
        )

    return True


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BLOC 1 — QUALITÉ DES DONNÉES API                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _max_consecutive_nans(series: pd.Series) -> int:
    mask = series.isna()
    if not mask.any():
        return 0
    groups = (mask != mask.shift()).cumsum()
    return int(mask.groupby(groups).sum().max())


def _detect_outliers_mad(series: pd.Series, factor: float = 4.0) -> int:
    s = series.dropna()
    if len(s) < 5:
        return 0
    med = s.median()
    mad = np.median(np.abs(s - med))
    if mad < 1e-12:
        return 0
    modified_z = 0.6745 * (s - med) / mad
    return int((np.abs(modified_z) > factor).sum())


def _bloc1_quality(
    series: pd.Series, cfg: PreTestConfig, result: VarTestResult,
) -> bool:
    s = series.dropna()
    result.n_obs = len(s)
    result.max_consecutive_na = _max_consecutive_nans(series)
    result.n_outliers = _detect_outliers_mad(s, cfg.outlier_mad_factor)

    if result.n_obs < cfg.min_obs:
        result.rejection_reasons.append(
            f"n_obs={result.n_obs} < min_obs={cfg.min_obs}"
        )
        return False

    if result.max_consecutive_na > cfg.max_consecutive_na:
        result.rejection_reasons.append(
            f"max_consecutive_NA={result.max_consecutive_na} "
            f"> seuil={cfg.max_consecutive_na}"
        )
        return False

    if result.n_outliers > 0:
        result.warnings.append(
            f"{result.n_outliers} outlier(s) MAD détecté(s) — à inspecter"
        )
    return True


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BLOC 2 — STATIONNARITÉ (ADF + KPSS croisés)                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _safe_adf(series: pd.Series) -> Optional[float]:
    try:
        from statsmodels.tsa.stattools import adfuller
        s = series.dropna().values
        if len(s) < 8:
            return None
        max_lag = min(int(len(s) ** 0.5), len(s) // 4)
        result = adfuller(s, maxlag=max(1, max_lag), autolag="AIC", regression="c")
        return float(result[1])
    except Exception:
        return None


def _safe_kpss(series: pd.Series) -> Optional[float]:
    try:
        from statsmodels.tsa.stattools import kpss
        s = series.dropna().values
        if len(s) < 8:
            return None
        nlags = min(int(4 * (len(s) / 100) ** 0.25), len(s) // 4)
        _, p_val, _, _ = kpss(s, regression="c", nlags=max(1, nlags))
        return float(p_val)
    except Exception:
        return None


def _bloc2_stationarity(
    series: pd.Series, cfg: PreTestConfig, result: VarTestResult,
) -> Tuple[bool, pd.Series]:
    s = series.dropna()
    result.adf_pvalue = _safe_adf(s)
    result.kpss_pvalue = _safe_kpss(s)

    if result.adf_pvalue is None and result.kpss_pvalue is None:
        result.is_stationary = True
        result.integration_order = "unknown"
        result.warnings.append(
            "Tests stationnarité non exécutables (série trop courte) — acceptée par défaut"
        )
        return True, s

    adf_reject  = result.adf_pvalue  is not None and result.adf_pvalue  < cfg.adf_pvalue
    kpss_reject = result.kpss_pvalue is not None and result.kpss_pvalue < cfg.kpss_pvalue

    if adf_reject and not kpss_reject:
        result.is_stationary = True
        result.integration_order = "I(0)"
        return True, s

    if not adf_reject and kpss_reject:
        result.is_stationary = False
        result.integration_order = "I(1)"
        if cfg.auto_difference:
            s_diff = s.diff().dropna()
            adf_diff = _safe_adf(s_diff)
            if adf_diff is not None and adf_diff < cfg.adf_pvalue:
                result.transformed_as = "diff_1"
                result.integration_order = "I(1)→diff→I(0)"
                result.warnings.append(
                    "Variable I(1) → différence première appliquée automatiquement"
                )
                return True, s_diff
            else:
                result.rejection_reasons.append(
                    f"I(1) confirmé (ADF p={result.adf_pvalue:.3f}, "
                    f"KPSS p={result.kpss_pvalue:.3f}) — "
                    f"différence première reste non-stationnaire"
                )
                return False, s
        result.rejection_reasons.append(
            f"I(1) confirmé (ADF p={result.adf_pvalue:.3f}, "
            f"KPSS p={result.kpss_pvalue:.3f}) — auto_difference désactivé"
        )
        return False, s

    if adf_reject and kpss_reject:
        result.is_stationary = True
        result.integration_order = "trend-stationary"
        result.warnings.append(
            "Trend-stationnaire — considérer un filtre HP/BK en amont"
        )
        return True, s

    result.is_stationary = True
    result.integration_order = "ambiguous"
    result.warnings.append(
        f"Stationnarité ambiguë (ADF p={result.adf_pvalue}, "
        f"KPSS p={result.kpss_pvalue}) — acceptée par défaut, à vérifier"
    )
    return True, s


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BLOC 3 — RUPTURES STRUCTURELLES (Zivot-Andrews)                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _bloc3_breaks(
    series: pd.Series, cfg: PreTestConfig, result: VarTestResult,
) -> None:
    try:
        from statsmodels.tsa.stattools import zivot_andrews
        s = series.dropna()
        if len(s) < 15:
            return
        za_stat, za_pval, _, _, za_bpidx = zivot_andrews(
            s.values, model="both", autolag="AIC"
        )
        result.za_stationary = za_pval < cfg.za_pvalue
        if s.index.dtype in ("int64", "int32", "Int64"):
            result.za_break_year = int(s.index[za_bpidx])
        else:
            result.za_break_year = int(za_bpidx)
        if not result.za_stationary and result.is_stationary is False:
            result.warnings.append(
                f"Rupture structurelle détectée en {result.za_break_year} "
                f"(ZA p={za_pval:.3f}) — la non-stationnarité peut être "
                f"due à cette rupture plutôt qu'à une vraie racine unitaire"
            )
    except Exception as e:
        LOG.debug("Zivot-Andrews failed for %s: %s", series.name, e)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BLOC 4 — DISTRIBUTION ET MÉMOIRE                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _bloc4_distribution(
    series: pd.Series, cfg: PreTestConfig, result: VarTestResult,
) -> None:
    s = series.dropna()
    if len(s) < 10:
        return

    try:
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import acorr_breusch_godfrey
        # Breusch-Godfrey teste l'autocorrélation des résidus d'UNE
        # régression, pas d'une série brute — contrairement au Ljung-Box
        # qu'il remplace ici. On ajuste donc un AR(1) sur la variable
        # elle-même (y_t = c + rho·y_{t-1} + e_t), puis on applique BG sur
        # ses résidus : le test devient "reste-t-il de l'autocorrélation
        # AU-DELÀ de ce qu'un AR(1) capture déjà ?", plus rigoureux que le
        # Ljung-Box sur niveaux bruts (qui ignore la persistance propre de
        # la série).
        lags = min(cfg.bg_lags, len(s) // 3)
        if lags >= 1 and len(s) > lags + 2:
            y = s.values[1:]
            y_lag = s.values[:-1]
            ar1_fit = sm.OLS(y, sm.add_constant(y_lag)).fit()
            bg_stat, bg_pvalue, _, _ = acorr_breusch_godfrey(ar1_fit, nlags=lags)
            p_val = float(bg_pvalue)
            result.breusch_godfrey_p = p_val
            result.has_autocorrelation = p_val < 0.05
            if result.has_autocorrelation:
                result.warnings.append(
                    f"Autocorrélation résiduelle significative au-delà de "
                    f"l'AR(1) (Breusch-Godfrey p={p_val:.3f}) — considérer "
                    f"un lag supplémentaire dans le satellite ou erreurs HAC"
                )
    except Exception:
        pass

    try:
        from statsmodels.stats.diagnostic import het_arch
        nlags = min(cfg.arch_lags, len(s) // 4)
        if nlags >= 1:
            _, arch_p, _, _ = het_arch(s.values, nlags=nlags)
            result.arch_lm_pvalue = float(arch_p)
            result.has_arch = arch_p < 0.05
            if result.has_arch:
                result.warnings.append(
                    f"Effet ARCH détecté (p={arch_p:.3f}) — "
                    f"utiliser erreurs HAC Newey-West dans le satellite"
                )
    except Exception:
        pass


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BLOC 5 — RELATION AVEC LA VARIABLE CIBLE (optionnel)                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _compute_ccf(
    x: np.ndarray, y: np.ndarray, max_lag: int,
) -> List[Tuple[int, float]]:
    results = []
    n = len(x)
    for lag in range(0, max_lag + 1):
        if lag >= n - 3:
            break
        x_lagged = x[:n - lag] if lag > 0 else x
        y_current = y[lag:]
        min_len = min(len(x_lagged), len(y_current))
        if min_len < 5:
            continue
        x_l, y_c = x_lagged[:min_len], y_current[:min_len]
        denom = np.std(x_l) * np.std(y_c) * min_len
        if denom < 1e-12:
            continue
        corr = float(np.sum((x_l - x_l.mean()) * (y_c - y_c.mean())) / denom)
        results.append((lag, corr))
    results.sort(key=lambda t: -abs(t[1]))
    return results


def _safe_granger(
    y: np.ndarray, x: np.ndarray, max_lag: int,
) -> Optional[Tuple[float, int]]:
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        data = np.column_stack([y, x])
        n = len(data)
        max_testable = min(max_lag, n // 3 - 1)
        if max_testable < 1:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = grangercausalitytests(data, maxlag=max_testable, verbose=False)
        best_p, best_lag = 1.0, 1
        for lag in range(1, max_testable + 1):
            if lag in results:
                tests = results[lag][0]
                p_ssr = tests.get("ssr_ftest", (None, 1.0))[1]
                p_f   = tests.get("params_ftest", (None, 1.0))[1]
                p = min(p_ssr, p_f)
                if p < best_p:
                    best_p, best_lag = p, lag
        return best_p, best_lag
    except Exception:
        return None


def _bloc5_relation_target(
    series: pd.Series,
    target_series: pd.Series,
    expected_sign: int,
    cfg: PreTestConfig,
    result: VarTestResult,
) -> bool:
    common = series.dropna().index.intersection(target_series.dropna().index)
    if len(common) < 8:
        result.warnings.append(
            f"Seulement {len(common)} observations communes avec la cible — "
            f"tests bivariés non fiables"
        )
        return True

    x = series.loc[common].values.astype(float)
    y = target_series.loc[common].values.astype(float)

    ccf_results = _compute_ccf(x, y, cfg.ccf_max_lag)
    if ccf_results:
        result.best_ccf_lag   = ccf_results[0][0]
        result.best_ccf_value = round(ccf_results[0][1], 4)

    granger = _safe_granger(y, x, cfg.max_granger_lag)
    if granger is not None:
        result.granger_best_p, result.granger_best_lag = granger
        result.granger_best_p = round(result.granger_best_p, 4)
        result.granger_causal = result.granger_best_p < cfg.granger_pvalue

        if not result.granger_causal:
            result.granger_flagged = True
            result.warnings.append(
                f"Granger-causalité X→cible non confirmée "
                f"(meilleur p={result.granger_best_p:.3f} au lag "
                f"{result.granger_best_lag}, seuil={cfg.granger_pvalue}) "
                f"— variable conservée (filtre souple) ; "
                f"vérifier la justification économique avant inclusion."
            )
    else:
        result.warnings.append("Test de Granger non exécutable — acceptée par défaut")

    result.sign_expected = expected_sign
    if len(x) >= 5:
        corr = float(np.corrcoef(x, y)[0, 1])
        result.sign_observed = int(np.sign(corr)) if abs(corr) > 0.05 else 0
        if expected_sign != 0 and result.sign_observed != 0:
            result.sign_coherent = (result.sign_observed == expected_sign)
            if not result.sign_coherent:
                result.warnings.append(
                    f"Signe incohérent : observé={result.sign_observed:+d}, "
                    f"attendu={expected_sign:+d} — corrélation brute = {corr:.3f}"
                )
        else:
            result.sign_coherent = True

    return True


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ORCHESTRATEUR PRINCIPAL                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_macro_pre_tests(
    macro_df: pd.DataFrame,
    target_series: Optional[pd.Series] = None,
    universe_idx: Optional[Dict[str, Any]] = None,
    config: Optional[PreTestConfig] = None,
    module: str = "credit",
    # Alias rétrocompatibilité
    pd_series: Optional[pd.Series] = None,
) -> MacroPreTestResult:
    """
    Pipeline complet de pré-tests sur les variables macro.

    Appelé AVANT l'estimation de n'importe quel module, dès le fetch macro.

    Args:
        macro_df      : DataFrame (index=year, colonnes=variables macro)
        target_series : Série cible historique (PD pour crédit, None pour liquidité)
                        → si None, Bloc 5 (Granger/CCF) est sauté
        universe_idx  : {var_name: MacroIndicator} pour expected_sign
        config        : PreTestConfig (None → défauts)
        module        : 'credit' | 'liquidity' | 'climate' — contrôle le Bloc 5
        pd_series     : alias rétrocompatible pour target_series (credit module)

    Returns:
        MacroPreTestResult avec accepted / rejected / transformed / var_results / report
    """
    # Rétrocompatibilité : pd_series était le nom original dans le module crédit
    if target_series is None and pd_series is not None:
        target_series = pd_series

    # Pour le module liquidité et climat → pas de variable cible → sauter Bloc 5
    run_bloc5 = (target_series is not None) and (module == "credit")

    cfg = config or PreTestConfig()
    out = MacroPreTestResult()
    n_total = len(macro_df.columns)

    LOG.info("=" * 60)
    LOG.info("MACRO PRE-TESTS [module=%s] : %d variables candidates", module, n_total)
    LOG.info("Bloc 0 actif (filtre stress-relevance, hard=%s)", cfg.hard_filter_irrelevant)
    LOG.info("Bloc 5 %s (target_series=%s)", "ACTIF" if run_bloc5 else "SAUTÉ",
             "fournie" if target_series is not None else "None")
    LOG.info("=" * 60)

    for col in macro_df.columns:
        series = macro_df[col].copy()
        series.name = col
        vr = VarTestResult(variable=col)

        # ── Bloc 0 : pertinence stress-test ────────────────────────────
        if not _bloc0_relevance(col, cfg, vr):
            vr.accepted = False
            out.rejected[col] = vr.rejection_reasons
            out.var_results[col] = vr
            LOG.info("  %-35s → REJETÉ Bloc0 (hors périmètre stress-test)", col)
            continue

        # ── Bloc 1 : qualité ───────────────────────────────────────────
        if not _bloc1_quality(series, cfg, vr):
            vr.accepted = False
            out.rejected[col] = vr.rejection_reasons
            out.var_results[col] = vr
            LOG.info("  %-35s → REJETÉ Bloc1 %s", col, vr.rejection_reasons)
            continue

        # ── Bloc 2 : stationnarité ─────────────────────────────────────
        passed_b2, series_clean = _bloc2_stationarity(series, cfg, vr)
        if not passed_b2:
            vr.accepted = False
            out.rejected[col] = vr.rejection_reasons
            out.var_results[col] = vr
            LOG.info("  %-35s → REJETÉ Bloc2 %s", col, vr.rejection_reasons)
            continue
        if vr.transformed_as:
            out.transformed[col] = vr.transformed_as

        # ── Bloc 3 : ruptures (diagnostic) ────────────────────────────
        _bloc3_breaks(series, cfg, vr)

        # ── Bloc 4 : distribution & mémoire (diagnostic) ──────────────
        _bloc4_distribution(series_clean, cfg, vr)

        # ── Bloc 5 : relation avec la cible (optionnel) ────────────────
        if run_bloc5:
            exp_sign = 0
            if universe_idx and col in universe_idx:
                exp_sign = getattr(universe_idx[col], "expected_sign", 0)
            _bloc5_relation_target(series_clean, target_series,
                                   exp_sign, cfg, vr)

            # Override analyste : inclusion explicite malgré Granger négatif
            if vr.granger_flagged and col in cfg.granger_override:
                vr.granger_overridden = True
                vr.warnings.append(
                    f"Override analyste : '{col}' incluse manuellement "
                    f"malgré l'échec Granger (p={vr.granger_best_p})."
                )
                LOG.info(
                    "  %-35s → OVERRIDE Bloc5 (Granger p=%.3f, inclusion manuelle)",
                    col, vr.granger_best_p or 1.0,
                )

            # Rejet dur uniquement si hard_filter_granger=True ET pas d'override
            elif vr.granger_flagged and cfg.hard_filter_granger:
                vr.accepted = False
                vr.rejection_reasons.append(
                    f"Granger-causalité rejetée (p={vr.granger_best_p:.3f} "
                    f"> {cfg.granger_pvalue}) — rejet dur (hard_filter_granger=True)"
                )
                out.rejected[col] = vr.rejection_reasons
                out.var_results[col] = vr
                LOG.info(
                    "  %-35s → REJETÉ Bloc5 (hard_filter_granger) p=%.3f",
                    col, vr.granger_best_p or 1.0,
                )
                continue

        # ── Variable acceptée ──────────────────────────────────────────
        vr.accepted = True
        out.accepted.append(col)
        out.var_results[col] = vr

        flags = f" [{'; '.join(vr.warnings[:2])}]" if vr.warnings else ""
        LOG.info("  %-35s → ACCEPTÉ  %-18s %s%s",
                 col, vr.integration_order or "",
                 f"Granger p={vr.granger_best_p:.3f}" if vr.granger_best_p else "",
                 flags)

    # ── Rapport global ─────────────────────────────────────────────────
    n_accepted    = len(out.accepted)
    n_rejected    = len(out.rejected)
    n_transformed = len(out.transformed)
    n_irrelevant  = sum(1 for vr in out.var_results.values()
                        if vr.stress_relevance == "irrelevant" and not vr.accepted)

    out.report = {
        "module":          module,
        "n_total":         n_total,
        "n_accepted":      n_accepted,
        "n_rejected":      n_rejected,
        "n_transformed":   n_transformed,
        "n_irrelevant":    n_irrelevant,
        "acceptance_rate": round(n_accepted / max(n_total, 1), 3),
        "bloc5_active":    run_bloc5,
        "rejected_vars":   out.rejected,
        "transformed_vars": out.transformed,
        "per_variable": {
            col: {
                "accepted":            vr.accepted,
                "stress_relevance":    vr.stress_relevance,
                "integration_order":   vr.integration_order,
                "adf_p":               vr.adf_pvalue,
                "kpss_p":              vr.kpss_pvalue,
                "granger_p":           vr.granger_best_p,
                "granger_lag":         vr.granger_best_lag,
                "ccf_best_lag":        vr.best_ccf_lag,
                "sign_coherent":       vr.sign_coherent,
                "has_autocorrelation": vr.has_autocorrelation,
                "has_arch":            vr.has_arch,
                "za_break_year":       vr.za_break_year,
                "warnings":            vr.warnings,
                "rejection_reasons":   vr.rejection_reasons,
            }
            for col, vr in out.var_results.items()
        },
    }

    LOG.info("-" * 60)
    LOG.info("RÉSULTAT [%s]: %d/%d acceptées, %d rejetées (%d hors périmètre), %d transformées",
             module, n_accepted, n_total, n_rejected, n_irrelevant, n_transformed)
    if out.rejected:
        for var, reasons in out.rejected.items():
            LOG.info("  REJETÉ %-28s : %s", var, "; ".join(reasons))
    LOG.info("=" * 60)

    return out