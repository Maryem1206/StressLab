"""
capital_engine.py  — v4
=======================
Moteur ASRF VaR + Capital Bâle III — 100 % générique, zéro paramètre exogène
non observable.

Formules ASRF (Pilier 1 — BCBS CRE31) :
    VaR_α  = LGD_DT × Φ[(Φ⁻¹(PD) − √ρ(PD)·Φ⁻¹(1−α)) / √(1−ρ(PD))]
    K      = C_M(PD) × max(VaR_α − EL, 0)
    RWA    = K × 12.5 × EAD

RWA stressé (Approche BCE/EBA §4.2 — delta ASRF) :
    ΔRWA_ratio_t = (K_scénario_t − K_baseline_t) / K_baseline_t
    RWA_stressed = RWA_rapport × (EAD_t / EAD_0) × (1 + ΔRWA_ratio_t)
    RWA_total    = RWA_crédit + RWA_marché + RWA_opérationnel

Capital stressé (Approche B+ — NI stressé + cascade Bâle III) :
    ΔEL_t       = EL_stress_t − EL_baseline_t
    PPNR_t      = ppnr_rate_hist × RWA_t × atténuation_scénario
    NI_t        = (PPNR_t − ΔEL_t) × (1 − τ_hist)
    Capital_t   = Capital_{t-1} + NI_t (cascade CET1 → AT1 → T2)

Hypothèse documentée :
    Si PPNR / tax_paid absents du CSV historique → fallback NI = 0
    (identique à v3 : « static balance sheet » EBA). Compatibilité
    ascendante garantie (Principe #2).

Calibrations data-driven (Principe #3) :
    • ppnr_rate   = _historical_ratio(ppnr, rwa)
    • tax_rate    = _historical_ratio(tax_paid, pretax_income)
    • severity    = _calibrate_severity(pd_arrays)

Choix théoriques défendables :
    • ρ(PD) dynamique      : BCBS CRE31.44 — corrélation décroissante avec PD
    • PD floor = 0.03 %    : BCBS CRE31.15 — floor réglementaire
    • LGD downturn          : BCBS CRE36.86 / EBA GL 2019/03
    • EAD co-cyclique       : EBA §3.3 — volume de crédit scénario-sensible
    • Maturity M ∈ [1, 5]   : BCBS CRE32.47 — contrainte réglementaire
    • Cascade CET1→AT1→T2  : CRR2 Art. 60–62 — absorption des pertes

Références :
    BCBS (2017) CRE31 — IRB formula (ASRF)
    BCBS (2017) CRE32 — Maturity adjustment
    BCBS (2017) CRE36 — LGD downturn estimation
    EBA (2023) Methodological Note — EU-wide stress test
    CRR2 Art. 153–154, 160–161, 429
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

LOG = logging.getLogger("capital_engine")

# ── Floors / caps réglementaires ──────────────────────────────────────────
PD_FLOOR  = 0.0003    # BCBS CRE31.15 — 30 bp, corporate/retail
PD_CAP    = 0.9999
LGD_FLOOR = 0.01      # Floor pratique (LGD = 0 annulerait tout le capital)
LGD_CAP   = 1.0
M_MIN     = 1.0       # BCBS CRE32.47
M_MAX     = 5.0

# ── Synonymes de colonnes pour auto-détection CSV ─────────────────────────
_COL_SYNONYMS: Dict[str, set] = {
    "year":              {"year", "année", "annee", "date", "yr"},
    "tier1":             {"tier1", "tier_1", "tier 1", "t1", "capital_tier1",
                          "tier1_egp_m", "tier_1_capital",
                          "tier1_total", "tier 1 total", "tier1total",
                          "total_tier1", "total tier1"},
    "total_capital":     {"total_capital", "capital_total", "fonds_propres",
                          "total_cap", "capital total", "totalcapital",
                          "total_capital_egp_m"},
    "rwa":               {"rwa", "risk_weighted_assets", "actifs_ponderes",
                          "rwa_egp_m"},
    "leverage_exposure": {"leverage_exposure", "expo_levier", "total_exposure",
                          "exposition_levier", "total exposure",
                          "leverage_exposure_egp_m"},
    "cet1":              {"cet1", "cet_1", "common_equity_tier1"},
    # ── Colonnes P&L (Modification #1) ───────────────────────────────────
    "provisions":        {"provisions", "provision", "ecl_provisions",
                          "provisions_egp_m", "cout_du_risque", "cost_of_risk",
                          "loan_loss_provisions", "llp"},
    "ppnr":              {"ppnr", "pre_provision_net_revenue",
                          "ppnr_egp_m", "resultat_avant_provisions",
                          "pre_provision_profit", "operating_income_before_provisions"},
    "net_income":        {"net_income", "resultat_net", "net_profit",
                          "net_income_egp_m", "benefice_net"},
    "tax_paid":          {"tax_paid", "impots", "taxes", "tax_paid_egp_m",
                          "impots_payes", "income_tax", "tax_expense"},
    "pretax_income":     {"pretax_income", "resultat_avant_impots",
                          "pretax_income_egp_m", "profit_before_tax",
                          "pbt", "ebt", "resultat_brut"},
    # ── Cascade capital Bâle III (Modification #1) ───────────────────────
    "at1":               {"at1", "at_1", "additional_tier1", "tier1_additionnel",
                          "additional_tier_1"},
    "tier2":             {"tier2", "tier_2", "t2", "capital_tier2",
                          "tier2_capital", "fonds_complementaires"},
    # ── RWA par type de risque (Modification #1) ─────────────────────────
    "rwa_market":        {"rwa_market", "rwa_marche", "market_rwa",
                          "risque_marche_rwa"},
    "rwa_operational":   {"rwa_operational", "rwa_operationnel",
                          "operational_rwa", "risque_operationnel_rwa",
                          "oprisk_rwa"},
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. CORRÉLATION D'ACTIFS ρ(PD) — BCBS CRE31.44                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def asset_correlation(
    pd_v: float,
    portfolio_type: str = "corporate",
    sme_turnover_meur: Optional[float] = None,
) -> float:
    """
    ρ(PD) — BCBS CRE31.44–57 / CRR2 Art. 153.

    Corporate : ρ = 0.12×f + 0.24×(1−f)   avec f = (1−e^{−50·PD})/(1−e^{−50})
    SME corp. : ρ_SME = ρ(PD) − 0.04×(1 − max(S−5,0)/45)
    Retail mortgage    : 0.15 (fixe)
    Retail renouvelable: 0.04 (fixe)
    Retail autre       : 0.03–0.16 (formule modifiée, exp = −35)
    """
    pd_v = float(np.clip(pd_v, PD_FLOOR, PD_CAP))

    if portfolio_type == "retail_mortgage":
        return 0.15
    if portfolio_type == "retail_revolving":
        return 0.04

    exp_term = (1.0 - np.exp(-50.0 * pd_v)) / (1.0 - np.exp(-50.0))
    rho = 0.12 * exp_term + 0.24 * (1.0 - exp_term)

    if portfolio_type == "retail_other":
        exp_r = (1.0 - np.exp(-35.0 * pd_v)) / (1.0 - np.exp(-35.0))
        return float(0.03 * exp_r + 0.16 * (1.0 - exp_r))

    if portfolio_type == "sme_corporate":
        s = float(np.clip(sme_turnover_meur or 5.0, 5.0, 50.0))
        return float(max(rho - 0.04 * (1.0 - max(s - 5.0, 0.0) / 45.0), 0.001))

    return float(rho)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. FORMULES ASRF                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def var_asrf(pd_v: float, lgd_dt: float, rho: float, alpha: float = 0.999) -> float:
    """VaR ASRF — BCBS CRE31.2."""
    pd_v   = float(np.clip(pd_v,   PD_FLOOR,  PD_CAP))
    lgd_dt = float(np.clip(lgd_dt, LGD_FLOOR, LGD_CAP))
    rho    = float(np.clip(rho, 1e-6, 1.0 - 1e-6))
    num    = norm.ppf(pd_v) - np.sqrt(rho) * norm.ppf(1.0 - alpha)
    return float(lgd_dt * norm.cdf(num / np.sqrt(1.0 - rho)))


def maturity_adjustment(pd_v: float, M: float = 2.5) -> float:
    """Facteur d'ajustement de maturité — BCBS CRE32.47."""
    pd_v = float(np.clip(pd_v, PD_FLOOR, PD_CAP))
    if not (M_MIN <= M <= M_MAX):
        raise ValueError(
            f"Maturity M={M:.2f} hors [{M_MIN}, {M_MAX}] (BCBS CRE32.47)."
        )
    b = (0.11852 - 0.05478 * np.log(pd_v)) ** 2
    denom = 1.0 - 1.5 * b
    if denom <= 0:
        return 1.0
    return float((1.0 + (M - 2.5) * b) / denom)


def capital_requirement(
    pd_v: float, lgd_dt: float, rho: float,
    alpha: float = 0.999, M: float = 2.5,
) -> Dict[str, float]:
    """K, VaR, EL, CM, RWA_ratio — BCBS CRE31."""
    VaR = var_asrf(pd_v, lgd_dt, rho, alpha)
    EL  = float(np.clip(pd_v, PD_FLOOR, PD_CAP) *
                np.clip(lgd_dt, LGD_FLOOR, LGD_CAP))
    CM  = maturity_adjustment(pd_v, M)
    K   = float(max(CM * (VaR - EL), 0.0))
    return {"K": K, "VaR": VaR, "EL": EL, "CM": CM, "RWA_ratio": K * 12.5}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. LGD DOWNTURN — BCBS CRE36.86                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def lgd_downturn(
    lgd_baseline: float,
    scenario_severity: float = 0.0,
    lgd_dt_floor: float = 0.0,
    lgd_dt_stress_add: float = 0.0,
) -> float:
    """
    LGD_DT = max(LGD, LGD_LR) + δ × severity.
    severity ∈ [0, 1] : 0 = baseline, 0.5 = adverse, 1.0 = severe.
    Réf. : Frye 2000, Altman et al. 2005 — corrélation positive PD / (1−RR).

    δ (lgd_dt_stress_add) : surcharge additive downturn au-dessus du LGD Frye-Jacobs.
    Défaut 0.0 — Frye-Jacobs encode déjà la cyclicité PD/LGD ; δ>0 produit
    un double comptage. Utiliser δ>0 uniquement si lgd_baseline est un LGD TTC plat
    (non Frye-Jacobs) qui ne capte pas le lien PD/LGD en downturn.
    """
    lgd_stressed = max(lgd_baseline, lgd_dt_floor) + lgd_dt_stress_add * scenario_severity
    return float(np.clip(lgd_stressed, LGD_FLOOR, LGD_CAP))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. CHARGEMENT CSV                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _detect_col(df_cols: List[str], target: str) -> Optional[str]:
    import re
    synonyms = _COL_SYNONYMS.get(target, set())
    for c in df_cols:
        normalized = re.sub(r"\s*\(.*?\)\s*", "", c).lower().strip()
        if normalized in synonyms:
            return c
    return None


def load_capital_history(
    path: Optional[str],
    unit_multiplier: float = 1.0,
) -> Optional[pd.DataFrame]:
    """
    Charge capital_historique.csv et normalise en unités absolues.

    unit_multiplier : 1e6 si CSV en millions, 1e9 si milliards, 1.0 si absolu.
    IMPORTANT : seul point de normalisation — l'engine ne multiplie jamais après.
    """
    if not path:
        return None
    p = Path(str(path))
    if not p.exists():
        LOG.info("Capital file not found at '%s'.", path)
        return None

    try:
        if p.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        else:
            for sep in [",", ";", "\t"]:
                try:
                    df = pd.read_csv(p, sep=sep)
                    if len(df.columns) >= 3:
                        break
                except Exception:
                    continue
            else:
                return None
    except Exception as e:
        LOG.warning("Failed to read capital file: %s", e)
        return None

    if len(df.columns) < 3:
        return None

    col_map: Dict[str, str] = {}
    for target in ("year", "tier1", "total_capital", "rwa",
                   "leverage_exposure", "cet1",
                   # ── Nouvelles colonnes (Modification #1) ──────────────
                   "provisions", "ppnr", "net_income",
                   "tax_paid", "pretax_income",
                   "at1", "tier2",
                   "rwa_market", "rwa_operational"):
        detected = _detect_col(list(df.columns), target)
        if detected:
            col_map[target] = detected

    if "year" not in col_map:
        return None
    if "rwa" not in col_map and "total_capital" not in col_map:
        return None

    rename = {v: k for k, v in col_map.items()}
    out = df.rename(columns=rename)
    out["year"] = out["year"].astype(str).str.replace(r"\s+", "", regex=True)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["year"]).set_index("year").sort_index()

    for col in ("tier1", "total_capital", "rwa", "leverage_exposure", "cet1",
                # ── Nouvelles colonnes (Modification #1) ─────────────────
                "provisions", "ppnr", "net_income",
                "tax_paid", "pretax_income",
                "at1", "tier2",
                "rwa_market", "rwa_operational"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.replace(r"\s+", "", regex=True)
            out[col] = pd.to_numeric(out[col], errors="coerce")
            if unit_multiplier != 1.0:
                out[col] = out[col] * unit_multiplier

    # Unit consistency check: implied CAR > 200% means capital and RWA are in
    # different units. Guessing the scale factor is not acceptable in a regulatory
    # model — raise an error so the user fixes their input explicitly.
    if "rwa" in out.columns and "total_capital" in out.columns:
        _sample = out[["total_capital", "rwa"]].dropna()
        if len(_sample) >= 1:
            _avg_car = (_sample["total_capital"] / _sample["rwa"]).mean()
            if _avg_car > 2.0:
                raise ValueError(
                    f"Capital CSV unit mismatch detected: implied average CAR is "
                    f"{_avg_car * 100:.1f}%, which exceeds 200%. This means "
                    f"'total_capital' and 'rwa' are in different units "
                    f"(e.g. one in millions, the other in billions). "
                    f"Please ensure all columns use the same unit, or pass "
                    f"unit_multiplier to rescale before loading."
                )

    LOG.info("Capital file loaded: %d years, columns=%s", len(out), list(out.columns))
    return out


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5. SORTIE                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class CapitalOutput:
    """Sortie du moteur capital — consommée par le Module 5."""
    historical_df:        Optional[pd.DataFrame]
    stress_df:            pd.DataFrame
    kpis:                 Dict[str, Any]
    capital_trajectories: Dict[str, Dict]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6. MOTEUR PRINCIPAL                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class CreditCapitalEngine:
    """
    Moteur ASRF + Capital Bâle III — v4 (NI stressé + cascade + data-driven).

    v4 vs v3 :
        • NI stressé (PPNR − ΔEL) × (1−τ) au lieu de NI = 0
        • Sévérité calibrée dynamiquement depuis les PD projetées
        • Cascade capital : pertes absorbent CET1 → AT1 → T2
        • RWA total = RWA_crédit + RWA_marché + RWA_opérationnel
        • ppnr_rate et tax_rate calculés historiquement (Principe #3)
        • Compatibilité v3 totale si colonnes P&L absentes (Principe #2)

    Paramètres (tous dans params dict) :
        var_alpha               : niveau de confiance VaR (défaut 0.999)
        loan_maturity           : maturité effective M ∈ [1, 5] ans
        portfolio_type          : type portefeuille pour ρ(PD)
        sme_turnover_meur       : CA SME en M€ (si sme_corporate)
        portfolio_share         : fraction du bilan couverte (défaut 1.0)
        min_car / min_tier1 / min_cet1 / min_leverage : seuils réglementaires
        ead_growth_by_scenario  : {baseline, adverse, severe} taux croissance EAD
        lgd_dt_floor            : LGD long-run plancher
        lgd_dt_stress_add       : surcharge additive downturn (défaut 0.0 ; voir lgd_downturn())
        leverage_exposure_factor: ratio TEM/EAD si expo absente du CSV
        leverage_exposure_growth: taux croissance exposition levier

    Aucun ROTE, payout, rote_stress, tax_rate, ppnr_rate requis en params.
    Tout est calculé depuis l'historique CSV (Principe #3).
    """

    # ── Sévérité : calibrée dynamiquement (Modification #3) ────────────
    _SEVERITY_FALLBACK = {"baseline": 0.0, "adverse": 0.5, "severe": 1.0}

    def __init__(self, params: Dict[str, Any]):
        self.alpha          = float(params.get("var_alpha", 0.999))
        self.M              = float(params.get("loan_maturity", 2.5))
        if not (M_MIN <= self.M <= M_MAX):
            raise ValueError(
                f"loan_maturity={self.M} hors [{M_MIN}, {M_MAX}] (BCBS CRE32.47)."
            )
        self.portfolio_type = str(params.get("portfolio_type", "corporate"))
        self.sme_turnover   = params.get("sme_turnover_meur", None)
        self.portfolio_share = float(params.get("portfolio_share", 1.0))

        self.min_car      = float(params.get("min_car",      0.1275))
        self.min_tier1    = float(params.get("min_tier1",     0.085))
        self.min_cet1     = float(params.get("min_cet1",      0.07))
        self.min_leverage = float(params.get("min_leverage",  0.030))

        # EAD co-cyclique par scénario — doit toujours être fourni par le wrapper
        # à partir du CAGR historique calculé sur la série EAD du dataset.
        # Aucune valeur par défaut : une croissance EAD inventée biaise silencieusement
        # toutes les projections RWA et ECL.
        if "ead_growth_rate" not in params and "ead_growth_by_scenario" not in params:
            raise ValueError(
                "ead_growth_rate is required and must be computed from the historical "
                "EAD series (CAGR). The credit wrapper computes this automatically. "
                "If calling CreditCapitalEngine directly, pass ead_growth_rate as the "
                "compound annual growth rate of EAD from your input dataset."
            )
        _g_bl = float(params["ead_growth_rate"]) if "ead_growth_rate" in params else 0.0
        _ead_default = {
            "baseline": _g_bl,
            "adverse":  float(params.get("ead_growth_adverse",  _g_bl - params.get("ead_growth_std", 0.0))),
            "severe":   float(params.get("ead_growth_severe",   _g_bl - 2.0 * params.get("ead_growth_std", 0.0))),
        }
        self.ead_growth = params.get("ead_growth_by_scenario", _ead_default)

        # Demi-vie (années) de la composante de choc dans la croissance
        # décroissante EAD/exposition (voir _decayed_growth_curve). Au-delà
        # d'un horizon de quelques années, un taux de croissance constant
        # composé (1+g)^t diverge de façon irréaliste (ex. horizon NGFS de
        # 25 ans) — seule la composante liée au stress décroît vers 0 ; la
        # tendance structurelle (CAGR historique) reste inchangée.
        self._growth_half_life = float(params.get("growth_decay_half_life", 5.0))

        self.lgd_dt_floor      = float(params.get("lgd_dt_floor",      0.0))
        self.lgd_dt_stress_add = float(params.get("lgd_dt_stress_add", 0.0))

        self.leverage_exp_factor = float(params.get("leverage_exposure_factor", 1.0))
        _g_lev_bl = float(params.get("leverage_exposure_growth", 0.05))
        # Mécanique levier en stress (BCBS CRE §40, EBA ST §4.3, BIS WP 2019) :
        #   Ratio = Tier1 / TEM   (TEM = exposition totale de levier)
        # En stress, deux effets simultanés compriment le ratio :
        #   1. Numérateur ↓  : le capital Tier1 s'érode par les pertes ECL
        #   2. Dénominateur ↑ : les emprunteurs tirent sur les lignes de crédit
        #      et les engagements hors-bilan se convertissent (BCBS CRE §40).
        #      → l'exposition AUGMENTE plus vite en stress qu'en baseline.
        # L'ordonnancement correct (Baseline ≥ Adverse ≥ Sévère) est assuré
        # par ces deux effets qui jouent dans le même sens.
        _lev_default = {
            "baseline": _g_lev_bl,
            # Adverse : tirages modérés sur lignes (+2 pp vs baseline)
            "adverse":  float(params.get("leverage_exposure_growth_adverse",
                                         _g_lev_bl + 0.02)),
            # Sévère : tirages massifs + conversion d'engagements (+5 pp vs baseline)
            "severe":   float(params.get("leverage_exposure_growth_severe",
                                         _g_lev_bl + 0.05)),
        }
        self.leverage_growth = params.get("leverage_growth_by_scenario", _lev_default)

    # ── ρ(PD) ────────────────────────────────────────────────────────────

    def _rho(self, pd_v: float) -> float:
        return asset_correlation(pd_v, self.portfolio_type, self.sme_turnover)

    # ── Croissance décroissante (long horizon) ────────────────────────────

    @staticmethod
    def _decayed_growth_curve(
        g_base: float, g_shock: float, n_years: int, half_life: float = 5.0,
    ) -> np.ndarray:
        """
        Facteurs de croissance cumulés value(t)/value(0) pour t=1..n_years.

        g_eff(t) = g_base + g_shock × 0.5^(t/half_life)

        g_base (tendance structurelle, ex. CAGR historique) reste constant ;
        g_shock (composante liée au scénario de stress) décroît vers 0 avec
        une demi-vie de `half_life` années. Évite qu'un taux constant composé
        (1+g)^t diverge de façon irréaliste sur un horizon NGFS long (25 ans+)
        — le choc est traité comme transitoire, cohérent avec les récits de
        transition NGFS eux-mêmes (concentrés sur 2020s-2030s puis stables).

        g_shock=0.0 (scénario baseline) reproduit exactement (1+g_base)^t —
        aucun changement de comportement pour la baseline.
        """
        factors = np.empty(n_years, dtype=float)
        cum = 1.0
        for t in range(1, n_years + 1):
            g_t = g_base + g_shock * (0.5 ** (t / half_life))
            cum *= (1.0 + g_t)
            factors[t - 1] = cum
        return factors

    # ── MODIFICATION #2 — Helper générique data-driven ───────────────────

    @staticmethod
    def _historical_ratio(
        df: Optional[pd.DataFrame],
        num_col: str,
        den_col: str,
        min_obs: int = 3,
        fallback: float = 0.0,
        valid_range: tuple = (-np.inf, np.inf),
    ) -> float:
        """
        Calcule la moyenne historique de num/den sur le DataFrame capital,
        en filtrant les aberrations et en exigeant un minimum d'observations.

        Si les données sont insuffisantes → retourne `fallback` (Principe #2).
        """
        if df is None:
            return fallback
        if num_col not in df.columns or den_col not in df.columns:
            return fallback

        num = pd.to_numeric(df[num_col], errors="coerce")
        den = pd.to_numeric(df[den_col], errors="coerce")
        mask = den.abs() > 1e-6  # éviter division par zéro
        ratio = (num[mask] / den[mask]).dropna()

        # Filtrage des aberrations
        lo, hi = valid_range
        ratio = ratio[(ratio >= lo) & (ratio <= hi)]

        if len(ratio) < min_obs:
            LOG.warning(
                "_historical_ratio(%s/%s): %d obs < min_obs=%d → fallback=%.4f",
                num_col, den_col, len(ratio), min_obs, fallback,
            )
            return fallback

        result = float(ratio.mean())
        LOG.info(
            "_historical_ratio(%s/%s) = %.4f (n=%d)",
            num_col, den_col, result, len(ratio),
        )
        return result

    # ── MODIFICATION #3 — Sévérité calibrée data-driven ──────────────────

    def _calibrate_severity(
        self, pd_arrays: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Calibre les sévérités adverse/severe à partir des PD projetées.

        severity_adverse = (PD_adv_mean − PD_bl_mean) / (PD_sev_mean − PD_bl_mean)
        severity_severe  = 1.0 (par construction)

        Remplace le hard-codé {0.0, 0.5, 1.0} par une calibration endogène.
        """
        EPS = 1e-10
        try:
            pd_bl  = float(np.mean(pd_arrays["baseline"]))
            pd_adv = float(np.mean(pd_arrays["adverse"]))
            pd_sev = float(np.mean(pd_arrays["severe"]))
        except (KeyError, TypeError):
            LOG.warning("_calibrate_severity: arrays manquants → fallback.")
            return self._SEVERITY_FALLBACK.copy()

        spread_max = pd_sev - pd_bl
        if spread_max <= EPS:
            LOG.warning(
                "_calibrate_severity: spread_max=%.6f ≤ EPS — PD projections are "
                "flat (PD_bl=%.4f, PD_sev=%.4f). Falling back to default severity "
                "{0.0, 0.5, 1.0}. CONSEQUENCE: capital stress will use fixed "
                "severity levels, not data-driven ones.",
                spread_max, pd_bl, pd_sev,
            )
            return self._SEVERITY_FALLBACK.copy()

        sev_adverse = float(np.clip((pd_adv - pd_bl) / spread_max, 0.0, 1.0))

        LOG.info(
            "Sévérité calibrée: baseline=0.0, adverse=%.3f, severe=1.0 "
            "(PD_bl=%.4f, PD_adv=%.4f, PD_sev=%.4f)",
            sev_adverse, pd_bl, pd_adv, pd_sev,
        )
        return {"baseline": 0.0, "adverse": sev_adverse, "severe": 1.0}

    # ── BLOC A : historique ──────────────────────────────────────────────

    def compute_historical(
        self, pd_hist: pd.Series, lgd_hist: np.ndarray,
        ead: float, capital_df: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """Ratios historiques observés (depuis le CSV) + métriques ASRF."""
        years   = sorted(pd_hist.index.tolist())
        pd_vals = pd_hist.sort_index().values.astype(float)
        rows: List[Dict] = []

        for i, yr in enumerate(years):
            pd_v   = float(np.clip(pd_vals[i], PD_FLOOR, PD_CAP))
            lgd_v  = float(np.clip(lgd_hist[i], LGD_FLOOR, LGD_CAP))
            rho    = self._rho(pd_v)
            lgd_dt = lgd_downturn(lgd_v, 0.0, self.lgd_dt_floor, self.lgd_dt_stress_add)
            res    = capital_requirement(pd_v, lgd_dt, rho, self.alpha, self.M)

            tier1 = total_cap = rwa_report = expo_lev = cet1_val = np.nan
            car_real = t1_ratio = lev_ratio = cet1_ratio = np.nan

            if capital_df is not None and yr in capital_df.index:
                r = capital_df.loc[yr]
                for col, var in [("tier1", "tier1"), ("total_capital", "total_cap"),
                                 ("rwa", "rwa_report"), ("leverage_exposure", "expo_lev"),
                                 ("cet1", "cet1_val")]:
                    if col in capital_df.columns:
                        locals()[var]  # just reference — assignment below
                tier1      = r.get("tier1",             np.nan) if "tier1"             in capital_df.columns else np.nan
                total_cap  = r.get("total_capital",     np.nan) if "total_capital"     in capital_df.columns else np.nan
                rwa_report = r.get("rwa",               np.nan) if "rwa"               in capital_df.columns else np.nan
                expo_lev   = r.get("leverage_exposure", np.nan) if "leverage_exposure" in capital_df.columns else np.nan
                cet1_val   = r.get("cet1",              np.nan) if "cet1"              in capital_df.columns else np.nan

                tc_adj = total_cap * self.portfolio_share if not np.isnan(total_cap) else np.nan
                if not np.isnan(rwa_report) and rwa_report > 0:
                    car_real   = tc_adj      / rwa_report if not np.isnan(tc_adj)   else np.nan
                    t1_ratio   = tier1       / rwa_report if not np.isnan(tier1)    else np.nan
                    cet1_ratio = cet1_val    / rwa_report if not np.isnan(cet1_val) else np.nan
                if not np.isnan(tier1) and not np.isnan(expo_lev) and expo_lev > 0:
                    lev_ratio = tier1 / expo_lev

            rows.append({
                "year": yr, "pd": pd_v, "lgd": lgd_v, "lgd_dt": lgd_dt, "rho": rho,
                "var_99_9": res["VaR"], "k_required": res["K"],
                "rwa_asrf": res["RWA_ratio"] * ead,
                "rwa_report": rwa_report, "tier1": tier1, "total_capital": total_cap,
                "car": car_real, "tier1_ratio": t1_ratio, "cet1_ratio": cet1_ratio,
                "leverage_ratio": lev_ratio,
                "cushion_car": (car_real - self.min_car) if not np.isnan(car_real) else np.nan,
            })

        return pd.DataFrame(rows) if rows else None

    # ── BLOC B : stress (Approche B+ — NI stressé + cascade Bâle III) ──

    def compute_stress(
        self, pd_arrays: Dict[str, np.ndarray], lgd_arrays: Dict[str, np.ndarray],
        ead: float, horizon: List[int], capital_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Projection ASRF + capital sous stress — v4 data-driven.

        Améliorations v4 vs v3 :
        ─────────────────────────
        1. NI stressé (PPNR − ΔEL × (1−τ)) au lieu de NI = 0
        2. Sévérité calibrée dynamiquement depuis les PD projetées
        3. Cascade capital : pertes absorbent CET1 → AT1 → T2
        4. RWA total = RWA_crédit + RWA_marché + RWA_opérationnel
        5. PPNR_rate et tax_rate calculés historiquement (data-driven)

        Compatibilité v3 :
        ───────────────────
        Si les colonnes P&L manquent → NI = 0 (fallback identique à v3).
        Si CET1/AT1/T2 absents → érosion uniforme comme v3.
        Si RWA_market/operational absents → RWA = RWA_crédit seul.
        """
        # ── Lire les bases ───────────────────────────────────────────────
        rwa_base       = self._get_base("rwa",               capital_df, 0.0)
        tier1_base     = self._get_base("tier1",             capital_df, 0.0)
        total_cap_base = self._get_base("total_capital",     capital_df, 0.0)
        expo_base      = self._get_base("leverage_exposure", capital_df, 0.0)

        # ── Nouvelles bases (Modification #1 — fallback = 0) ─────────
        cet1_base       = self._get_base("cet1",             capital_df, 0.0)
        at1_base        = self._get_base("at1",              capital_df, 0.0)
        tier2_base      = self._get_base("tier2",            capital_df, 0.0)
        rwa_market_base = self._get_base("rwa_market",       capital_df, 0.0)
        rwa_oper_base   = self._get_base("rwa_operational",  capital_df, 0.0)

        if expo_base <= 0:
            expo_base = ead * self.leverage_exp_factor
            LOG.warning(
                "leverage_exposure absent du CSV — fallback TEM ≈ EAD × %.2f.",
                self.leverage_exp_factor
            )

        has_capital = tier1_base > 0 or total_cap_base > 0

        # ── Cascade Bâle III : déterminer la source de CET1 ──────────
        # Priorité 1 : colonne CET1 explicite dans le CSV
        # Priorité 2 : dériver CET1 = Tier1 − AT1 (si AT1 est disponible)
        # Sinon      : CET1 non calculable → cet1_ratio = NaN
        has_cet1_direct  = cet1_base > 0
        can_derive_cet1  = (not has_cet1_direct) and (at1_base > 0) and (tier1_base > 0)

        if can_derive_cet1:
            cet1_base = max(tier1_base - at1_base, 0.0)
            LOG.info(
                "CET1 dérivé (Tier1 − AT1): CET1=%.0f, AT1=%.0f",
                cet1_base, at1_base,
            )

        has_cascade = has_cet1_direct or can_derive_cet1

        if has_cet1_direct:
            cet1_source = "direct"
        elif can_derive_cet1:
            cet1_source = "derived_tier1_minus_at1"
        else:
            cet1_source = "unavailable"

        if has_cascade and at1_base == 0 and tier2_base == 0:
            # Inférer AT1 et T2 par différence (CET1 direct sans AT1/T2 explicites)
            at1_base  = max(tier1_base - cet1_base, 0.0)
            tier2_base = max(total_cap_base - tier1_base, 0.0)
            LOG.info(
                "Cascade inférée: CET1=%.0f, AT1=%.0f, T2=%.0f",
                cet1_base, at1_base, tier2_base,
            )
        elif not has_cascade and has_capital:
            # Mode v3 : pas de CET1 calculable — érosion uniforme Tier1/Total Capital
            at1_base   = 0.0
            tier2_base = max(total_cap_base - tier1_base, 0.0)
            LOG.info(
                "CET1 non disponible (ni colonne CET1 ni AT1) — mode v3 érosion uniforme."
            )

        # ── Ratios data-driven (Modification #2) ─────────────────────
        ppnr_rate = self._historical_ratio(
            capital_df, "ppnr", "rwa",
            min_obs=3, fallback=0.0,
            valid_range=(-0.05, 0.10),
        )
        tax_rate = self._historical_ratio(
            capital_df, "tax_paid", "pretax_income",
            min_obs=3, fallback=0.0,
            valid_range=(0.0, 0.60),
        )

        has_ni = ppnr_rate > 0  # NI actif seulement si PPNR disponible
        if has_ni:
            LOG.info(
                "Mode NI actif: ppnr_rate=%.4f, tax_rate=%.4f",
                ppnr_rate, tax_rate,
            )
        else:
            LOG.info(
                "Mode NI = 0 (v3 compat): ppnr ou rwa absent du CSV historique."
            )

        # ── Sévérité calibrée (Modification #3) ─────────────────────
        severity_map = self._calibrate_severity(pd_arrays)

        # ── PPNR atténuation par scénario (data-driven) ──────────────
        # baseline: 100% du PPNR_rate, adverse/severe atténuées
        # proportionnellement à la sévérité
        _PPNR_FLOOR = 0.3  # plancher : PPNR ne tombe pas sous 30% du baseline
        ppnr_attenuation = {
            "baseline": 1.0,
            "adverse":  max(1.0 - 0.5 * severity_map["adverse"], _PPNR_FLOOR),
            "severe":   max(1.0 - 0.5 * severity_map["severe"],  _PPNR_FLOOR),
        }

        # ── K baseline pour le ratio delta ASRF ──────────────────────
        n_years = len(horizon)
        _ead_bl_curve = self._decayed_growth_curve(
            self.ead_growth["baseline"], 0.0, n_years, self._growth_half_life
        )
        k_bl_by_yr: Dict[int, float] = {}
        ecl_bl_by_yr: Dict[int, float] = {}
        for i, yr in enumerate(horizon):
            pd_b  = float(np.clip(pd_arrays["baseline"][i], PD_FLOOR, PD_CAP))
            lgd_b = float(np.clip(lgd_arrays["baseline"][i], LGD_FLOOR, LGD_CAP))
            lgd_dt_b = lgd_downturn(lgd_b, 0.0, self.lgd_dt_floor, self.lgd_dt_stress_add)
            k_bl_by_yr[yr] = capital_requirement(
                pd_b, lgd_dt_b, self._rho(pd_b), self.alpha, self.M
            )["K"]
            # EAD baseline pour ΔEL
            ead_bl_t = ead * _ead_bl_curve[i]
            ecl_bl_by_yr[yr] = pd_b * lgd_dt_b * ead_bl_t * self.portfolio_share

        rows: List[Dict] = []

        for level in ("baseline", "adverse", "severe"):
            severity = severity_map[level]
            # Décomposition tendance structurelle (constante) + choc de stress
            # (décroît vers 0, voir _decayed_growth_curve) — g_shock=0 pour
            # "baseline" reproduit exactement l'ancien (1+g)^t, sans changement.
            _ead_curve = self._decayed_growth_curve(
                self.ead_growth["baseline"],
                self.ead_growth[level] - self.ead_growth["baseline"],
                n_years, self._growth_half_life,
            )
            # Garde-fou : l'exposition de levier ne peut jamais se contracter
            # (g_lev < 0 inverse la contrainte et fait monter le ratio en stress).
            _lev_base = max(self.leverage_growth["baseline"], 0.0)
            _lev_curve = self._decayed_growth_curve(
                _lev_base,
                self.leverage_growth[level] - self.leverage_growth["baseline"],
                n_years, self._growth_half_life,
            )

            # ── État capital cumulatif ───────────────────────────────
            cet1_proj      = cet1_base
            at1_proj       = at1_base
            tier2_proj     = tier2_base
            ecl_cumul      = 0.0
            ni_cumul       = 0.0

            for i, yr in enumerate(horizon):
                pd_v  = float(np.clip(pd_arrays[level][i], PD_FLOOR, PD_CAP))
                lgd_v = float(np.clip(lgd_arrays[level][i], LGD_FLOOR, LGD_CAP))
                lgd_dt = lgd_downturn(lgd_v, severity,
                                      self.lgd_dt_floor, self.lgd_dt_stress_add)
                rho   = self._rho(pd_v)
                res   = capital_requirement(pd_v, lgd_dt, rho, self.alpha, self.M)
                k_sc  = res["K"]

                # EAD co-cyclique
                ead_t = ead * _ead_curve[i]

                # ── RWA stressé — delta ASRF + market + operational ──
                k_bl = k_bl_by_yr.get(yr, k_sc)
                delta_k = (k_sc - k_bl) / k_bl if k_bl > 0 else 0.0

                # Ramp-in du choc RWA sur l'horizon (EBA §4.2 / BIS WP).
                # Fondement théorique : seule la fraction du portefeuille qui a
                # «roulé» (maturé et renouvelé) supporte les nouvelles conditions
                # de crédit.  La vitesse de roulement suit une loi exponentielle
                # de paramètre 1/M (maturité moyenne du portefeuille, BCBS CRE32).
                #   ramp_t = 1 − e^{−t/M}   avec t ∈ {1, 2, …, n_years}
                # Pour M = 2.5 ans : ramp = 33 % an 1, 55 % an 2, 70 % an 3.
                _t = i + 1  # année 1, 2, …
                ramp = float(1.0 - np.exp(-_t / max(self.M, 1.0)))

                if rwa_base > 0:
                    rwa_credit = max(
                        rwa_base * (ead_t / ead) * (1.0 + delta_k * ramp), 1.0
                    )
                else:
                    rwa_credit = max(res["RWA_ratio"] * ead_t, 1.0)

                # RWA total = crédit + marché + opérationnel
                # (marché et opérationnel constants en stress — hypothèse
                # standard EBA ; lus du CSV si dispo, sinon 0)
                rwa_stressed = rwa_credit + rwa_market_base + rwa_oper_base

                # ── ECL et ΔEL ───────────────────────────────────────
                ecl_t = pd_v * lgd_dt * ead_t * self.portfolio_share
                ecl_bl_t = ecl_bl_by_yr.get(yr, ecl_t)
                delta_el_t = max(ecl_t - ecl_bl_t, 0.0)
                ecl_cumul += ecl_t

                # ── Net Income stressé ────────────────────────────
                # Cascade + PPNR  : NI = (PPNR - delta_el) * (1-tax)
                # Cascade no PPNR : NI = -delta_el_t  (EL erode CET1 directement)
                # v3 + PPNR       : NI = (PPNR - ecl) * (1-tax)
                # v3 no PPNR      : NI = 0  (erosion via ecl_cumul bloc capital)
                if has_ni:
                    ppnr_t = ppnr_rate * rwa_stressed * ppnr_attenuation[level]
                    if has_cascade:
                        ni_t = (ppnr_t - delta_el_t) * (1.0 - tax_rate)
                    else:
                        ni_t = (ppnr_t - ecl_t) * (1.0 - tax_rate)
                else:
                    ppnr_t = 0.0
                    if has_cascade:
                        # Sans PPNR : delta EL reduit CET1 directement (pas de buffer revenus)
                        ni_t = -delta_el_t
                    else:
                        ni_t = 0.0  # erosion via ecl_cumul ci-dessous

                ni_cumul += ni_t

                # ── Cascade Bâle III : absorption des pertes ─────────
                if has_capital:
                    if has_cascade:
                        # NI positif renforce CET1, NI négatif l'érode
                        cet1_proj += ni_t
                        if cet1_proj < 0:
                            at1_proj += cet1_proj
                            cet1_proj = 0.0
                        if at1_proj < 0:
                            tier2_proj += at1_proj
                            at1_proj = 0.0
                        if tier2_proj < 0:
                            tier2_proj = 0.0

                        tier1_proj     = cet1_proj + at1_proj
                        total_cap_proj = tier1_proj + tier2_proj
                    else:
                        if has_ni:
                            # NI inclut déjà ECL complète → capital = base + NI cumulé
                            tier1_proj     = max(tier1_base + ni_cumul, 0.0)
                            total_cap_proj = max(total_cap_base + ni_cumul, 0.0)
                        else:
                            # Bilan statique EBA : pertes nettes sur capital
                            tier1_proj     = max(tier1_base - ecl_cumul, 0.0)
                            total_cap_proj = max(total_cap_base - ecl_cumul, 0.0)
                else:
                    tier1_proj     = 0.0
                    total_cap_proj = 0.0

                # Exposition levier projetée
                expo_t = expo_base * _lev_curve[i]

                # Ratios
                car       = total_cap_proj / rwa_stressed if (has_capital and rwa_stressed > 0) else np.nan
                t1_ratio  = tier1_proj     / rwa_stressed if (has_capital and rwa_stressed > 0) else np.nan
                cet1_ratio = cet1_proj     / rwa_stressed if (has_capital and has_cascade and rwa_stressed > 0) else np.nan
                lev_ratio = tier1_proj     / expo_t       if (has_capital and expo_t > 0)       else np.nan

                car_flag = (
                    "OK"      if (not np.isnan(car) and car >= self.min_car)   else
                    "WARNING" if (not np.isnan(car) and car >= self.min_tier1) else
                    "BREACH"
                )
                lev_flag = "OK" if (not np.isnan(lev_ratio) and lev_ratio >= self.min_leverage) else "BREACH"

                rows.append({
                    "scenario":       level,
                    "year":           yr,
                    "pd":             pd_v,
                    "lgd":            lgd_v,
                    "lgd_dt":         lgd_dt,
                    "rho":            rho,
                    "var_99_9":       res["VaR"],
                    "k_required":     k_sc,
                    "k_abs":          k_sc * ead_t,
                    "ead":            ead_t,
                    "delta_k_ratio":  delta_k,
                    "rwa_credit":     rwa_credit,
                    "rwa_market":     rwa_market_base,
                    "rwa_operational": rwa_oper_base,
                    "rwa_stressed":   rwa_stressed,
                    "ecl":            ecl_t,
                    "delta_el":       delta_el_t,
                    "ecl_cumul":      ecl_cumul,
                    "ppnr":           ppnr_t,
                    "net_income":     ni_t,
                    "ni_cumul":       ni_cumul,
                    "cet1_proj":      cet1_proj   if has_cascade else np.nan,
                    "at1_proj":       at1_proj    if has_cascade else np.nan,
                    "tier2_proj":     tier2_proj  if has_cascade else np.nan,
                    "tier1_proj":     tier1_proj,
                    "total_cap_proj": total_cap_proj,
                    "car":            car,
                    "tier1_ratio":    t1_ratio,
                    "cet1_ratio":     cet1_ratio  if has_cascade else np.nan,
                    "leverage_ratio": lev_ratio,
                    "cushion_car":    (car - self.min_car) if not np.isnan(car) else np.nan,
                    "deficit":        max(self.min_car * rwa_stressed - total_cap_proj, 0.0)
                                      if (has_capital and rwa_stressed > 0) else np.nan,
                    "car_flag":       car_flag,
                    "leverage_flag":  lev_flag,
                    "severity":       severity,
                    "ppnr_rate_used": ppnr_rate,
                    "tax_rate_used":  tax_rate,
                })

        return pd.DataFrame(rows)

    # ── Helper ──────────────────────────────────────────────────────────

    def _get_base(self, col: str, capital_df: Optional[pd.DataFrame],
                  default: float) -> float:
        if capital_df is not None and col in capital_df.columns:
            valid = capital_df[col].dropna()
            if not valid.empty:
                return float(valid.iloc[-1])
        return default

    # ── Run ──────────────────────────────────────────────────────────────

    def run(
        self, pd_hist: pd.Series, lgd_hist: np.ndarray,
        pd_stress: Dict[str, np.ndarray], lgd_stress: Dict[str, np.ndarray],
        ead: float, horizon: List[int], capital_df: Optional[pd.DataFrame],
    ) -> CapitalOutput:
        """Exécution complète : historique + stress + KPIs + trajectoires."""
        hist_df   = self.compute_historical(pd_hist, lgd_hist, ead, capital_df)
        stress_df = self.compute_stress(pd_stress, lgd_stress, ead, horizon, capital_df)

        # KPIs
        kpis: Dict[str, Any] = {}
        for level in ("baseline", "adverse", "severe"):
            sub = stress_df[stress_df["scenario"] == level]
            if sub.empty:
                continue
            kpis[f"{level}_var_peak"]   = round(float(sub["var_99_9"].max()), 6)
            kpis[f"{level}_k_peak"]    = round(float(sub["k_required"].max()), 6)
            kpis[f"{level}_rwa_last"]  = round(float(sub["rwa_stressed"].iloc[-1]), 0)
            kpis[f"{level}_ecl_total"] = round(float(sub["ecl"].sum()), 0)
            if not sub["car"].isna().all():
                kpis[f"{level}_car_last"]  = round(float(sub["car"].iloc[-1]), 6)
                kpis[f"{level}_car_min"]   = round(float(sub["car"].min()), 6)
            if not sub["leverage_ratio"].isna().all():
                kpis[f"{level}_lev_last"]  = round(float(sub["leverage_ratio"].iloc[-1]), 6)
            if not sub["deficit"].isna().all():
                kpis[f"{level}_deficit_max"] = round(float(sub["deficit"].max()), 0)
            # ── KPIs v4 ──────────────────────────────────────────────
            if "net_income" in sub.columns:
                kpis[f"{level}_ni_total"]  = round(float(sub["net_income"].sum()), 0)
            if "ppnr" in sub.columns:
                kpis[f"{level}_ppnr_total"] = round(float(sub["ppnr"].sum()), 0)
            if "delta_el" in sub.columns:
                kpis[f"{level}_delta_el_total"] = round(float(sub["delta_el"].sum()), 0)
            if "cet1_ratio" in sub.columns and not sub["cet1_ratio"].isna().all():
                kpis[f"{level}_cet1_ratio_min"] = round(float(sub["cet1_ratio"].min()), 6)
            if "severity" in sub.columns:
                kpis[f"{level}_severity"] = round(float(sub["severity"].iloc[0]), 4)

        # Trajectoires graphiques
        traj_all: Dict[str, Dict] = {}
        for metric in ("car", "leverage_ratio", "var_99_9", "rwa_stressed",
                       "ecl_cumul", "tier1_ratio", "deficit",
                       # ── Nouvelles métriques v4 ────────────────────────
                       "cet1_ratio", "ppnr", "net_income", "ni_cumul",
                       "delta_el", "cet1_proj", "tier1_proj",
                       "total_cap_proj"):
            traj: Dict[str, Dict] = {}
            hx: List[str] = []
            hy: List[float] = []
            if hist_df is not None and metric in hist_df.columns:
                vh = hist_df.dropna(subset=[metric])
                hx = [str(int(y)) for y in vh["year"].tolist()]
                hy = vh[metric].tolist()
            if hx:
                traj["historical"] = {"x": hx, "y": hy}
            for level in ("baseline", "adverse", "severe"):
                sub = stress_df[stress_df["scenario"] == level]
                sx_all = [str(int(y)) for y in sub["year"].tolist()]
                sy_all = sub[metric].tolist()
                if hx:
                    # Supprimer de la projection toute année déjà présente dans
                    # l'historique (évite le doublon visuel quand le CSV capital
                    # couvre la même année que le début de l'horizon de stress).
                    hist_set = set(hx)
                    pairs = [(x, y) for x, y in zip(sx_all, sy_all)
                             if x not in hist_set]
                    if pairs:
                        sx, sy = map(list, zip(*pairs))
                    else:
                        sx, sy = [], []
                    # Pont visuel : ajouter le dernier point historique en tête
                    # pour que la courbe de projection parte exactement de la
                    # valeur observée (pas d'écart entre gris et pointillé).
                    sx = [hx[-1]] + sx
                    sy = [hy[-1]] + sy
                else:
                    sx, sy = sx_all, sy_all
                traj[level] = {"x": sx, "y": sy}
            traj_all[metric] = traj

        LOG.info("Capital engine v4 complete: %d hist, %d stress, %d KPIs.",
                 len(hist_df) if hist_df is not None else 0,
                 len(stress_df), len(kpis))

        return CapitalOutput(
            historical_df=hist_df, stress_df=stress_df,
            kpis=kpis, capital_trajectories=traj_all,
        )
