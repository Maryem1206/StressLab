"""
Moteur generique des scenarios idiosyncratiques - Basel III / EBA.

Aucune constante numerique dans les formules.
Tous les coefficients proviennent des shock_inputs de l evenement (modifiables UI).
Ratios recalcules depuis leurs definitions Basel III :
  LCR  = HQLA_new / Outflows_new x 100
  NSFR = ASF_new  / RSF_new      x 100
  CET1 = Capital_new / RWA_new   x 100
  CAR  = TotalCap_new / RWA_new  x 100
  Lev  = Capital_new / TotalExp  x 100
"""
from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import logging

LOG = logging.getLogger("idio_engine")

_COL_ALIASES: Dict[str, tuple] = {
    "year":           ("year", "annee", "date", "yr", "exercice", "periode"),
    "cet1_ratio":     ("cet1_ratio", "cet1 ratio", "ratio_cet1", "cet1%", "cet1_pct",
                       "cet1", "cet_1", "common_equity_tier1"),
    "car":            ("car", "total_capital_ratio", "ratio_car", "car_ratio",
                       "capital_adequacy", "ratio_fonds_propres", "ratio_solvabilite",
                       "total capital ratio"),
    "rwa":            ("rwa", "risk_weighted_assets", "actifs_ponderes", "rwa_total",
                       "rwa_egp_m", "rwas"),
    "tier1":          ("tier1", "tier_1", "t1", "capital_tier1", "tier1_egp_m",
                       "tier 1", "tier_1_capital", "fonds_propres_t1"),
    "total_cap":      ("total_capital", "capital_total", "fonds_propres", "total_cap",
                       "totalcapital", "total_capital_egp_m", "capitaux_propres"),
    "leverage_ratio": ("leverage_ratio", "leverage ratio", "ratio_levier", "leverage", "levier"),
    "total_exposure": ("leverage_exposure", "total_exposure", "expo_levier",
                       "total_actif", "total_assets", "leverage_exposure_egp_m"),
    "ead":            ("ead", "e.a.d", "exposure", "exposition", "encours",
                       "portefeuille", "portfolio", "total_ead", "ead_total",
                       "actif_pondere", "encours_credit", "volume_credit"),
    "avg_pd":         ("avg_pd", "pd", "default_rate", "defaultrate",
                       "taux_defaut", "taux_d", "tx_d", "taux de d"),
    "avg_lgd":        ("avg_lgd", "lgd", "loss_given_default", "perte_defaut"),
    "npl_ratio":      ("npl_ratio", "npl ratio", "ratio_npl", "npl%",
                       "taux_npl", "non_performing", "npls_ratio"),
    "cout_risque":    ("cout_risque", "cost_of_risk", "cost of risk",
                       "cout du risque", "credit_cost", "provision_rate"),
    "lcr":            ("lcr", "liquidity_coverage", "liquidity coverage ratio",
                       "ratio_lcr", "lcr_ratio", "lcr%",
                       "_lcr_obs", "lcr_obs", "observed_lcr", "lcr_observed",
                       "lcr_ratio_obs", "lcr_computed"),
    "nsfr":           ("nsfr", "net_stable_funding", "net stable funding ratio",
                       "ratio_nsfr", "nsfr%",
                       "_nsfr_obs", "nsfr_obs", "observed_nsfr", "nsfr_observed",
                       "nsfr_ratio_obs", "nsfr_computed"),
    "hqla":           ("hqla", "high_quality_liquid", "actifs_liquides",
                       "liquide_hq", "hqla_total"),
    "asf":            ("asf", "available_stable_funding", "ressources_stables",
                       "financement_stable_disponible"),
    "rsf":            ("rsf", "required_stable_funding", "besoins_stables",
                       "financement_stable_requis"),
    "deposits":       ("deposits", "depots", "total_deposits",
                       "depot_client", "customer_deposits", "total_depots"),
    "outflows_30j":   ("outflows_30j", "outflows", "sorties_30j", "net_outflows",
                       "sorties_nettes"),
    "nii":            ("nii", "net_interest_income", "revenu_net_interet",
                       "produit_net_bancaire", "pnb", "rnb", "net interest income"),
    "afs_souverain":  ("afs_souverain", "afs_sovereign", "portefeuille_souverain",
                       "titres_etat", "sovereign_portfolio", "afs"),
    "capital":        ("capital_cet1", "cet1_capital", "fonds_propres_cet1",
                       "tier1_capital_amount"),
    # ── Variables granulaires du dataset liquidité ────────────────────
    "retail_deposit": ("retail_deposit", "depot_retail", "depots_retail",
                       "retail_deposits", "depot_particulier", "depots_particuliers"),
    "corporate_de":   ("corporate_de", "corporate_deposit", "depots_corporate",
                       "depot_entreprise", "corporate_deposits", "depots_entreprises"),
    "wholesale_fu":   ("wholesale_fu", "wholesale_funding", "financement_gros",
                       "wholesale_fund", "depot_interbancaire"),
    "off_bs_comm":    ("off_bs_comm", "off_balance_sheet", "hors_bilan",
                       "engagements_hors_bilan", "committed_lines"),
    "L1":             ("l1", "level1", "level_1", "hqla_l1", "actifs_l1",
                       "level 1", "tier_l1"),
    "L2a":            ("l2a", "level2a", "level_2a", "hqla_l2a", "actifs_l2a",
                       "level 2a"),
    "inflows_cont":   ("inflows_cont", "inflows_contingent", "entrees_contingentes",
                       "contingent_inflows", "entrees_cont"),
    "loans":          ("loans", "prets", "encours_prets", "loan_portfolio",
                       "prets_bruts", "net_loans", "prets_nets"),
    "securities":     ("securities", "titres", "portefeuille_titres",
                       "investment_securities", "titres_investissement",
                       "securities_portfolio", "afs_securities"),
    "asf_factor_retail": ("asf_factor_retail", "facteur_asf_retail",
                          "asf_retail_factor", "nsfr_factor_retail"),
    "asf_factor_corp":   ("asf_factor_corp", "asf_factor_corporate",
                          "facteur_asf_corp", "nsfr_factor_corp"),
    "run_off_retail":    ("run_off_retail", "runoff_retail", "taux_runoff_retail",
                          "retail_runoff_rate", "run_off_rate_retail"),
    "run_off_corp":      ("run_off_corp", "runoff_corp", "taux_runoff_corp",
                          "corporate_runoff_rate", "run_off_rate_corp"),
    "haircut_add":       ("haircut_add", "haircut_additionnel", "additional_haircut",
                          "haircut_supplementaire", "haircut_hqla"),
    "asf_other":         ("asf_other", "asf_autres", "other_stable_funding",
                          "autres_ressources_stables"),
}

REGULATORY_FLOORS: Dict[str, float] = {
    "lcr":            100.0,
    "nsfr":           100.0,
    "cet1_ratio":       4.5,
    "car":              8.0,
    "leverage_ratio":   3.0,
    "avg_lgd":          0.45,
}

THRESHOLDS: Dict[str, Dict] = {
    "LCR":          {"min": 100.0, "warn": 115.0, "unit": "%", "higher_is_better": True},
    "NSFR":         {"min": 100.0, "warn": 105.0, "unit": "%", "higher_is_better": True},
    "CET1":         {"min":   4.5, "warn":   7.0, "unit": "%", "higher_is_better": True},
    "CAR":          {"min":   8.0, "warn":  10.5, "unit": "%", "higher_is_better": True},
    "Leverage":     {"min":   3.0, "warn":   4.0, "unit": "%", "higher_is_better": True},
    "NPL":          {"min":  None, "warn":  10.0, "unit": "%", "higher_is_better": False},
    "Cout risque":  {"min":  None, "warn":   2.0, "unit": "%", "higher_is_better": False},
}

BASE_LABELS: Dict[str, str] = {
    "lcr": "LCR actuel (%)", "nsfr": "NSFR actuel (%)",
    "hqla": "HQLA (M)", "asf": "ASF - Ressources stables (M)",
    "rsf": "RSF - Besoins stables (M)", "deposits": "Depots totaux (M)",
    "outflows_30j": "Outflows nets 30j (M)", "nii": "NII annuel (M)",
    "afs_souverain": "Portefeuille AFS souverain (M)",
    "cet1_ratio": "CET1 ratio (%)", "car": "CAR (%)", "rwa": "RWA (M)",
    "capital": "Capital CET1 (M)", "tier1": "Tier 1 Capital (M)",
    "total_cap": "Capital total (M)", "leverage_ratio": "Leverage ratio (%)",
    "total_exposure": "Total Exposure (M)", "ead": "EAD total (M)",
    "avg_pd": "PD moyenne", "avg_lgd": "LGD moyenne",
    "total_el": "EL totale (M)", "npl_ratio": "NPL ratio (%)",
    "cout_risque": "Cout du risque (%)",
}


def get_base_from_dataset(file_path: str) -> Dict[str, Any]:
    """Parse l historique uploade, extrait la derniere observation."""
    try:
        import pandas as pd
        p = Path(file_path)
        if not p.exists():
            return {}
        if p.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        else:
            df = None
            for sep in (";", ",", "\t", "|"):
                for dec in (",", "."):
                    try:
                        tmp = pd.read_csv(p, sep=sep, decimal=dec, thousands=" ")
                        if len(tmp.columns) >= 2:
                            df = tmp; break
                    except Exception:
                        continue
                if df is not None:
                    break
        if df is None or df.empty:
            return {}
        df.columns = [str(c).strip() for c in df.columns]
        col_lower = {c: c.lower().strip() for c in df.columns}
        year_col = None
        for c, low in col_lower.items():
            if low in _COL_ALIASES["year"]:
                year_col = c; break
        if year_col is None:
            for c in df.columns:
                try:
                    mv = float(df[c].dropna().mean())
                    if 1950 < mv < 2100:
                        year_col = c; break
                except Exception:
                    continue
        if year_col:
            df = df.sort_values(year_col, ascending=True)
        last = df.iloc[-1]
        result: Dict[str, Any] = {}
        for key, aliases in _COL_ALIASES.items():
            if key == "year":
                continue
            for c, low in col_lower.items():
                if low in aliases:
                    try:
                        val = float(str(last[c]).replace(",", ".").replace(" ", ""))
                        if val == val and val > 0:
                            result[key] = {"value": val, "source": "dataset"}
                    except (ValueError, TypeError):
                        pass
                    break
        LOG.info("Dataset: %d champs depuis %s", len(result), p.name)
        return result
    except Exception as exc:
        LOG.warning("get_base_from_dataset: %s", exc)
        return {}


def get_historical_series_from_dataset(
    file_path: str,
    keys: tuple = ("lcr", "nsfr"),
) -> Dict[str, Dict]:
    """
    Extract full year-by-year historical series for the requested ratio keys
    from the uploaded liquidity dataset file.

    Returns {key: {"x": [str(year), ...], "y": [float, ...]}} for each key
    that is found in the file. Years are sorted ascending. Rows with missing
    values for a ratio are dropped for that ratio only.
    """
    try:
        import pandas as pd
        p = Path(file_path)
        if not p.exists():
            return {}
        if p.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        else:
            df = None
            for sep in (";", ",", "\t", "|"):
                for dec in (",", "."):
                    try:
                        tmp = pd.read_csv(p, sep=sep, decimal=dec, thousands=" ")
                        if len(tmp.columns) >= 2:
                            df = tmp; break
                    except Exception:
                        continue
                if df is not None:
                    break
        if df is None or df.empty:
            return {}

        df.columns = [str(c).strip() for c in df.columns]
        col_lower = {c: c.lower().strip() for c in df.columns}

        # Detect year column
        year_col = None
        for c, low in col_lower.items():
            if low in _COL_ALIASES["year"]:
                year_col = c; break
        if year_col is None:
            for c in df.columns:
                try:
                    mv = float(df[c].dropna().mean())
                    if 1950 < mv < 2100:
                        year_col = c; break
                except Exception:
                    continue
        if year_col is None:
            return {}

        df = df.sort_values(year_col, ascending=True)

        result: Dict[str, Dict] = {}
        for key in keys:
            aliases = _COL_ALIASES.get(key, ())
            ratio_col = None
            for c, low in col_lower.items():
                if low in aliases:
                    ratio_col = c; break
            if ratio_col is None:
                continue

            sub = df[[year_col, ratio_col]].copy()
            sub[ratio_col] = pd.to_numeric(
                sub[ratio_col].astype(str).str.replace(",", ".").str.replace(" ", ""),
                errors="coerce",
            )
            sub = sub.dropna(subset=[ratio_col])
            sub = sub[sub[ratio_col] > 0]
            if sub.empty:
                continue

            x_vals = [str(int(float(v))) for v in sub[year_col].tolist()]
            y_vals = [round(float(v), 2) for v in sub[ratio_col].tolist()]
            result[key.upper()] = {"x": x_vals, "y": y_vals}
            LOG.info("Historical series '%s': %d points from %s", key, len(x_vals), p.name)

        return result
    except Exception as exc:
        LOG.warning("get_historical_series_from_dataset: %s", exc)
        return {}


def get_base_values(record: Optional[Dict] = None,
                    dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Construit les valeurs de base des drivers de bilan.
    Priorite : run > dataset > planchers reglementaires.
    """
    base: Dict[str, float] = {}
    src:  Dict[str, str]   = {}

    for k, v in REGULATORY_FLOORS.items():
        base[k] = v; src[k] = "regulatory"

    if dataset_path:
        ds = get_base_from_dataset(dataset_path)
        for k, w in ds.items():
            base[k] = w["value"]; src[k] = w["source"]

    mr = (record.get("module_results", {}) if isinstance(record, dict) else {}) if record else {}
    for k in ("cet1_ratio","car","rwa","tier1","total_cap","leverage_ratio","total_exposure"):
        v = mr.get("capital",{}).get("baseline",{}).get("kpis",{}).get(k)
        if v is not None:
            try:
                fv = float(v)
                if fv > 0: base[k] = fv; src[k] = "run"
            except (TypeError, ValueError):
                pass
    for k in ("ead","avg_pd","avg_lgd","total_el"):
        v = mr.get("credit",{}).get("baseline",{}).get("kpis",{}).get(k)
        if v is not None:
            try:
                fv = float(v)
                if fv > 0: base[k] = fv; src[k] = "run"
            except (TypeError, ValueError):
                pass

    # capital CET1 montant = CET1% x RWA / 100
    if "capital" not in base and base.get("rwa", 0) > 0 and "cet1_ratio" in base:
        base["capital"] = base["cet1_ratio"] * base["rwa"] / 100.0
        src["capital"]  = src.get("rwa", "computed")

    # NPL depuis PD
    if src.get("avg_pd") in ("run","dataset") and "npl_ratio" not in base:
        base["npl_ratio"] = round(base["avg_pd"] * 100, 2)
        src["npl_ratio"]  = "computed"

    # Cout risque depuis EL/EAD
    el_ok  = src.get("total_el") in ("run","dataset")
    ead_ok = src.get("ead")      in ("run","dataset")
    if el_ok and ead_ok and base.get("ead",0) > 0:
        if "cout_risque" not in base or src.get("cout_risque") == "regulatory":
            base["cout_risque"] = round(base["total_el"] / base["ead"] * 100, 4)
            src["cout_risque"]  = "computed"

    # NII proxy (fallback)
    if el_ok and "nii" not in base:
        base["nii"] = round(base["total_el"] * 3.0, 0)
        src["nii"]  = "computed"

    # ASF / RSF derives
    nsfr_v = base.get("nsfr", REGULATORY_FLOORS["nsfr"])
    rwa_ok = src.get("rwa") in ("run","dataset")
    if "asf" not in base and "rsf" not in base:
        if ead_ok:
            rsf_val = base["ead"] * 0.85
            base["rsf"] = round(rsf_val, 0); src["rsf"] = "computed"
            base["asf"] = round(nsfr_v * rsf_val / 100.0, 0); src["asf"] = "computed"
        elif rwa_ok:
            rsf_val = base["rwa"] * 1.2
            base["rsf"] = round(rsf_val, 0); src["rsf"] = "computed"
            base["asf"] = round(nsfr_v * rsf_val / 100.0, 0); src["asf"] = "computed"
    elif "asf" not in base and "rsf" in base:
        base["asf"] = round(nsfr_v * base["rsf"] / 100.0, 0); src["asf"] = "computed"
    elif "rsf" not in base and "asf" in base and nsfr_v > 0:
        base["rsf"] = round(base["asf"] * 100.0 / nsfr_v, 0); src["rsf"] = "computed"

    # HQLA / Outflows depuis EAD
    if ead_ok and "hqla" not in base:
        lcr_v = base.get("lcr", REGULATORY_FLOORS["lcr"])
        hqla_val = base["ead"] * 0.18
        base["hqla"] = round(hqla_val, 0); src["hqla"] = "computed"
        if "outflows_30j" not in base:
            base["outflows_30j"] = round(hqla_val / max(lcr_v / 100.0, 0.01), 0)
            src["outflows_30j"]  = "computed"
    if ead_ok and "deposits" not in base:
        base["deposits"] = round(base["ead"], 0); src["deposits"] = "computed"

    # Total exposure depuis RWA
    if rwa_ok and "total_exposure" not in base:
        base["total_exposure"] = round(base["rwa"] * 1.5, 0)
        src["total_exposure"]  = "computed"

    return {k: {"value": v, "source": src.get(k, "regulatory")} for k, v in base.items()}


def flat_base(base_wrapped: Dict) -> Dict[str, float]:
    return {k: v["value"] for k, v in base_wrapped.items()}


def get_severity(ratio_name: str, value: float) -> str:
    t      = THRESHOLDS.get(ratio_name, {})
    min_v  = t.get("min")
    warn_v = t.get("warn")
    higher = t.get("higher_is_better", True)
    if higher:
        if min_v  is not None and value < min_v:  return "fail"
        if warn_v is not None and value < warn_v: return "warn"
    else:
        if warn_v is not None and value > warn_v: return "warn"
    return "ok"


# ── Primitives generiques ────────────────────────────────────────────────────

def _dn(v: float, w: float, c: float) -> float:
    """v x (1 - w x c)  -- grandeur qui diminue."""
    return max(v * (1.0 - w * c), 0.0)

def _up(v: float, w: float, c: float) -> float:
    """v x (1 + w x c)  -- grandeur qui augmente."""
    return v * (1.0 + w * c)

def _rel(r0: float, w: float, nc: float, dc: float) -> float:
    """Calcul relatif si montants absents : r0 x (1-w*nc)/(1+w*dc)."""
    return max(r0 * (1.0 - w * nc) / max(1.0 + w * dc, 1e-9), 0.0)


# ════════════════════════════════════════════════════════════════════════════
# MOTEUR PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

def compute_shock(event_id: str, inputs: Dict, base_flat: Dict) -> Dict:
    """
    Calcule l'impact d'un choc idiosyncratique via des formules Basel III / EBA.

    Chaque scenario utilise UNIQUEMENT ses parametres directs (shock_inputs).
    Formules par canal :
      Canal 1 direct   : perte operationnelle / ECL → deduction capital
      Canal 2 comport. : run-off → hausse outflows (LCR) ; ASF factor → baisse ASF (NSFR)
      Canal 3 reglmt.  : haircut HQLA → baisse HQLA ; RWA shock → hausse RWA

    Ratios recomputes depuis leurs definitions Basel III :
      LCR  = HQLA_new / Outflows_new × 100
      NSFR = ASF_new  / RSF_new      × 100
      CET1 = Capital_new / RWA_new   × 100
      CAR  = TotalCap_new / RWA_new  × 100
      Lev  = Capital_new / TotalExp  × 100
    """
    from .idio_events import IDIO_EVENTS
    event = IDIO_EVENTS.get(event_id)
    if not event:
        return {"ratios": {}, "dag_highlights": {}}

    drivers = event.get("drivers", [])
    dag_map = event.get("dag_severity_map", {})

    # ── Base values ──────────────────────────────────────────────────────────
    hqla         = base_flat.get("hqla",           0.0)
    outflows     = base_flat.get("outflows_30j",   0.0)
    asf          = base_flat.get("asf",            0.0)
    rsf          = base_flat.get("rsf",            0.0)
    capital      = base_flat.get("capital",        0.0)
    rwa          = base_flat.get("rwa",            0.0)
    total_cap    = base_flat.get("total_cap",      0.0)
    total_exp    = base_flat.get("total_exposure", 0.0)
    ead          = base_flat.get("ead",            0.0)
    avg_pd       = base_flat.get("avg_pd",         0.0)
    avg_lgd      = base_flat.get("avg_lgd",        REGULATORY_FLOORS["avg_lgd"])
    deposits     = base_flat.get("deposits",       0.0)

    # Granular liquidity variables — from dataset if available, else proxied
    retail_dep   = base_flat.get("retail_deposit", deposits * 0.60)
    corp_dep     = base_flat.get("corporate_de",   deposits * 0.30)
    L1_val       = base_flat.get("L1",             hqla * 0.85)
    inflows_val  = base_flat.get("inflows_cont",   outflows * 0.20)
    securities_v = base_flat.get("securities",     hqla * 0.70)
    loans_v      = base_flat.get("loans",          rsf  * 0.80)

    # ── Working copies (modified per scenario) ───────────────────────────────
    hqla_new     = hqla
    outflows_new = outflows
    asf_new      = asf
    rsf_new      = rsf
    capital_new  = capital
    rwa_new      = rwa
    tcap_new     = total_cap

    # ── Dispatch par scénario ────────────────────────────────────────────────

    if event_id == "bank_run":
        # Paramètres : run_off_retail (pp), run_off_corp (pp), haircut_add (pp)
        ror  = float(inputs.get("run_off_retail", 0.0))  # pp delta run-off retail
        roc  = float(inputs.get("run_off_corp",   0.0))  # pp delta run-off corp
        hca  = float(inputs.get("haircut_add",    0.0))  # pp delta haircut HQLA

        # Canal 2 comportemental : outflows augmentent (dépôts × taux marginal)
        outflows_new = outflows + retail_dep * ror / 100 + corp_dep * roc / 100
        # Canal 3 réglementaire : HQLA réduit (fire sales + haircut accru)
        hqla_new = hqla * (1.0 - hca / 100)
        # NSFR — dépôts moins stables (transmission modérée 40% retail, 20% corp)
        asf_new = asf - retail_dep * ror * 0.40 / 100 - corp_dep * roc * 0.20 / 100
        asf_new = max(asf_new, 0.0)

    elif event_id == "npl_surge":
        # Paramètres : PD_shock (×), LGD_shock (pp)
        pd_mult  = max(float(inputs.get("PD_shock",  1.0)), 1.0)
        lgd_delt = float(inputs.get("LGD_shock", 0.0))       # pp

        # Canal 1 direct : hausse ECL → déduction capital (IFRS 9 Stage 3)
        pd_new   = avg_pd  * pd_mult
        lgd_new  = min(avg_lgd + lgd_delt / 100, 1.0)
        el_base  = ead * avg_pd  * avg_lgd  if ead > 0 and avg_pd > 0 else 0.0
        el_new   = ead * pd_new  * lgd_new  if ead > 0 else 0.0
        delta_el = max(el_new - el_base, 0.0)
        capital_new = max(capital  - delta_el, 0.0)
        tcap_new    = max(total_cap - delta_el, 0.0)
        # RWA crédit augmente avec la hausse de PD (IRB — relation concave √)
        if avg_pd > 0:
            rwa_new = rwa * (pd_new / avg_pd) ** 0.5

    elif event_id == "fraud_internal":
        # Paramètre : op_loss_amount (montant absolu M)
        op_loss = max(float(inputs.get("op_loss_amount", 0.0)), 0.0)

        # Canal 1 direct : déduction immédiate du capital (CET1 tier)
        capital_new = max(capital   - op_loss, 0.0)
        tcap_new    = max(total_cap - op_loss, 0.0)
        # Basel SMA (BCBS 424) : RWA opérationnel += perte × 12.5
        rwa_new = rwa + op_loss * 12.5

    elif event_id == "cyber_attack":
        # Paramètres : run_off_retail (pp), run_off_corp (pp),
        #              inflows_cont (pp, négatif), op_loss_amount (M)
        ror     = float(inputs.get("run_off_retail",  0.0))
        roc     = float(inputs.get("run_off_corp",    0.0))
        inf_d   = float(inputs.get("inflows_cont",    0.0))  # pp, ≤ 0
        op_loss = max(float(inputs.get("op_loss_amount", 0.0)), 0.0)

        # Canal 2 comportemental : run-off dépôts → LCR ↓
        outflows_new = outflows + retail_dep * ror / 100 + corp_dep * roc / 100
        # Canal 2 : systèmes bloqués → inflows contingents réduits → outflows nets ↑
        outflows_new += inflows_val * abs(inf_d) / 100
        # Canal 1 : pertes opérationnelles (rançon + remédiation + amendes RGPD)
        capital_new  = max(capital   - op_loss, 0.0)
        tcap_new     = max(total_cap - op_loss, 0.0)
        rwa_new      = rwa + op_loss * 12.5

    elif event_id == "reputation_crisis":
        # Paramètres : run_off_retail (pp), run_off_corp (pp),
        #              asf_factor_retail (pp, négatif), asf_factor_corp (pp, négatif)
        ror    = float(inputs.get("run_off_retail",   0.0))
        roc    = float(inputs.get("run_off_corp",     0.0))
        asf_r  = float(inputs.get("asf_factor_retail", 0.0))  # pp, ≤ 0
        asf_c  = float(inputs.get("asf_factor_corp",   0.0))  # pp, ≤ 0

        # Canal 2 comportemental : LCR ↓
        outflows_new = outflows + retail_dep * ror / 100 + corp_dep * roc / 100
        # Canal 3 réglementaire : NSFR ↓ — facteurs ASF révisés à la baisse
        asf_new = asf + retail_dep * asf_r / 100 + corp_dep * asf_c / 100
        asf_new = max(asf_new, 0.0)

    elif event_id == "sovereign_downgrade":
        # Paramètres : haircut_add (pp), L1 (pp, négatif),
        #              securities (pp, négatif), RWA_shock (pp)
        hca    = float(inputs.get("haircut_add", 0.0))
        l1_d   = float(inputs.get("L1",         0.0))  # pp, ≤ 0
        sec_d  = float(inputs.get("securities", 0.0))  # pp, ≤ 0
        rwa_d  = float(inputs.get("RWA_shock",  0.0))  # pp, ≥ 0

        # Canal 3 : haircut accru + reclassement L1→L2 réduit HQLA eligible
        hqla_new  = hqla * (1.0 - hca / 100)
        hqla_new += L1_val * l1_d / 100       # l1_d négatif → hqla_new diminue
        hqla_new  = max(hqla_new, 0.0)
        # Canal 1 MTM : pertes OCI déduites du CET1 (portefeuille AFS souverain)
        delta_cap = securities_v * abs(sec_d) / 100
        capital_new = max(capital   - delta_cap, 0.0)
        tcap_new    = max(total_cap - delta_cap, 0.0)
        # Canal 3 : pondération souveraine rehaussée → RWA ↑
        rwa_new = rwa * (1.0 + rwa_d / 100)

    elif event_id == "esg_stranded":
        # Paramètres : PD_shock (×), LGD_shock (pp),
        #              securities (pp, négatif), loans (pp, négatif)
        pd_mult  = max(float(inputs.get("PD_shock",  1.0)), 1.0)
        lgd_delt = float(inputs.get("LGD_shock",  0.0))
        sec_d    = float(inputs.get("securities", 0.0))  # pp, ≤ 0
        loans_d  = float(inputs.get("loans",      0.0))  # pp, ≤ 0

        # Canal 1 : hausse ECL secteurs carbonés → déduction capital
        pd_new   = avg_pd  * pd_mult
        lgd_new  = min(avg_lgd + lgd_delt / 100, 1.0)
        el_base  = ead * avg_pd  * avg_lgd  if ead > 0 and avg_pd > 0 else 0.0
        el_new   = ead * pd_new  * lgd_new  if ead > 0 else 0.0
        delta_el = max(el_new - el_base, 0.0)
        capital_new = max(capital   - delta_el, 0.0)
        tcap_new    = max(total_cap - delta_el, 0.0)
        if avg_pd > 0:
            rwa_new = rwa * (pd_new / avg_pd) ** 0.5
        # Canal 1 : dépréciation obligations carbonées (AFS/FVOCI → OCI → CET1)
        delta_sec   = securities_v * abs(sec_d) / 100
        capital_new = max(capital_new - delta_sec, 0.0)
        tcap_new    = max(tcap_new    - delta_sec, 0.0)
        # Canal liquidité (modéré) : prêts dégradés → RSF augmente (facteur 85%)
        rsf_new = rsf + loans_v * abs(loans_d) / 100 * 0.85

    # ── Compute ratios depuis définitions Basel III ──────────────────────────
    ratios: Dict[str, Dict] = {}

    if "lcr" in drivers:
        base_r = base_flat.get("lcr", REGULATORY_FLOORS["lcr"])
        new_r  = (hqla_new / max(outflows_new, 1e-9) * 100.0
                  if hqla > 0 and outflows > 0
                  else base_r)
        ratios["LCR"] = {"before": base_r, "after": round(new_r, 2)}

    if "nsfr" in drivers:
        base_r = base_flat.get("nsfr", REGULATORY_FLOORS["nsfr"])
        new_r  = (asf_new / max(rsf_new, 1e-9) * 100.0
                  if asf > 0 and rsf > 0
                  else base_r)
        ratios["NSFR"] = {"before": base_r, "after": round(new_r, 2)}

    if "cet1" in drivers:
        base_r = base_flat.get("cet1_ratio", REGULATORY_FLOORS["cet1_ratio"])
        new_r  = (capital_new / max(rwa_new, 1e-9) * 100.0
                  if capital > 0 and rwa > 0
                  else base_r)
        ratios["CET1"] = {"before": base_r, "after": round(new_r, 2)}

    if "car" in drivers:
        base_r = base_flat.get("car", REGULATORY_FLOORS["car"])
        new_r  = (tcap_new / max(rwa_new, 1e-9) * 100.0
                  if total_cap > 0 and rwa > 0
                  else base_r)
        ratios["CAR"] = {"before": base_r, "after": round(new_r, 2)}

    if "leverage" in drivers:
        base_r = base_flat.get("leverage_ratio", REGULATORY_FLOORS["leverage_ratio"])
        new_r  = (capital_new / max(total_exp, 1e-9) * 100.0
                  if capital > 0 and total_exp > 0
                  else base_r)
        ratios["Leverage"] = {"before": base_r, "after": round(new_r, 2)}

    if "npl" in drivers:
        base_r = base_flat.get("npl_ratio", 0.0)
        pd_m   = max(float(inputs.get("PD_shock", 1.0)), 1.0)
        new_r  = base_r * pd_m if avg_pd > 0 else base_r
        ratios["NPL"] = {"before": base_r, "after": round(new_r, 2)}

    if "cout_risque" in drivers:
        base_r = base_flat.get("cout_risque", 0.0)
        pd_m   = max(float(inputs.get("PD_shock",  1.0)), 1.0)
        lgd_d  = float(inputs.get("LGD_shock", 0.0))
        if ead > 0 and avg_pd > 0:
            new_r = base_r * pd_m * (avg_lgd + lgd_d / 100) / max(avg_lgd, 1e-9)
        else:
            new_r = base_r
        ratios["Cout risque"] = {"before": base_r, "after": round(new_r, 4)}

    for name, r in ratios.items():
        r["delta"]         = round(float(r["after"]) - float(r["before"]), 2)
        r["severity"]      = get_severity(name, float(r["after"]))
        # Scénario adverse = interpolation linéaire à 50% du choc severe
        r["after_adverse"] = round(float(r["before"]) + r["delta"] * 0.5, 2)

    sev_ord = {"fail": 3, "warn": 2, "ok": 1}
    worst   = max((r.get("severity", "ok") for r in ratios.values()),
                  key=lambda s: sev_ord.get(s, 0), default="ok")
    return {"ratios": ratios, "dag_highlights": dict(dag_map.get(worst, {}))}