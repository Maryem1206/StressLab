"""
variable_resolver.py
=====================
Résout les variables NGFS (Stage 1) en séries temporelles historiques.

Cascade de résolution :
  1. World Bank API (correspondance sémantique → fetch data)
  2. IMF WEO Datamapper API (correspondance sémantique → fetch data)
  3. DROP + warning

Note : le fichier historique uploadé sert UNIQUEMENT pour la variable TARGET.
       Toutes les variables macro NGFS sont résolues exclusivement via API.

Output :
  - hist_df enrichi avec les colonnes API-fetched
  - ResolverResult avec mapping + log de résolution
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .mapping import MappingResult
from .utils import EngineConfig, get_logger, normalize_label

log = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════
# 1) KNOWLEDGE BASE : Correspondance NGFS → codes WB / IMF
# ═════════════════════════════════════════════════════════════════════

@dataclass
class ResolutionRule:
    """Règle de correspondance variable NGFS → indicateur API."""
    concept: str                         # identifiant sémantique unique
    ngfs_keywords: List[str]             # mots-clés (lowercase) à chercher
    hist_col: str                        # nom de colonne dans hist_df
    wb_code: Optional[str] = None        # code World Bank
    imf_code: Optional[str] = None       # code IMF WEO Datamapper
    ngfs_exclude: List[str] = field(default_factory=list)  # mots-clés excluants
    unit: str = ""


# Ordre important : les règles plus spécifiques AVANT les plus génériques
RESOLUTION_RULES: List[ResolutionRule] = [
    # ── GDP & croissance ──────────────────────────────────────────────
    ResolutionRule(
        concept="gdp_growth",
        ngfs_keywords=["gdp growth", "gdp yoy", "real gdp growth"],
        hist_col="real_gdp_growth",
        wb_code="NY.GDP.MKTP.KD.ZG",
        imf_code="NGDP_RPCH",
        unit="%",
    ),
    ResolutionRule(
        concept="gdp_per_capita",
        ngfs_keywords=["gdp per capita"],
        hist_col="gdp_per_capita",
        wb_code="NY.GDP.PCAP.CD",
        imf_code="NGDPPC",
        unit="USD",
    ),
    ResolutionRule(
        concept="gdp_level",
        ngfs_keywords=["gdp", "gross domestic product"],
        ngfs_exclude=["growth", "per capita", "deflator", "yoy", "gap"],
        hist_col="gdp_current_usd",
        wb_code="NY.GDP.MKTP.CD",
        imf_code="NGDP",
        unit="USD",
    ),
    # ── Emploi / Chômage ──────────────────────────────────────────────
    ResolutionRule(
        concept="unemployment",
        ngfs_keywords=["unemployment"],
        hist_col="unemployment_rate",
        wb_code="SL.UEM.TOTL.ZS",
        imf_code="LUR",
        unit="%",
    ),
    ResolutionRule(
        concept="employment",
        ngfs_keywords=["employment"],
        ngfs_exclude=["unemployment"],
        hist_col="employment_ratio",
        wb_code="SL.EMP.TOTL.SP.ZS",
        unit="%",
    ),
    # ── Inflation ─────────────────────────────────────────────────────
    ResolutionRule(
        concept="inflation",
        ngfs_keywords=["inflation", "consumer price", "cpi"],
        ngfs_exclude=["food", "energy"],
        hist_col="cpi_inflation",
        wb_code="FP.CPI.TOTL.ZG",
        imf_code="PCPIPCH",
        unit="%",
    ),
    ResolutionRule(
        concept="price_level",
        ngfs_keywords=["price level", "price index"],
        ngfs_exclude=["carbon", "energy", "oil", "gas", "food"],
        hist_col="price_level_index",
        wb_code="FP.CPI.TOTL",
        unit="index",
    ),
    # ── Taux d'intérêt ────────────────────────────────────────────────
    ResolutionRule(
        concept="policy_rate",
        ngfs_keywords=["central bank rate", "policy rate", "policy interest",
                       "short term interest", "short-term interest",
                       "key interest rate", "intervention rate"],
        hist_col="policy_rate",
        wb_code="FR.INR.DPST",     # deposit rate proxy
        unit="%",
    ),
    ResolutionRule(
        concept="lending_rate",
        ngfs_keywords=["lending rate", "long term interest", "long-term interest",
                       "bond yield", "10y yield", "10 year"],
        hist_col="lending_rate",
        wb_code="FR.INR.LEND",
        unit="%",
    ),
    ResolutionRule(
        concept="real_interest_rate",
        ngfs_keywords=["real interest rate"],
        hist_col="real_interest_rate",
        wb_code="FR.INR.RINR",
        unit="%",
    ),
    # ── Taux de change ────────────────────────────────────────────────
    ResolutionRule(
        concept="reer",
        ngfs_keywords=["real effective exchange rate", "reer",
                       "effective exchange"],
        hist_col="real_effective_xrate",   # align CREDIT_MACRO_UNIVERSE
        wb_code="PX.REX.REER",
        unit="index",
    ),
    ResolutionRule(
        concept="exchange_rate",
        ngfs_keywords=["exchange rate", "fx rate", "nominal exchange"],
        ngfs_exclude=["effective", "real effective"],
        hist_col="exchange_rate_lcu_usd",
        wb_code="PA.NUS.FCRF",
        unit="LCU/USD",
    ),
    # ── Commerce extérieur ────────────────────────────────────────────
    ResolutionRule(
        concept="exports",
        ngfs_keywords=["export"],
        ngfs_exclude=["import", "net"],
        hist_col="exports_goods_services",
        wb_code="NE.EXP.GNFS.CD",
        unit="USD",
    ),
    ResolutionRule(
        concept="imports",
        ngfs_keywords=["import"],
        ngfs_exclude=["export", "net"],
        hist_col="imports_goods_services",
        wb_code="NE.IMP.GNFS.CD",
        unit="USD",
    ),
    ResolutionRule(
        concept="trade_balance",
        ngfs_keywords=["trade balance", "net trade", "trade deficit"],
        hist_col="trade_balance",
        wb_code="BN.GSR.GNFS.CD",
        unit="USD",
    ),
    ResolutionRule(
        concept="current_account",
        ngfs_keywords=["current account"],
        hist_col="current_account_pct_gdp",
        wb_code="BN.CAB.XOKA.GD.ZS",
        imf_code="BCA_NGDPD",
        unit="%GDP",
    ),
    # ── Investissement & Consommation ─────────────────────────────────
    ResolutionRule(
        concept="investment",
        ngfs_keywords=["investment", "capital formation", "gross fixed"],
        hist_col="investment_pct_gdp",
        wb_code="NE.GDI.TOTL.ZS",
        imf_code="NID_NGDP",
        unit="%GDP",
    ),
    ResolutionRule(
        concept="consumption",
        ngfs_keywords=["consumption", "household consumption",
                       "private consumption", "domestic demand"],
        hist_col="private_consumption_pct_gdp",
        wb_code="NE.CON.PRVT.ZS",
        unit="%GDP",
    ),
    # ── Finances publiques ────────────────────────────────────────────
    ResolutionRule(
        concept="gov_debt",
        ngfs_keywords=["government debt", "public debt", "sovereign debt",
                       "debt to gdp"],
        hist_col="gov_debt_gdp",           # align CREDIT_MACRO_UNIVERSE
        wb_code="GC.DOD.TOTL.GD.ZS",
        imf_code="GGXWDG_NGDP",
        unit="%GDP",
    ),
    ResolutionRule(
        concept="gov_revenue",
        ngfs_keywords=["government revenue", "fiscal revenue", "tax revenue"],
        hist_col="gov_revenue_pct_gdp",
        wb_code="GC.REV.XGRT.GD.ZS",
        unit="%GDP",
    ),
    ResolutionRule(
        concept="fiscal_balance",
        ngfs_keywords=["fiscal balance", "budget balance", "fiscal deficit",
                       "budget deficit"],
        hist_col="fiscal_balance_gdp",     # align CREDIT_MACRO_UNIVERSE
        wb_code="GC.NLD.TOTL.GD.ZS",
        imf_code="GGXCNL_NGDP",
        unit="%GDP",
    ),
    # ── Énergie & Matières premières ──────────────────────────────────
    ResolutionRule(
        concept="oil_price",
        ngfs_keywords=["oil price", "brent", "crude oil", "petroleum"],
        hist_col="oil_price_usd",
        wb_code=None,       # pas de code WB par pays
        imf_code=None,      # fetch spécial commodité
        unit="USD/bbl",
    ),
    ResolutionRule(
        concept="carbon_price",
        ngfs_keywords=["carbon price", "co2 price", "emission price",
                       "carbon tax"],
        hist_col="carbon_price",
        wb_code=None,
        unit="USD/tCO2",
    ),
    ResolutionRule(
        concept="coal_price",
        ngfs_keywords=["coal price"],
        hist_col="coal_price",
        wb_code=None,
        unit="USD/ton",
    ),
    ResolutionRule(
        concept="gas_price",
        ngfs_keywords=["gas price", "natural gas price"],
        hist_col="gas_price",
        wb_code=None,
        unit="USD/mmbtu",
    ),
    ResolutionRule(
        concept="energy_price",
        ngfs_keywords=["energy price", "final energy"],
        ngfs_exclude=["carbon", "coal", "gas", "oil"],
        hist_col="energy_price_index",
        wb_code=None,
        unit="index",
    ),
    # ── Marchés financiers ────────────────────────────────────────────
    ResolutionRule(
        concept="equity",
        ngfs_keywords=["equity", "stock market", "share price", "stock index",
                       "equity price"],
        hist_col="equity_index",
        wb_code="CM.MKT.INDX.ZG",
        unit="%",
    ),
    ResolutionRule(
        concept="house_price",
        ngfs_keywords=["house price", "housing price", "property price",
                       "real estate"],
        hist_col="house_price_index",
        wb_code=None,
        unit="index",
    ),
    # ── Démographie ───────────────────────────────────────────────────
    ResolutionRule(
        concept="population",
        ngfs_keywords=["population"],
        hist_col="population",
        wb_code="SP.POP.TOTL",
        imf_code="LP",
        unit="persons",
    ),
    # ── FDI ───────────────────────────────────────────────────────────
    ResolutionRule(
        concept="fdi",
        ngfs_keywords=["foreign direct investment", "fdi"],
        hist_col="fdi_net_inflows",
        wb_code="BX.KLT.DINV.CD.WD",
        unit="USD",
    ),
    # ── Salaires ──────────────────────────────────────────────────────
    ResolutionRule(
        concept="wages",
        ngfs_keywords=["wage", "labor cost", "compensation", "salary"],
        hist_col="wages_index",
        wb_code=None,
        unit="index",
    ),
]


# ═════════════════════════════════════════════════════════════════════
# 2) COUNTRY CODE RESOLVER
# ═════════════════════════════════════════════════════════════════════

# ISO-2 (World Bank) et ISO-3 (IMF) — principaux pays NGFS
_COUNTRY_CODES: Dict[str, Tuple[str, str]] = {
    # name_lower → (iso2, iso3)
    "egypt": ("EG", "EGY"),
    "tunisia": ("TN", "TUN"),
    "morocco": ("MA", "MAR"),
    "algeria": ("DZ", "DZA"),
    "nigeria": ("NG", "NGA"),
    "south africa": ("ZA", "ZAF"),
    "kenya": ("KE", "KEN"),
    "ghana": ("GH", "GHA"),
    "ethiopia": ("ET", "ETH"),
    "tanzania": ("TZ", "TZA"),
    "senegal": ("SN", "SEN"),
    "côte d'ivoire": ("CI", "CIV"),
    "cameroon": ("CM", "CMR"),
    # Europe
    "france": ("FR", "FRA"),
    "germany": ("DE", "DEU"),
    "united kingdom": ("GB", "GBR"),
    "uk": ("GB", "GBR"),
    "italy": ("IT", "ITA"),
    "spain": ("ES", "ESP"),
    "netherlands": ("NL", "NLD"),
    "belgium": ("BE", "BEL"),
    "switzerland": ("CH", "CHE"),
    "austria": ("AT", "AUT"),
    "portugal": ("PT", "PRT"),
    "greece": ("GR", "GRC"),
    "poland": ("PL", "POL"),
    "romania": ("RO", "ROU"),
    "sweden": ("SE", "SWE"),
    "norway": ("NO", "NOR"),
    "denmark": ("DK", "DNK"),
    "finland": ("FI", "FIN"),
    "ireland": ("IE", "IRL"),
    "czech republic": ("CZ", "CZE"),
    "czechia": ("CZ", "CZE"),
    "hungary": ("HU", "HUN"),
    "bulgaria": ("BG", "BGR"),
    "croatia": ("HR", "HRV"),
    "turkey": ("TR", "TUR"),
    "ukraine": ("UA", "UKR"),
    "russia": ("RU", "RUS"),
    # Americas
    "united states": ("US", "USA"),
    "usa": ("US", "USA"),
    "canada": ("CA", "CAN"),
    "brazil": ("BR", "BRA"),
    "mexico": ("MX", "MEX"),
    "argentina": ("AR", "ARG"),
    "colombia": ("CO", "COL"),
    "chile": ("CL", "CHL"),
    "peru": ("PE", "PER"),
    # Asia
    "china": ("CN", "CHN"),
    "japan": ("JP", "JPN"),
    "india": ("IN", "IND"),
    "south korea": ("KR", "KOR"),
    "korea": ("KR", "KOR"),
    "indonesia": ("ID", "IDN"),
    "thailand": ("TH", "THA"),
    "vietnam": ("VN", "VNM"),
    "malaysia": ("MY", "MYS"),
    "philippines": ("PH", "PHL"),
    "singapore": ("SG", "SGP"),
    "pakistan": ("PK", "PAK"),
    "bangladesh": ("BD", "BGD"),
    "saudi arabia": ("SA", "SAU"),
    "uae": ("AE", "ARE"),
    "united arab emirates": ("AE", "ARE"),
    "qatar": ("QA", "QAT"),
    "kuwait": ("KW", "KWT"),
    "jordan": ("JO", "JOR"),
    "lebanon": ("LB", "LBN"),
    "iraq": ("IQ", "IRQ"),
    "iran": ("IR", "IRN"),
    "israel": ("IL", "ISR"),
    # Oceania
    "australia": ("AU", "AUS"),
    "new zealand": ("NZ", "NZL"),
}



# ──────────────────────────────────────────────────────────────────────
# World Bank region codes — iso2 → WB aggregate region
# Utilisé pour résoudre les données régionales en mode CT
# ──────────────────────────────────────────────────────────────────────
_ISO2_TO_WB_REGION: Dict[str, str] = {
    # Middle East & North Africa
    "EG": "MNA", "SA": "MNA", "AE": "MNA", "QA": "MNA", "KW": "MNA",
    "JO": "MNA", "LB": "MNA", "IQ": "MNA", "IR": "MNA", "IL": "MNA",
    "TN": "MNA", "MA": "MNA", "DZ": "MNA", "LY": "MNA", "YE": "MNA",
    # Sub-Saharan Africa
    "NG": "SSA", "ZA": "SSA", "KE": "SSA", "GH": "SSA", "ET": "SSA",
    "TZ": "SSA", "SN": "SSA", "CI": "SSA", "CM": "SSA",
    # Europe & Central Asia
    "FR": "ECS", "DE": "ECS", "GB": "ECS", "IT": "ECS", "ES": "ECS",
    "NL": "ECS", "BE": "ECS", "CH": "ECS", "AT": "ECS", "PT": "ECS",
    "GR": "ECS", "PL": "ECS", "RO": "ECS", "SE": "ECS", "NO": "ECS",
    "DK": "ECS", "FI": "ECS", "IE": "ECS", "CZ": "ECS", "HU": "ECS",
    "BG": "ECS", "HR": "ECS", "TR": "ECS", "UA": "ECS", "RU": "ECS",
    # Latin America & Caribbean
    "BR": "LCN", "MX": "LCN", "AR": "LCN", "CO": "LCN", "CL": "LCN",
    "PE": "LCN",
    # South Asia
    "IN": "SAS", "PK": "SAS", "BD": "SAS",
    # East Asia & Pacific
    "CN": "EAP", "JP": "EAP", "KR": "EAP", "ID": "EAP", "TH": "EAP",
    "VN": "EAP", "MY": "EAP", "PH": "EAP", "SG": "EAP", "AU": "EAP",
    "NZ": "EAP",
    # North America
    "US": "NAC", "CA": "NAC",
}


def get_wb_region_code(country_name: str) -> Optional[str]:
    """
    Retourne le code région World Bank pour un pays donné.
    Retourne None si le pays ou sa région est inconnu.

    Codes WB regions :
        MNA = Middle East & North Africa
        SSA = Sub-Saharan Africa
        ECS = Europe & Central Asia
        LCN = Latin America & Caribbean
        SAS = South Asia
        EAP = East Asia & Pacific
        NAC = North America
    """
    iso2, _ = _get_country_codes(country_name)
    if iso2 is None:
        return None
    return _ISO2_TO_WB_REGION.get(iso2)


def _get_country_codes(country_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Résout un nom de pays en (ISO-2, ISO-3). Retourne (None, None) si inconnu."""
    key = country_name.strip().lower()
    if key in _COUNTRY_CODES:
        return _COUNTRY_CODES[key]
    # Essai partiel
    for k, v in _COUNTRY_CODES.items():
        if key in k or k in key:
            return v
    log.warning("Country code inconnu pour '%s'.", country_name)
    return None, None


# ═════════════════════════════════════════════════════════════════════
# 3) SEMANTIC MATCHER : NGFS variable name → ResolutionRule
# ═════════════════════════════════════════════════════════════════════

def _normalize_ngfs_name(name: str) -> str:
    """Normalise un nom de variable NGFS pour le matching sémantique."""
    # Retirer unité entre ; ou entre parenthèses
    name = re.sub(r'\s*;.*$', '', name)
    name = re.sub(r'\s*\(.*?\)', '', name)
    # Retirer pipe GEM-E3
    name = name.replace('|', ' ')
    return name.strip().lower()


def _match_rule(ngfs_var: str) -> Optional[ResolutionRule]:
    """Trouve la meilleure ResolutionRule pour une variable NGFS."""
    norm = _normalize_ngfs_name(ngfs_var)

    best_rule: Optional[ResolutionRule] = None
    best_score: int = 0

    for rule in RESOLUTION_RULES:
        # Vérifier exclusions d'abord
        excluded = False
        for excl in rule.ngfs_exclude:
            if excl.lower() in norm:
                excluded = True
                break
        if excluded:
            continue

        # Scorer par nombre de mots-clés trouvés
        score = 0
        for kw in rule.ngfs_keywords:
            kw_lower = kw.lower()
            # Match exact substring
            if kw_lower in norm:
                score += len(kw_lower.split()) * 2  # bonus substring exact
            else:
                # Match tous les mots du keyword présents (ordre libre)
                kw_words = kw_lower.split()
                if all(w in norm for w in kw_words):
                    score += len(kw_words)

        if score > best_score:
            best_score = score
            best_rule = rule

    if best_rule and best_score > 0:
        log.debug("Matched NGFS '%s' → rule '%s' (score=%d).",
                  ngfs_var, best_rule.concept, best_score)
        return best_rule

    return None


# ═════════════════════════════════════════════════════════════════════
# 4) API FETCHERS
# ═════════════════════════════════════════════════════════════════════

_REQUEST_TIMEOUT = 15  # secondes
_REQUEST_DELAY = 0.3   # délai entre appels (rate limiting)


def _api_get(url: str) -> Optional[Any]:
    """Fetch JSON depuis une URL avec gestion d'erreurs."""
    try:
        req = Request(url, headers={"User-Agent": "StressTestEngine/1.0"})
        with urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        time.sleep(_REQUEST_DELAY)  # rate limiting
        return data
    except HTTPError as e:
        log.warning("API HTTP %d pour %s", e.code, url)
        return None
    except URLError as e:
        log.warning("API erreur réseau pour %s: %s", url, e.reason)
        return None
    except Exception as e:
        log.warning("API erreur inattendue pour %s: %s", url, e)
        return None


def fetch_worldbank(
    indicator: str,
    iso2: str,
    start_year: int = 2000,
    end_year: int = 2025,
) -> Optional[pd.Series]:
    """
    Fetch un indicateur World Bank pour un pays.

    Returns: pd.Series indexée par année (int), ou None si échec.
    """
    url = (
        f"https://api.worldbank.org/v2/country/{iso2}/indicator/{indicator}"
        f"?format=json&date={start_year}:{end_year}&per_page=500"
    )
    log.info("WB API: fetching %s for %s...", indicator, iso2)

    data = _api_get(url)
    if data is None or len(data) < 2 or not isinstance(data[1], list):
        log.warning("WB API: pas de données pour %s/%s.", iso2, indicator)
        return None

    records = []
    for entry in data[1]:
        year = int(entry.get("date", 0))
        value = entry.get("value")
        if year > 0 and value is not None:
            records.append((year, float(value)))

    if not records:
        log.warning("WB API: données vides pour %s/%s.", iso2, indicator)
        return None

    series = pd.Series(
        {yr: val for yr, val in records},
        name=indicator,
        dtype=float,
    ).sort_index()

    log.info("WB API: %s/%s → %d points (%d-%d).",
             iso2, indicator, len(series),
             series.index.min(), series.index.max())
    return series


def fetch_imf_weo(
    indicator: str,
    iso3: str,
) -> Optional[pd.Series]:
    """
    Fetch un indicateur IMF WEO Datamapper pour un pays.

    Returns: pd.Series indexée par année (int), ou None si échec.
    """
    url = (
        f"https://www.imf.org/external/datamapper/api/v1/{indicator}/{iso3}"
    )
    log.info("IMF API: fetching %s for %s...", indicator, iso3)

    data = _api_get(url)
    if data is None:
        return None

    try:
        values_dict = data.get("values", {}).get(indicator, {}).get(iso3, {})
    except (AttributeError, TypeError):
        log.warning("IMF API: format inattendu pour %s/%s.", iso3, indicator)
        return None

    if not values_dict:
        log.warning("IMF API: pas de données pour %s/%s.", iso3, indicator)
        return None

    records = {}
    for year_str, val in values_dict.items():
        try:
            year = int(year_str)
            records[year] = float(val)
        except (ValueError, TypeError):
            continue

    if not records:
        return None

    series = pd.Series(records, dtype=float).sort_index()
    series.name = indicator

    log.info("IMF API: %s/%s → %d points (%d-%d).",
             iso3, indicator, len(series),
             series.index.min(), series.index.max())
    return series


# ═════════════════════════════════════════════════════════════════════
# 5) OIL PRICE (spécial — indicateur global, pas par pays)
# ═════════════════════════════════════════════════════════════════════

def _fetch_oil_price_wb(start_year: int = 2000, end_year: int = 2025) -> Optional[pd.Series]:
    """
    Fetch le prix du pétrole Brent via la World Bank Commodity Prices API.
    Utilise le code spécial pour les données mondiales.
    """
    # World Bank ne fournit pas le pétrole par pays mais globalement
    # On utilise le proxy IMF Primary Commodity Prices
    url = (
        f"https://api.worldbank.org/v2/country/WLD/indicator/EP.PMP.SGAS.CD"
        f"?format=json&date={start_year}:{end_year}&per_page=500"
    )
    data = _api_get(url)
    if data and len(data) >= 2 and isinstance(data[1], list):
        records = {}
        for entry in data[1]:
            yr = int(entry.get("date", 0))
            val = entry.get("value")
            if yr > 0 and val is not None:
                records[yr] = float(val)
        if records:
            return pd.Series(records, dtype=float, name="oil_price").sort_index()

    # Fallback : IMF commodity prices
    log.info("Oil price: essai IMF commodity...")
    url2 = "https://www.imf.org/external/datamapper/api/v1/POILBRE"
    data2 = _api_get(url2)
    if data2:
        try:
            wld = data2.get("values", {}).get("POILBRE", {}).get("WEOWORLD", {})
            if wld:
                records = {int(y): float(v) for y, v in wld.items()
                           if start_year <= int(y) <= end_year}
                if records:
                    return pd.Series(records, dtype=float, name="oil_price").sort_index()
        except Exception:
            pass

    return None


# ═════════════════════════════════════════════════════════════════════
# 6) CACHE MANAGEMENT
# ═════════════════════════════════════════════════════════════════════

_DEFAULT_CACHE_DIR = "data_cache"
_CACHE_MAX_AGE_DAYS = 30  # refresh si plus vieux que 30 jours


def _cache_path(cache_dir: str, source: str, country: str, indicator: str) -> Path:
    """Génère le chemin du fichier cache."""
    safe_ind = re.sub(r'[^\w\-.]', '_', indicator)
    return Path(cache_dir) / f"{source}_{country}_{safe_ind}.csv"


def _cache_read(path: Path, max_age_days: int = _CACHE_MAX_AGE_DAYS) -> Optional[pd.Series]:
    """Lit un cache CSV si il existe et n'est pas périmé."""
    if not path.exists():
        return None
    mod_time = datetime.fromtimestamp(path.stat().st_mtime)
    if datetime.now() - mod_time > timedelta(days=max_age_days):
        log.info("Cache périmé: %s (âge=%d jours).", path.name,
                 (datetime.now() - mod_time).days)
        return None
    try:
        df = pd.read_csv(path)
        return pd.Series(df["value"].values, index=df["year"].astype(int).values,
                         dtype=float)
    except Exception as e:
        log.warning("Cache lecture échouée %s: %s", path.name, e)
        return None


def _cache_write(path: Path, series: pd.Series) -> None:
    """Écrit une série dans le cache CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"year": series.index, "value": series.values})
    df.to_csv(path, index=False)
    log.debug("Cache écrit: %s (%d points).", path.name, len(series))


# ═════════════════════════════════════════════════════════════════════
# 7) FUZZY MATCH AGAINST FILE
# ═════════════════════════════════════════════════════════════════════

def _fuzzy_match_file_column(
    ngfs_var: str,
    hist_cols: List[str],
    threshold: float = 0.99,
) -> Optional[str]:
    """
    Match une variable NGFS contre les colonnes du fichier historique.
    Utilise la similarité de séquence (SequenceMatcher) avec seuil strict.
    """
    from difflib import SequenceMatcher

    norm_ngfs = normalize_label(ngfs_var)
    best_col = None
    best_ratio = 0.0

    for col in hist_cols:
        norm_col = normalize_label(col)
        ratio = SequenceMatcher(None, norm_ngfs, norm_col).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = col

    if best_ratio >= threshold and best_col is not None:
        log.debug("Fuzzy file match: '%s' → '%s' (ratio=%.2f).",
                  ngfs_var, best_col, best_ratio)
        return best_col

    return None


# ═════════════════════════════════════════════════════════════════════
# 8) MAIN RESOLVER
# ═════════════════════════════════════════════════════════════════════

@dataclass
class ResolvedVariable:
    """Résultat de résolution pour une variable NGFS."""
    ngfs_var: str                         # nom NGFS original
    hist_col: str = ""                    # nom de colonne dans hist_df
    source: str = "dropped"               # "file" | "worldbank" | "imf" | "dropped"
    api_code: str = ""                    # code WB ou IMF utilisé
    concept: str = ""                     # concept sémantique
    status: str = "pending"               # "ok" | "dropped" | "error"
    message: str = ""                     # détails


@dataclass
class ResolverResult:
    """Résultat complet de la résolution de variables."""
    mapping: MappingResult = field(default_factory=MappingResult)
    resolution_log: List[ResolvedVariable] = field(default_factory=list)
    enriched_hist_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    sources: Dict[str, str] = field(default_factory=dict)  # hist_col → source


def _resolve_concept_data(
    rule: ResolutionRule,
    iso2: Optional[str],
    iso3: Optional[str],
    start_year: int,
    end_year: int,
    cache_dir: str,
) -> Dict[str, Any]:
    """
    Resolve ONE concept's historical series via cache → WB API → IMF API →
    oil-global → none.

    Pure w.r.t. in-memory shared state — only touches the on-disk cache
    (per-indicator file, safe to read/write across threads) and performs
    network I/O. Designed to run concurrently, one call per distinct
    concept; the caller applies the result to the shared DataFrame and
    resolution log sequentially afterwards.
    """
    # ── Étape 1a/1b : World Bank (cache puis API) ────────────────────
    if iso2 and rule.wb_code:
        cached = _cache_read(_cache_path(cache_dir, "wb", iso2, rule.wb_code))
        if cached is not None:
            return {
                "series": cached, "source": "worldbank", "api_code": rule.wb_code,
                "method": "worldbank_cache", "icon": "cache/wb",
                "message": f"Cache WB: {rule.wb_code}",
            }
        series = fetch_worldbank(rule.wb_code, iso2, start_year, end_year)
        if series is not None and not series.empty:
            _cache_write(_cache_path(cache_dir, "wb", iso2, rule.wb_code), series)
            return {
                "series": series, "source": "worldbank", "api_code": rule.wb_code,
                "method": "worldbank", "icon": "wb",
                "message": f"World Bank API: {rule.wb_code} ({len(series)} pts)",
            }

    # ── Étape 1c : IMF WEO (cache puis API) ───────────────────────────
    if iso3 and rule.imf_code:
        cached = _cache_read(_cache_path(cache_dir, "imf", iso3, rule.imf_code))
        if cached is not None:
            return {
                "series": cached, "source": "imf", "api_code": rule.imf_code,
                "method": "imf_cache", "icon": "cache/imf",
                "message": f"Cache IMF: {rule.imf_code}",
            }
        series = fetch_imf_weo(rule.imf_code, iso3)
        if series is not None and not series.empty:
            _cache_write(_cache_path(cache_dir, "imf", iso3, rule.imf_code), series)
            return {
                "series": series, "source": "imf", "api_code": rule.imf_code,
                "method": "imf", "icon": "imf",
                "message": f"IMF WEO API: {rule.imf_code} ({len(series)} pts)",
            }

    # ── Étape 1d : Oil price (global, sans code pays) ─────────────────
    if rule.concept == "oil_price":
        cached = _cache_read(_cache_path(cache_dir, "wb", "WLD", "oil_price"))
        if cached is not None:
            return {
                "series": cached, "source": "worldbank", "api_code": "",
                "method": "worldbank_global", "icon": "cache/oil",
                "message": "Cache oil price global",
            }
        series = _fetch_oil_price_wb(start_year, end_year)
        if series is not None and not series.empty:
            _cache_write(_cache_path(cache_dir, "wb", "WLD", "oil_price"), series)
            return {
                "series": series, "source": "worldbank", "api_code": "",
                "method": "worldbank_global", "icon": "wb/oil",
                "message": f"Oil price global ({len(series)} pts)",
            }

    # ── Étape 2 : DROP ─────────────────────────────────────────────────
    return {
        "series": None, "source": "dropped", "api_code": "",
        "method": "", "icon": "drop",
        "message": (
            f"Concept='{rule.concept}' — "
            f"WB={'N/A' if not rule.wb_code else rule.wb_code}, "
            f"IMF={'N/A' if not rule.imf_code else rule.imf_code} — "
            f"pas de données"
        ),
    }


def resolve_variables(
    ngfs_vars: List[str],
    cfg: EngineConfig,
    hist_df: Optional[pd.DataFrame] = None,
    start_year: int = 2000,
    end_year: int = 2025,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    override_iso2: Optional[str] = None,
) -> ResolverResult:
    """
    Résout les variables NGFS en séries historiques.

    Cascade: World Bank API → IMF WEO API → DROP.

    Le fichier hist_df est enrichi avec les séries API mais ses colonnes
    ne sont PAS utilisées comme source de matching pour les variables NGFS.

    Parameters
    ----------
    ngfs_vars : variables NGFS issues de Stage 1
    cfg : EngineConfig (contient country, historical_path, etc.)
    hist_df : DataFrame historique uploadé (optionnel)
    start_year : première année à fetcher via API
    end_year : dernière année à fetcher via API
    cache_dir : dossier pour le cache des données API
    override_iso2 : si fourni, remplace le code ISO2 du pays pour les appels
                    World Bank. Permet d'utiliser des données régionales (ex:
                    'MNA' pour Middle East & North Africa) au lieu de données
                    pays, évitant ainsi le rebasing et ses artefacts.

    Returns
    -------
    ResolverResult avec hist_df enrichi, mapping, et log
    """
    log.info("=== VARIABLE RESOLVER: %d variables à résoudre ===", len(ngfs_vars))

    # ── Préparer les codes pays / région ─────────────────────────────
    if override_iso2:
        iso2 = override_iso2
        iso3 = None
        log.info(
            "RESOLVER: mode régional explicite — utilisation de '%s' "
            "au lieu du pays '%s'.",
            iso2, cfg.country,
        )
    else:
        iso2, iso3 = _get_country_codes(cfg.country)
        if not iso2:
            log.warning("Pas de code ISO pour '%s'. API fetch désactivé.", cfg.country)

    # ── Préparer le DataFrame historique ─────────────────────────────
    year_col = cfg.hist_year_col
    if hist_df is not None and not hist_df.empty:
        enriched = hist_df.copy()
        log.info("Fichier historique fourni: %d lignes (utilisé pour TARGET uniquement).",
                 len(enriched))
    else:
        enriched = pd.DataFrame({year_col: list(range(start_year, end_year + 1))})
        log.info("Pas de fichier historique.")
    # Les colonnes fichier ne sont JAMAIS utilisées pour résoudre les vars NGFS.
    # Seule l'API (WB → IMF) est autorisée comme source de matching.

    # ── Résolution par variable ──────────────────────────────────────
    result = ResolverResult()

    # ── Phase 1 (séquentielle, sans I/O) : matching + groupement par
    #    concept, en conservant l'ordre de première apparition. Le
    #    dédoublonnage "même concept → même fetch" est ainsi décidé
    #    AVANT tout appel réseau, ce qui rend la Phase 2 parallélisable
    #    sans dépendance d'ordre ni état mutable partagé.
    var_rule: Dict[str, ResolutionRule] = {}
    concept_order: List[str] = []
    concept_members: Dict[str, List[str]] = {}

    for ngfs_var in ngfs_vars:
        rule = _match_rule(ngfs_var)
        if rule is None:
            rv = ResolvedVariable(
                ngfs_var=ngfs_var, status="dropped",
                message="Aucune règle sémantique trouvée",
            )
            result.mapping.unmatched.append(ngfs_var)
            result.resolution_log.append(rv)
            log.warning("✗ [drop]  %s — pas de règle sémantique.", ngfs_var)
            continue

        var_rule[ngfs_var] = rule
        if rule.concept not in concept_members:
            concept_members[rule.concept] = []
            concept_order.append(rule.concept)
        concept_members[rule.concept].append(ngfs_var)

    # ── Phase 2 (parallèle, I/O réseau) : un seul fetch par concept
    #    distinct (cache → WB → IMF → oil-global → drop).
    concept_data: Dict[str, Dict[str, Any]] = {}
    if concept_order:
        n_workers = min(len(concept_order), 8)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _resolve_concept_data,
                    var_rule[concept_members[c][0]],
                    iso2, iso3, start_year, end_year, cache_dir,
                ): c
                for c in concept_order
            }
            for fut in as_completed(futures):
                concept_data[futures[fut]] = fut.result()

    # ── Phase 3 (séquentielle, sans I/O) : applique les résultats au
    #    DataFrame partagé et au log, dans l'ordre d'entrée original.
    for ngfs_var in ngfs_vars:
        rule = var_rule.get(ngfs_var)
        if rule is None:
            continue  # déjà journalisé en Phase 1 (pas de règle)

        concept = rule.concept
        data = concept_data[concept]
        members = concept_members[concept]
        is_primary = ngfs_var == members[0]
        series = data["series"]

        if series is None:
            rv = ResolvedVariable(
                ngfs_var=ngfs_var, concept=concept, status="dropped",
                message=data["message"],
            )
            result.mapping.unmatched.append(ngfs_var)
            result.resolution_log.append(rv)
            log.warning("✗ [drop]  %s — concept '%s', pas de source API.",
                        ngfs_var, concept)
            continue

        if is_primary:
            _add_series_to_df(enriched, series, rule.hist_col, year_col)
            rv = ResolvedVariable(
                ngfs_var=ngfs_var, hist_col=rule.hist_col,
                source=data["source"], api_code=data["api_code"],
                concept=concept, status="ok", message=data["message"],
            )
            result.mapping.matches[ngfs_var] = rule.hist_col
            result.mapping.method_per_match[ngfs_var] = data["method"]
            result.sources[rule.hist_col] = data["source"]
            result.resolution_log.append(rv)
            log.info("✓ [%s] %s → %s (%s)",
                      data["icon"], ngfs_var, rule.hist_col, data["message"])
        else:
            rv = ResolvedVariable(
                ngfs_var=ngfs_var, hist_col=rule.hist_col,
                source="api_dedup", concept=concept, status="ok",
                message=f"Concept '{concept}' déjà résolu via API → réutilisé",
            )
            result.mapping.matches[ngfs_var] = rule.hist_col
            result.mapping.method_per_match[ngfs_var] = "api_dedup"
            result.sources[rule.hist_col] = "api"
            result.resolution_log.append(rv)
            log.info("⊘ [dedup] %s → %s (concept '%s' déjà en cache)",
                     ngfs_var, rule.hist_col, concept)

    # ── Résumé ───────────────────────────────────────────────────────
    result.enriched_hist_df = enriched
    n_ok = sum(1 for rv in result.resolution_log if rv.status == "ok")
    n_drop = sum(1 for rv in result.resolution_log if rv.status == "dropped")
    sources_count = {}
    for rv in result.resolution_log:
        if rv.status == "ok":
            sources_count[rv.source] = sources_count.get(rv.source, 0) + 1

    log.info(
        "=== RESOLVER DONE: %d/%d résolues | %d droppées | sources: %s ===",
        n_ok, len(ngfs_vars), n_drop, sources_count,
    )

    return result


# ═════════════════════════════════════════════════════════════════════
# 9) HELPERS
# ═════════════════════════════════════════════════════════════════════

def _add_series_to_df(
    df: pd.DataFrame,
    series: pd.Series,
    col_name: str,
    year_col: str,
) -> None:
    """Ajoute une série temporelle (indexée par année) au DataFrame."""
    if year_col not in df.columns:
        df[year_col] = list(range(series.index.min(), series.index.max() + 1))

    years = df[year_col].astype(int)
    values = years.map(series.to_dict())
    df[col_name] = values.astype(float)


def print_resolution_log(result: ResolverResult) -> None:
    """Affiche le log de résolution de manière lisible."""
    print("\n" + "=" * 78)
    print("  VARIABLE RESOLUTION LOG")
    print("=" * 78)

    for rv in result.resolution_log:
        if rv.status == "ok":
            icon = "OK"
        elif rv.status == "dropped":
            icon = "X"
        else:
            icon = "?"

        src = f"[{rv.source:10s}]" if rv.source != "dropped" else "[DROPPED   ]"
        print(f"  {icon} {src}  {rv.ngfs_var}")
        if rv.hist_col:
            print(f"              -> col='{rv.hist_col}'  api={rv.api_code or 'N/A'}")
        if rv.status == "dropped":
            print(f"              -> {rv.message}")

    n_ok = sum(1 for rv in result.resolution_log if rv.status == "ok")
    n_drop = sum(1 for rv in result.resolution_log if rv.status == "dropped")
    print(f"\n  Resultat: {n_ok} resolues, {n_drop} droppees "
          f"sur {len(result.resolution_log)} variables.")
    print("=" * 78)
