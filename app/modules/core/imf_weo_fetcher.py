"""
imf_weo_fetcher.py
==================
Standardized macro variable collection for the credit risk module.

Pipeline :
    user → country_iso2 (ex: "EG", "FR", "TN") + historical period
         → fetch_credit_macro() returns a pandas DataFrame
            • rows : years
            • cols : ~30 macro indicators (standardized names)

Sources, in priority order :
    1. IMF WEO Datamapper API (https://www.imf.org/external/datamapper/api/v1)
       - ~50 high-quality macro indicators per country
       - Includes IMF projections (5-year forward) — used for baseline
    2. World Bank API (https://api.worldbank.org/v2)
       - ~1500 indicators per country, used as fallback
       - Historical only (no projections)

Caching :
    - Local CSV files in cache_dir/
    - TTL configurable (default 30 days)
    - Format : {source}_{country_iso2}_{indicator}.csv
    - Compatible with existing data_cache/ in the climate module

Standardization :
    - Country : ISO2 input, auto-converted to ISO3 for IMF API
    - Variable names : standardized to hist_col convention (snake_case)
    - Works for any of the ~190 countries in IMF WEO

Academic references :
    - IMF (2024) "World Economic Outlook Database — Subject Codes"
    - World Bank (2024) "World Development Indicators — Topic Codes"
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

LOG = logging.getLogger("imf_weo_fetcher")


# ═══════════════════════════════════════════════════════════════════════════
# 1. ISO2 → ISO3 mapping (190+ countries)
# ═══════════════════════════════════════════════════════════════════════════
ISO2_TO_ISO3: Dict[str, str] = {
    # MENA + Africa
    "EG": "EGY", "MA": "MAR", "TN": "TUN", "DZ": "DZA", "LY": "LBY",
    "SD": "SDN", "ET": "ETH", "KE": "KEN", "ZA": "ZAF", "NG": "NGA",
    "GH": "GHA", "CI": "CIV", "SN": "SEN", "AO": "AGO", "MZ": "MOZ",
    "TZ": "TZA", "UG": "UGA", "RW": "RWA", "ZM": "ZMB", "ZW": "ZWE",
    # Middle East
    "SA": "SAU", "AE": "ARE", "QA": "QAT", "KW": "KWT", "BH": "BHR",
    "OM": "OMN", "JO": "JOR", "LB": "LBN", "IL": "ISR", "TR": "TUR",
    "IR": "IRN", "IQ": "IRQ",
    # Europe
    "FR": "FRA", "DE": "DEU", "GB": "GBR", "IT": "ITA", "ES": "ESP",
    "PT": "PRT", "NL": "NLD", "BE": "BEL", "CH": "CHE", "AT": "AUT",
    "SE": "SWE", "NO": "NOR", "DK": "DNK", "FI": "FIN", "IE": "IRL",
    "PL": "POL", "CZ": "CZE", "HU": "HUN", "RO": "ROU", "BG": "BGR",
    "GR": "GRC", "RU": "RUS", "UA": "UKR",
    # Americas
    "US": "USA", "CA": "CAN", "MX": "MEX", "BR": "BRA", "AR": "ARG",
    "CL": "CHL", "CO": "COL", "PE": "PER", "VE": "VEN", "UY": "URY",
    # Asia-Pacific
    "CN": "CHN", "JP": "JPN", "KR": "KOR", "IN": "IND", "ID": "IDN",
    "MY": "MYS", "TH": "THA", "VN": "VNM", "PH": "PHL", "SG": "SGP",
    "AU": "AUS", "NZ": "NZL", "PK": "PAK", "BD": "BGD", "LK": "LKA",
    "MM": "MMR", "KH": "KHM", "LA": "LAO",
}


def _to_iso3(country: str) -> str:
    """Convert ISO2 to ISO3. If already ISO3, return as-is.
    Raises ValueError if unknown."""
    c = country.upper().strip()
    if len(c) == 3 and c.isalpha():
        return c
    if len(c) == 2 and c in ISO2_TO_ISO3:
        return ISO2_TO_ISO3[c]
    raise ValueError(
        f"Unknown country code '{country}'. Provide ISO2 (e.g. 'EG') "
        f"or ISO3 (e.g. 'EGY')."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. MACRO UNIVERSE (~35 standardized indicators)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class MacroIndicator:
    """Single standardized macro indicator definition.

    Attributes
    ----------
    name : standardized snake_case column name in the output DataFrame
    weo_code : IMF WEO Subject Code (None if not in WEO)
    wb_code : World Bank Indicator Code (None if not in WB)
    unit : "%" | "%GDP" | "index" | "USD" | "USD/bbl" | "LCU/USD" | "persons"
    shock_key : matches scenario_manager.py keys (gdp_delta, rate_bps, ...)
    shock_unit : how to apply the shock (see _apply_shock in wrapper)
                 "additive_pp" — shock in fraction → add as percentage points
                 "additive_pct" — shock already in same units (additive)
                 "bps_to_pp"   — bps to percentage points
                 "multiplicative" — shock as fraction of base value
                 None          — no shock applies
    expected_sign : economic sign of β in PD regression (-1, 0, +1)
    """
    name: str
    weo_code: Optional[str] = None
    wb_code: Optional[str] = None
    unit: str = ""
    shock_key: Optional[str] = None
    shock_unit: Optional[str] = None
    expected_sign: int = 0
    fred_code: Optional[str] = None   # FRED series ID (e.g. "EGYPTAINTRATE")


# Curated universe — covers the 5 macro transmission channels for credit risk
# (Drehmann 2008, Agresti et al. 2011, EBA 2023 Methodological Note)
CREDIT_MACRO_UNIVERSE: List[MacroIndicator] = [
    # ── Real demand channel ─────────────────────────────────────────────
    MacroIndicator("real_gdp_growth",          "NGDP_RPCH", "NY.GDP.MKTP.KD.ZG", "%",   "gdp_delta",          "additive_pp", -1),
    MacroIndicator("gdp_per_capita",           "NGDPPC",    "NY.GDP.PCAP.CD",    "USD", None,                 None,           -1),
    MacroIndicator("gdp_current_usd",          "NGDPD",     "NY.GDP.MKTP.CD",    "USD", None,                 None,           -1),
    MacroIndicator("unemployment_rate",        "LUR",       "SL.UEM.TOTL.ZS",    "%",   "unemployment_delta", "additive_pp", +1),
    MacroIndicator("private_consumption_gdp",  None,        "NE.CON.PRVT.ZS",    "%GDP", None,                None,           -1),
    MacroIndicator("investment_gdp",           "NID_NGDP",  "NE.GDI.TOTL.ZS",    "%GDP", None,                None,           -1),

    # ── Financial channel ───────────────────────────────────────────────
    MacroIndicator("policy_rate",              None,        "FR.INR.DPST",       "%",   "rate_bps",          "bps_to_pp",   +1),
    MacroIndicator("lending_rate",             None,        "FR.INR.LEND",       "%",   "rate_bps",          "bps_to_pp",   +1),
    MacroIndicator("real_interest_rate",       None,        "FR.INR.RINR",       "%",   "rate_bps",          "bps_to_pp",   +1),
    MacroIndicator("domestic_credit_gdp",      None,        "FS.AST.PRVT.GD.ZS", "%GDP", None,                None,           +1),

    # ── Price stability channel ─────────────────────────────────────────
    MacroIndicator("cpi_inflation",            "PCPIPCH",   "FP.CPI.TOTL.ZG",    "%",   "inflation_delta",   "additive_pp", +1),

    # ── External channel ────────────────────────────────────────────────
    MacroIndicator("real_effective_xrate",     None,        "PX.REX.REER",       "index","fx_pct",            "additive_pct",-1),
    MacroIndicator("exchange_rate_lcu_usd",    None,        "PA.NUS.FCRF",       "LCU/USD","fx_pct",          "multiplicative_neg",+1),
    MacroIndicator("exports_usd",              None,        "NE.EXP.GNFS.CD",    "USD", None,                 None,           -1),
    MacroIndicator("imports_usd",              None,        "NE.IMP.GNFS.CD",    "USD", None,                 None,            0),
    MacroIndicator("current_account_gdp",      "BCA_NGDPD", "BN.CAB.XOKA.GD.ZS", "%GDP","trade_delta",        "additive_pp", -1),

    # ── Sovereign / fiscal channel ──────────────────────────────────────
    MacroIndicator("gov_debt_gdp",             "GGXWDG_NGDP","GC.DOD.TOTL.GD.ZS","%GDP", None,                None,           +1),
    MacroIndicator("fiscal_balance_gdp",       "GGXCNL_NGDP","GC.NLD.TOTL.GD.ZS","%GDP", None,                None,           -1),
    MacroIndicator("gov_revenue_gdp",          None,        "GC.REV.XGRT.GD.ZS","%GDP",  None,                None,           -1),
    MacroIndicator("savings_gdp",              "NGSD_NGDP", None,                "%GDP", None,                None,           -1),

    # ── Asset price / external financing ─────────────────────────────────
    MacroIndicator("equity_index_growth",      None,        "CM.MKT.INDX.ZG",    "%",   None,                None,            -1),
    MacroIndicator("fdi_net_inflows_usd",      None,        "BX.KLT.DINV.CD.WD", "USD", None,                None,            -1),

    # ── External Reserves ────────────────────────────────────────────────────
    # Generic for all countries (WB FI.RES.TOTL.CD exists everywhere).
    # Monthly policy rate is country-specific: fetched via POLICY_RATE_FRED_BY_COUNTRY
    # in yield_curve_loader.py and added as "policy_rate_monthly" to macro_df there.
    MacroIndicator("fx_reserves_usd",   None, "FI.RES.TOTL.CD", "USD", None, None, 0),

    # ── Demographic ─────────────────────────────────────────────────────
    MacroIndicator("population",               "LP",        "SP.POP.TOTL",       "persons", None,             None,            0),
]


# ═══════════════════════════════════════════════════════════════════════════
# 3. CACHE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════
def _cache_path(cache_dir: Path, source: str, country: str, code: str) -> Path:
    """Path naming compatible with existing data_cache/ convention."""
    safe_code = code.replace("/", "_")
    return cache_dir / f"{source}_{country}_{safe_code}.csv"


def _read_cache(path: Path, ttl_days: int) -> Optional[pd.Series]:
    """Read cached series if exists and TTL not expired."""
    if not path.exists():
        return None
    age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
    if age > ttl_days:
        LOG.debug("Cache expired (%d > %d days): %s", age, ttl_days, path.name)
        return None
    try:
        df = pd.read_csv(path)
        if "year" not in df.columns or "value" not in df.columns:
            return None
        s = pd.Series(df["value"].values, index=df["year"].astype(int))
        s = s.dropna()
        return s if len(s) >= 2 else None
    except Exception as e:
        LOG.warning("Cannot read cache %s: %s", path, e)
        return None


def _write_cache(path: Path, series: pd.Series) -> None:
    """Persist a series to cache (year, value columns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"year": series.index.astype(int),
                       "value": series.values.astype(float)})
    df.to_csv(path, index=False)


# ═══════════════════════════════════════════════════════════════════════════
# 4. API FETCHERS
# ═══════════════════════════════════════════════════════════════════════════
_USER_AGENT = "stress-platform/2.0 (credit-module)"
_HTTP_TIMEOUT_SEC = 20
_HTTP_RETRY = 2


def _http_json(url: str) -> Optional[dict]:
    """Robust JSON fetch with timeout and retry."""
    for attempt in range(_HTTP_RETRY + 1):
        try:
            req = Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
            with urlopen(req, timeout=_HTTP_TIMEOUT_SEC) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            if attempt < _HTTP_RETRY:
                time.sleep(1.5 ** attempt)
                continue
            LOG.warning("HTTP fail (%s): %s", url, e)
            return None
        except Exception as e:
            LOG.warning("HTTP unexpected error (%s): %s", url, e)
            return None
    return None


def _fetch_weo(iso3: str, weo_code: str) -> Optional[pd.Series]:
    """Fetch a single indicator from IMF WEO Datamapper API.
    
    Returns a Series indexed by year, including IMF forward projections
    (typically up to current_year + 5).
    """
    url = f"https://www.imf.org/external/datamapper/api/v1/{weo_code}/{iso3}"
    data = _http_json(url)
    if not data or "values" not in data:
        return None
    try:
        country_data = data["values"][weo_code][iso3]
    except KeyError:
        return None
    if not country_data:
        return None
    s = pd.Series(country_data).astype(float)
    s.index = s.index.astype(int)
    s = s.sort_index().dropna()
    return s if len(s) >= 2 else None


def _fetch_wb(iso2: str, wb_code: str) -> Optional[pd.Series]:
    """Fetch a single indicator from World Bank API.
    Historical data only — no forward projections.
    """
    url = (f"https://api.worldbank.org/v2/country/{iso2}/indicator/{wb_code}"
           f"?format=json&per_page=200")
    data = _http_json(url)
    if not data or not isinstance(data, list) or len(data) < 2:
        return None
    rows = data[1]
    if not rows:
        return None
    try:
        records = [(int(r["date"]), float(r["value"]))
                   for r in rows if r.get("value") is not None]
    except (KeyError, ValueError, TypeError):
        return None
    if not records:
        return None
    s = pd.Series(dict(records)).astype(float)
    s = s.sort_index().dropna()
    return s if len(s) >= 2 else None


def fetch_fred_series(fred_code: str) -> Optional[pd.Series]:
    """Public alias -- see _fetch_fred."""
    return _fetch_fred(fred_code)


def _fetch_fred(fred_code: str) -> Optional[pd.Series]:
    """Fetch a monthly series from FRED via the public CSV endpoint.

    No API key required. URL pattern:
        https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}

    Returns a monthly pd.Series indexed by year (averaged to annual frequency
    so it can be merged with the annual macro DataFrame).
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_code}"
    try:
        req = Request(url, headers={"User-Agent": _USER_AGENT})
        import io
        with urlopen(req, timeout=_HTTP_TIMEOUT_SEC) as resp:
            raw_bytes = resp.read()
        df = pd.read_csv(io.StringIO(raw_bytes.decode("utf-8")))
        if df.shape[1] < 2:
            LOG.warning("FRED %s: unexpected CSV shape %s", fred_code, df.shape)
            return None
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        if len(df) < 2:
            return None
        # Average to annual frequency (year index) so it merges with annual macro df
        df["year"] = df["date"].dt.year
        annual = df.groupby("year")["value"].mean()
        annual.index = annual.index.astype(int)
        annual = annual.sort_index().dropna()
        return annual if len(annual) >= 2 else None
    except Exception as exc:
        LOG.warning("FRED fetch failed (%s): %s", fred_code, exc)
        return None


def _fetch_indicator(
    indicator: MacroIndicator,
    iso2: str,
    iso3: str,
    cache_dir: Path,
    cache_ttl_days: int,
) -> Tuple[Optional[pd.Series], str]:
    """Fetch one indicator. Try WEO first, fallback to WB.
    
    Returns (series, source) where source ∈ {"weo", "wb", "cache_weo", "cache_wb", "none"}.
    """
    # ── Try WEO cache, then WEO API ──────────────────────────────────────
    if indicator.weo_code:
        cache = _cache_path(cache_dir, "weo", iso2, indicator.weo_code)
        s = _read_cache(cache, cache_ttl_days)
        if s is not None:
            return s, "cache_weo"
        s = _fetch_weo(iso3, indicator.weo_code)
        if s is not None and len(s) >= 2:
            _write_cache(cache, s)
            return s, "weo"

    # ── Try WB cache, then WB API ────────────────────────────────────────
    if indicator.wb_code:
        cache = _cache_path(cache_dir, "wb", iso2, indicator.wb_code)
        s = _read_cache(cache, cache_ttl_days)
        if s is not None:
            return s, "cache_wb"
        s = _fetch_wb(iso2, indicator.wb_code)
        if s is not None and len(s) >= 2:
            _write_cache(cache, s)
            return s, "wb"

    # -- Try FRED cache, then FRED API (monthly series, averaged to annual) ----
    if indicator.fred_code:
        cache = _cache_path(cache_dir, "fred", iso2, indicator.fred_code)
        s = _read_cache(cache, cache_ttl_days)
        if s is not None:
            return s, "cache_fred"
        s = _fetch_fred(indicator.fred_code)
        if s is not None and len(s) >= 2:
            _write_cache(cache, s)
            return s, "fred"

    return None, "none"


# ═══════════════════════════════════════════════════════════════════════════
# 5. PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class FetchReport:
    """Diagnostic report of the fetch operation."""
    country_iso2: str
    country_iso3: str
    n_indicators_requested: int
    n_indicators_fetched: int
    sources: Dict[str, str] = field(default_factory=dict)   # name -> source tag
    missing: List[str] = field(default_factory=list)
    historical_period: Tuple[int, int] = (0, 0)
    has_projections: bool = False
    projections_until: Optional[int] = None



# Every module (climate, credit, liquidity, market) calls fetch_credit_macro()
# independently for the same country/params -- a single Phase 1 request was
# observed re-fetching the same 24-indicator universe 3-4 times (~2s each,
# all disk-cache reads/misses for the same data). This in-process cache
# collapses those into one real fetch; TTL is short (covers one request's
# worth of redundant calls) since the on-disk cache in _fetch_indicator()
# already handles longer-lived freshness (cache_ttl_days, default 30 days).
_MACRO_FETCH_CACHE: Dict[tuple, Tuple[float, pd.DataFrame, "FetchReport"]] = {}
_MACRO_FETCH_CACHE_TTL_SEC = 300


def fetch_credit_macro(
    country: str,
    start_year: int = 1990,
    end_year: Optional[int] = None,
    cache_dir: str = "data_cache",
    cache_ttl_days: int = 30,
    universe: Optional[List[MacroIndicator]] = None,
) -> Tuple[pd.DataFrame, FetchReport]:
    """Fetch the macro universe for a given country.

    Parameters
    ----------
    country : ISO2 (e.g. "EG") or ISO3 (e.g. "EGY")
    start_year : earliest year to keep (default 1990)
    end_year : latest year to keep INCLUDING projections; if None, keep all
    cache_dir : local cache directory (default "data_cache")
    cache_ttl_days : days before cache is refreshed
    universe : optional override of CREDIT_MACRO_UNIVERSE

    Returns
    -------
    df : DataFrame indexed by year, columns = standardized macro names
         Includes IMF WEO forward projections when available
    report : FetchReport with diagnostics

    Notes
    -----
    Variables with fewer than 5 non-null observations are dropped.
    Indicators that fail both WEO and WB API are listed in report.missing.
    """
    # NOTE: every cache return path below hands the caller df.copy(), never
    # the cached object itself. Callers (e.g. yield_curve_loader.py's
    # _fetch_macro(), which does `df.index = pd.to_datetime(...)`) mutate
    # the returned DataFrame in place -- without a copy, that mutation
    # corrupts the shared cache entry for every subsequent cache hit within
    # the TTL window. Two callers hitting the same cache key back-to-back
    # (climate's market-beta extraction + market-beta1 calibration both call
    # fetch_credit_macro(start_year=2005) via YieldCurveLoader) would then
    # apply that same index mutation TWICE -- the second pass turns an
    # already-datetime index into a malformed string ("...-01-01" appended to
    # an already-full timestamp), which pandas silently mis-parses as
    # carrying a UTC offset, producing a tz-aware index that then collides
    # with the tz-naive dates elsewhere ("Cannot compare tz-naive and
    # tz-aware timestamps").
    cache_key = None
    if universe is None:
        cache_key = (country.upper(), start_year, end_year, cache_dir, cache_ttl_days)
        hit = _MACRO_FETCH_CACHE.get(cache_key)
        if hit is not None and (time.time() - hit[0]) < _MACRO_FETCH_CACHE_TTL_SEC:
            return hit[1].copy(), hit[2]

    universe = universe or CREDIT_MACRO_UNIVERSE
    iso3 = _to_iso3(country)
    iso2 = country.upper()[:2] if len(country) == 2 else next(
        (k for k, v in ISO2_TO_ISO3.items() if v == iso3), iso3[:2]
    )
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    report = FetchReport(
        country_iso2=iso2,
        country_iso3=iso3,
        n_indicators_requested=len(universe),
        n_indicators_fetched=0,
    )

    series_dict: Dict[str, pd.Series] = {}

    LOG.info("Fetching %d macro indicators for %s (ISO3=%s)...",
             len(universe), iso2, iso3)
    current_year = datetime.now().year

    for ind in universe:
        s, source = _fetch_indicator(ind, iso2, iso3, cache_path, cache_ttl_days)
        if s is None or len(s.dropna()) < 5:
            report.missing.append(ind.name)
            report.sources[ind.name] = "none"
            continue
        series_dict[ind.name] = s
        report.sources[ind.name] = source
        report.n_indicators_fetched += 1
        # Detect WEO projections (future years > current_year)
        if source in ("weo", "cache_weo") and s.index.max() > current_year:
            report.has_projections = True
            report.projections_until = max(
                report.projections_until or 0, int(s.index.max())
            )

    if not series_dict:
        LOG.error("No macro data fetched for %s. Check country code and network.", iso2)
        empty_df = pd.DataFrame()
        if cache_key is not None:
            _MACRO_FETCH_CACHE[cache_key] = (time.time(), empty_df, report)
        return empty_df.copy(), report

    # ── Build the unified DataFrame ──────────────────────────────────────
    df = pd.DataFrame(series_dict)
    df.index.name = "year"
    df = df.sort_index()

    # Apply date filtering
    if end_year is None:
        end_year_eff = int(df.index.max())
    else:
        end_year_eff = end_year
    df = df.loc[(df.index >= start_year) & (df.index <= end_year_eff)]

    report.historical_period = (int(df.index.min()), int(df.index.max()))
    LOG.info("Fetched %d/%d indicators, %d-%d (%d rows). Projections: %s",
             report.n_indicators_fetched, report.n_indicators_requested,
             report.historical_period[0], report.historical_period[1],
             len(df),
             f"yes (until {report.projections_until})" if report.has_projections else "no")

    if cache_key is not None:
        _MACRO_FETCH_CACHE[cache_key] = (time.time(), df, report)
    return df.copy(), report


def split_historical_projection(
    df: pd.DataFrame,
    last_observed_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the WEO output into historical and IMF-projected parts.

    Anything strictly after last_observed_year is treated as IMF projection.
    Used by the wrapper to:
      • train satellite models on historical only
      • use WEO projections to inform / validate the AR/VAR baseline
    """
    hist = df.loc[df.index <= last_observed_year].copy()
    proj = df.loc[df.index >  last_observed_year].copy()
    return hist, proj


# ═══════════════════════════════════════════════════════════════════════════
# 6. UNIVERSE INDEX (helper for the wrapper)
# ═══════════════════════════════════════════════════════════════════════════
def universe_index(
    universe: Optional[List[MacroIndicator]] = None,
) -> Dict[str, MacroIndicator]:
    """Returns {var_name: MacroIndicator} for fast lookup in the wrapper."""
    universe = universe or CREDIT_MACRO_UNIVERSE
    return {ind.name: ind for ind in universe}
