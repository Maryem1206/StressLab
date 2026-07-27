"""
yield_curve_loader.py
=====================
Load the yield curve Excel file and align with macro API data.

Produces
--------
YieldCurveData
    yield_df       -- monthly DataFrame (date x maturity columns, in %)
    macro_df       -- monthly DataFrame (date x macro variables, aligned)
    portfolio_meta -- dict {tbills_egp_bn, tbonds_egp_bn} from Portfolio_CIB sheet
    country, start_date, end_date

Monthly policy rate
-------------------
The generic CREDIT_MACRO_UNIVERSE carries annual "policy_rate" (WB FR.INR.DPST).
For higher-frequency monthly data, a per-country FRED series is fetched here and
appended as "policy_rate_monthly".  Add entries to POLICY_RATE_FRED_BY_COUNTRY for
any new country -- no change to the generic macro engine is required.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..core.utils import FrequencyAligner
from ..core.imf_weo_fetcher import fetch_credit_macro
from ..core.fred_fetcher import fetch_fred_series as _fetch_fred_with_key


def fetch_fred_series(series_id: str):
    """Fetch via API-key-authenticated FRED fetcher.

    Converts PeriodIndex → DatetimeIndex (month-start) so the result can be
    joined with the annual macro DataFrame in _load_macro.
    Returns None (not empty Series) on failure so existing guard `is not None`
    still works.
    """
    s = _fetch_fred_with_key(series_id)
    if s.empty:
        return None
    # Convert PeriodIndex('M') to DatetimeIndex (month-start timestamps)
    if hasattr(s.index, 'to_timestamp'):
        s.index = s.index.to_timestamp(how='start')
    return s

LOG = logging.getLogger("market.yield_curve_loader")

# ---------------------------------------------------------------------------
# Full country name → ISO2 normalizer
# The UI sends display names (e.g. "Egypt") but WB/IMF fetchers require ISO codes.
# ---------------------------------------------------------------------------
_FULL_NAME_TO_ISO2: Dict[str, str] = {
    "egypt":          "EG",
    "égypte":         "EG",
    "france":         "FR",
    "germany":        "DE",
    "allemagne":      "DE",
    "united kingdom": "GB",
    "royaume-uni":    "GB",
    "morocco":        "MA",
    "maroc":          "MA",
    "tunisia":        "TN",
    "tunisie":        "TN",
    "algeria":        "DZ",
    "algérie":        "DZ",
    "nigeria":        "NG",
    "south africa":   "ZA",
    "afrique du sud": "ZA",
    "kenya":          "KE",
    "turkey":         "TR",
    "turquie":        "TR",
    "brazil":         "BR",
    "brésil":         "BR",
    "mexico":         "MX",
    "mexique":        "MX",
    "india":          "IN",
    "inde":           "IN",
    "usa":            "US",
    "united states":  "US",
    "états-unis":     "US",
    # Regional fallbacks
    "mena":           "EG",
    "north africa":   "MA",
    "afrique du nord":"MA",
    "gcc":            "SA",
    "europe":         "FR",
}


def _normalize_country(code: str) -> str:
    """Convert full country name or any case ISO code to uppercase ISO2/ISO3."""
    code = code.strip()
    iso2 = _FULL_NAME_TO_ISO2.get(code.lower())
    if iso2:
        LOG.info("country '%s' normalized to ISO2 '%s'", code, iso2)
        return iso2
    return code.upper()


# ---------------------------------------------------------------------------
# Per-country FRED series for the central-bank policy / overnight rate (monthly)
# Source: https://fred.stlouisfed.org  -- no API key required
# Add any country that has a reliable FRED monthly series here.
# The fetched series is appended as "policy_rate_monthly" in macro_df.
# ---------------------------------------------------------------------------
POLICY_RATE_FRED_BY_COUNTRY: Dict[str, str] = {
    # Africa / MENA
    # Egypt: no monthly CBE policy rate series available on FRED (2026-06) —
    # satellite uses WB annual lending_rate + policy_rate as fallback.
    # "EG":  None,
    "EG":  "",
    "EGY": "",
    "MA":  "INTDSMAM193N",    # Morocco: Bank Al-Maghrib rate
    "MAR": "INTDSMAM193N",
    "NG":  "INTDSNIAM193N",   # Nigeria: CBN monetary policy rate
    "NGA": "INTDSNIAM193N",
    "ZA":  "INTDSRZAM193N",   # South Africa: SARB repo rate
    "ZAF": "INTDSRZAM193N",
    "KE":  "INTDSKEA193N",    # Kenya: CBK central bank rate
    "KEN": "INTDSKEA193N",
    "TN":  "INTDSTNM193N",    # Tunisia: BCT key rate
    "TUN": "INTDSTNM193N",
    # Emerging markets
    "TR":  "INTRTRTM193N",    # Turkey: CBRT policy rate
    "TUR": "INTRTRTM193N",
    "BR":  "INTDSBRZM193N",   # Brazil: SELIC rate
    "BRA": "INTDSBRZM193N",
    "MX":  "INTDSMXM193N",    # Mexico: Banxico rate
    "MEX": "INTDSMXM193N",
    "IN":  "INTDSINN193N",    # India: RBI repo rate
    "IND": "INTDSINN193N",
    # Advanced economies
    "US":  "FEDFUNDS",        # USA: effective fed funds rate
    "USA": "FEDFUNDS",
    "GB":  "IUDSOIA",         # UK: Bank Rate
    "GBR": "IUDSOIA",
    "EU":  "ECBDFR",          # Euro area: ECB deposit facility rate
    "EMU": "ECBDFR",
}

# Recognised maturity column names (sheet may use a subset)
MATURITY_COLS: List[str] = [
    "T91j", "T182j", "T273j", "T364j", "T3Y", "T5Y", "T10Y"
]

# Macro variables required by the market module
# "policy_rate_monthly" is added dynamically if a FRED series is available.
MARKET_MACRO_VARS: List[str] = [
    "policy_rate",          # annual WB FR.INR.DPST -- generic baseline
    "lending_rate",         # annual WB FR.INR.LEND -- tracks CBE rate cycle closely
    "cpi_inflation",
    "real_gdp_growth",
    "fiscal_balance_gdp",
    "gov_debt_gdp",
    "fx_reserves_usd",
]


@dataclass
class YieldCurveData:
    """Container for the aligned yield-curve + macro dataset."""
    yield_df:       pd.DataFrame   # monthly, cols = maturity labels, values in %
    macro_df:       pd.DataFrame   # monthly, cols = MARKET_MACRO_VARS + policy_rate_monthly?
    portfolio_meta: Dict           # {tbills_egp_bn, tbonds_egp_bn, ...}
    country:        str
    start_date:     pd.Timestamp
    end_date:       pd.Timestamp


class YieldCurveLoader:
    """Load and monthly-align yield curve + macro data.

    Parameters
    ----------
    excel_path : str or Path
        Resolved path to the yield curve Excel file.
    country : str
        ISO2 or ISO3 country code.  Used for WB/IMF fetch AND FRED lookup
        in POLICY_RATE_FRED_BY_COUNTRY.
    cache_dir : str
        Directory for macro API cache files.
    cache_ttl_days : int
        Cache TTL for macro API calls.
    """

    def __init__(
        self,
        excel_path: str | Path,
        country: str = "EG",
        cache_dir: str = "data_cache",
        cache_ttl_days: int = 30,
    ) -> None:
        self.excel_path     = Path(excel_path)
        self.country        = _normalize_country(country)
        self.cache_dir      = cache_dir
        self.cache_ttl_days = int(cache_ttl_days)

    # ---- public --------------------------------------------------------

    def load(self) -> YieldCurveData:
        yield_raw, portfolio_meta = self._read_excel()
        macro_annual             = self._fetch_macro()
        yield_df, macro_df       = self._align(yield_raw, macro_annual)

        return YieldCurveData(
            yield_df       = yield_df,
            macro_df       = macro_df,
            portfolio_meta = portfolio_meta,
            country        = self.country,
            start_date     = yield_df.index[0],
            end_date       = yield_df.index[-1],
        )

    # ---- private -------------------------------------------------------

    def _read_excel(self) -> Tuple[pd.DataFrame, Dict]:
        LOG.info("Reading yield-curve file: %s (sheets: %s)",
                 self.excel_path.name,
                 pd.ExcelFile(self.excel_path).sheet_names)
        xl = pd.ExcelFile(self.excel_path)
        LOG.info("Sheets detected: %s", xl.sheet_names)

        # ---- Yield_Curve sheet ----------------------------------------
        sheet = self._find_sheet(
            xl.sheet_names,
            ["Yield_Curve", "yield", "curve", "taux", "rate", "courbe"],
            required=True,
        )
        raw = xl.parse(sheet)
        raw.columns = [str(c).strip() for c in raw.columns]

        date_col = next(
            (c for c in raw.columns if "date" in c.lower()), None
        )
        if date_col is None:
            date_col = raw.columns[0]
        raw = raw.rename(columns={date_col: "date"})
        raw["date"] = pd.to_datetime(raw["date"], dayfirst=False, errors="coerce")
        raw = raw.dropna(subset=["date"]).set_index("date").sort_index()

        present = [c for c in MATURITY_COLS if c in raw.columns]
        if not present:
            raise ValueError(
                f"No maturity columns found in sheet '{sheet}'. "
                f"Expected subset of {MATURITY_COLS}. Found: {list(raw.columns)}"
            )
        yield_df = raw[present].astype(float)
        LOG.info(
            "Yield curve: %d months x %d maturities (%s)",
            len(yield_df), len(present), present,
        )

        # ---- Portfolio_CIB sheet (key-value) ---------------------------
        portfolio_meta: Dict = {}
        port_sheet = self._find_sheet(
            xl.sheet_names,
            [
                "Portfolio", "CIB", "portfolio", "portefeuille",
                "ptf", "exposition", "exposure", "holding",
                "position", "banque", "bank",
            ],
            required=False,
        )
        if port_sheet:
            df_port = xl.parse(port_sheet, header=None)
            raw_kv: Dict[str, float] = {}
            for _, row in df_port.iterrows():
                if len(row) >= 2 and pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
                    raw_key = str(row.iloc[0]).strip().lower()
                    norm_key = raw_key.replace(" ", "_").replace("-", "_")
                    raw_val = row.iloc[1]
                    try:
                        if isinstance(raw_val, str):
                            # Remove thousands separators (, or space or .) then parse
                            raw_val = raw_val.strip().replace(",", "").replace(" ", "").replace("\xa0", "")
                        raw_kv[norm_key] = float(raw_val)
                        LOG.debug("Portfolio kv: %s = %s", norm_key, raw_kv[norm_key])
                    except (ValueError, TypeError) as e:
                        LOG.debug("Portfolio skip row '%s': cannot parse '%s' (%s)", norm_key, raw_val, e)

            # -- Fuzzy lookup for tbills / tbonds (accept any key containing the token) --
            # Excel values are in EGP thousands → divide by 1 000 000 to get Md EGP (billions)
            _K_EGP_TO_BN = 1_000_000.0
            for k, v in raw_kv.items():
                if any(tok in k for tok in ("tbill", "t_bill", "billet", "bon_court", "bills")):
                    portfolio_meta.setdefault("tbills_egp_bn", v / _K_EGP_TO_BN)
                elif any(tok in k for tok in ("tbond", "t_bond", "obligation", "bon_long", "bonds")):
                    portfolio_meta.setdefault("tbonds_egp_bn", v / _K_EGP_TO_BN)
                else:
                    portfolio_meta[k] = v

            LOG.info(
                "Portfolio sheet '%s': raw keys=%s → tbills=%.4f Md EGP  tbonds=%.4f Md EGP  (converted from K EGP)",
                port_sheet,
                list(raw_kv.keys()),
                portfolio_meta.get("tbills_egp_bn", float("nan")),
                portfolio_meta.get("tbonds_egp_bn", float("nan")),
            )
        else:
            LOG.warning(
                "No portfolio sheet found (checked: %s) -- "
                "tbills_egp_bn / tbonds_egp_bn must be provided via params",
                xl.sheet_names,
            )

        return yield_df, portfolio_meta

    def _fetch_macro(self) -> pd.DataFrame:
        """Fetch generic annual macro + optional monthly policy rate from FRED."""
        LOG.info("Fetching macro data for country=%s", self.country)

        # --- 1. Generic annual series (WB / IMF WEO) ---
        try:
            df, _ = fetch_credit_macro(
                country        = self.country,
                start_year     = 2005,
                cache_dir      = self.cache_dir,
                cache_ttl_days = self.cache_ttl_days,
            )
        except Exception as exc:
            LOG.error("Macro fetch failed: %s -- returning empty DataFrame", exc)
            return pd.DataFrame()

        if df.empty:
            LOG.warning("Macro fetch returned empty DataFrame for %s", self.country)
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index.astype(str) + "-01-01")
        df = df.sort_index()

        keep    = [v for v in MARKET_MACRO_VARS if v in df.columns]
        missing = [v for v in MARKET_MACRO_VARS if v not in df.columns]
        if missing:
            LOG.warning("Market macro vars not fetched: %s", missing)
        result = df[keep].copy() if keep else pd.DataFrame(index=df.index)

        # --- 2. Monthly policy rate from FRED (country-specific) ---
        fred_code = POLICY_RATE_FRED_BY_COUNTRY.get(self.country.upper())
        if fred_code:
            LOG.info(
                "Fetching monthly policy rate for %s from FRED series %s",
                self.country, fred_code,
            )
            try:
                monthly_rate = fetch_fred_series(fred_code)
                if monthly_rate is not None and len(monthly_rate) >= 12:
                    monthly_rate.name = "policy_rate_monthly"
                    result = result.join(monthly_rate.rename("policy_rate_monthly"),
                                         how="outer")
                    LOG.info(
                        "policy_rate_monthly added: %d monthly observations (FRED %s)",
                        int(monthly_rate.notna().sum()), fred_code,
                    )
                else:
                    LOG.warning(
                        "FRED series %s returned insufficient data -- "
                        "policy_rate_monthly not added", fred_code,
                    )
            except Exception as exc:
                LOG.warning(
                    "FRED fetch for %s failed: %s -- policy_rate_monthly not added",
                    fred_code, exc,
                )
        else:
            LOG.info(
                "No FRED policy rate series configured for country=%s -- "
                "using annual WB policy_rate only", self.country,
            )

        return result

    def _align(
        self,
        yield_df: pd.DataFrame,
        macro_annual: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align all series to the monthly index of the yield curve."""
        monthly_idx = pd.date_range(
            start = yield_df.index[0].to_period("M").to_timestamp(),
            end   = yield_df.index[-1].to_period("M").to_timestamp(),
            freq  = "MS",
        )

        # Yield curve is already monthly-observed: a short forward-fill (2 months)
        # bridges the occasional missed auction, but a multi-year reporting gap
        # (e.g. a tenor unquoted during a crisis) must stay NaN rather than be
        # silently frozen at its last pre-gap value -- freezing would corrupt
        # the NS fit and satellite calibration for that tenor across the gap.
        yield_aligner = FrequencyAligner(monthly_idx, ffill_limit=2)
        yield_aligned = yield_aligner.align(
            {col: yield_df[col] for col in yield_df.columns}
        )

        # Macro series (annual/quarterly) are intentionally broadcast across
        # every month of their reporting period -- unlimited ffill is correct here.
        macro_aligner = FrequencyAligner(monthly_idx)
        if macro_annual.empty:
            macro_aligned = pd.DataFrame(index=monthly_idx)
        else:
            macro_aligned = macro_aligner.align(
                {col: macro_annual[col] for col in macro_annual.columns}
            )

        LOG.info(
            "Alignment done: %d months | yield cols=%d | macro cols=%d",
            len(monthly_idx),
            len(yield_aligned.columns),
            len(macro_aligned.columns),
        )
        return yield_aligned, macro_aligned

    @staticmethod
    def _find_sheet(
        sheet_names: List[str],
        keywords:    List[str],
        required:    bool = True,
    ) -> Optional[str]:
        for name in sheet_names:
            if any(kw.lower() in name.lower() for kw in keywords):
                return name
        if required:
            raise ValueError(
                f"Cannot find sheet matching {keywords} among: {sheet_names}"
            )
        return None