"""
fred_fetcher.py
===============
Fetch monthly time-series from the FRED API (Federal Reserve Bank of St. Louis).
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("core.fred_fetcher")

# Free API key from https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = "a06fe72025f2e73ebd34da864559af7e"

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_TIMEOUT   = 20


def fetch_fred_series(
    series_id: str,
    start_date: str = "2005-01-01",
    end_date: str | None = None,
    cache_dir: str | None = None,
    cache_ttl_days: int = 30,
    frequency: str = "m",
) -> pd.Series:
    """Download a FRED series and return a monthly pd.Series.

    Returns empty Series if the key is blank or the fetch fails.
    """
    api_key = FRED_API_KEY.strip()
    if not api_key:
        LOG.warning("FRED: FRED_API_KEY not set in fred_fetcher.py — skipping %s", series_id)
        return pd.Series(dtype=float)

    if end_date is None:
        end_date = str(date.today())

    cache_path: Path | None = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"fred_{series_id}.csv"
        if cache_path.exists():
            age_days = (date.today() - date.fromtimestamp(cache_path.stat().st_mtime)).days
            if age_days < cache_ttl_days:
                try:
                    s = _read_cache(cache_path)
                    LOG.info("FRED: loaded %s from cache (%d obs)", series_id, len(s))
                    return s
                except Exception:
                    pass

    params = urllib.parse.urlencode({
        "series_id":         series_id,
        "api_key":           api_key,
        "file_type":         "json",
        "frequency":         frequency,
        "observation_start": start_date,
        "observation_end":   end_date,
        "units":             "lin",
    })
    url = f"{_FRED_BASE}?{params}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:
        LOG.error("FRED: HTTP fetch failed for %s: %s", series_id, exc)
        return pd.Series(dtype=float)

    if "error_message" in payload:
        LOG.error("FRED API error for %s: %s", series_id, payload["error_message"])
        return pd.Series(dtype=float)

    obs = payload.get("observations", [])
    rows = []
    for o in obs:
        val = o.get("value", ".")
        if val == ".":
            continue
        try:
            rows.append({"date": pd.Period(o["date"], freq="M"), "value": float(val)})
        except (ValueError, KeyError):
            continue

    if not rows:
        LOG.warning("FRED: no valid observations for %s", series_id)
        return pd.Series(dtype=float)

    s = pd.Series({r["date"]: r["value"] for r in rows}, dtype=float, name=series_id)
    s.index = pd.PeriodIndex(s.index, freq="M")
    LOG.info("FRED: fetched %s — %d obs  [%.2f, %.2f]", series_id, len(s), s.min(), s.max())

    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            s.to_csv(cache_path, header=True)
        except Exception:
            pass

    return s


def _read_cache(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=False)
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df.iloc[:, 0].astype(float)
