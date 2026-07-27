"""
geo_selector.py
===============
Filter the tidy NGFS frame by country, with automatic fallback to a region
aggregate (e.g., 'Africa', 'Asia', 'Developing Europe') when the country
itself is not present.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from .utils import EngineConfig, get_logger, normalize_label

log = get_logger(__name__)


# Default country -> NGFS aggregate region mapping. Users can override via
# EngineConfig.region_map. Only used when the country is not directly available
# in the NGFS file.
_DEFAULT_REGION_MAP = {
    # Europe (EU28-ish + others)
    "Tunisia": "Africa", "Morocco": "Africa", "Egypt": "Africa",
    "Algeria": "Africa", "Nigeria": "Africa", "Kenya": "Africa",
    "South Africa": "Africa",
    "Vietnam": "Asia", "Thailand": "Asia", "Singapore": "Asia",
    "Philippines": "Asia", "Malaysia": "Asia",
    "Romania": "Developing Europe", "Bulgaria": "Developing Europe",
    "Ukraine": "Developing Europe",
    # add more as needed; user can extend via cfg.region_map
}

# CT (GEM-E3) uses different region names — more granular R5 aggregates
_CT_REGION_MAP = {
    "Egypt": "Middle East & Africa (R5)",
    "Tunisia": "Middle East & Africa (R5)",
    "Morocco": "Middle East & Africa (R5)",
    "Algeria": "Middle East & Africa (R5)",
    "Nigeria": "Africa",
    "Kenya": "Africa",
    "South Africa": "Africa",
    "Vietnam": "Asia",
    "Thailand": "Asia",
    "Singapore": "Asia",
    "Philippines": "Asia",
    "Malaysia": "Asia",
    "Romania": "Rest of Europe",
    "Bulgaria": "Rest of Europe",
    "Ukraine": "Rest of Europe",
    # add more as needed; user can extend via cfg.region_map
}


def select_geography(ngfs_long: pd.DataFrame,
                     cfg: EngineConfig) -> Tuple[pd.DataFrame, str]:
    """Filter the NGFS frame to the requested country (or its region fallback).

    Returns
    -------
    (filtered_frame, resolved_region_label)
        resolved_region_label is the actual region used (the country itself
        if available, otherwise the aggregate region).
    """
    if not cfg.country:
        raise ValueError("EngineConfig.country must be set.")

    available = set(ngfs_long["region"].unique())
    norm_avail = {normalize_label(r): r for r in available}

    # 1) try direct country match (case-insensitive, punctuation-insensitive)
    key = normalize_label(cfg.country)
    if key in norm_avail:
        chosen = norm_avail[key]
        log.info("Geography: '%s' found directly in NGFS file as '%s'.",
                 cfg.country, chosen)
        return ngfs_long[ngfs_long["region"] == chosen].copy(), chosen

    # 2) try region map fallback — use CT-specific regions when in CT mode
    base_map = _DEFAULT_REGION_MAP.copy()
    if getattr(cfg, "ngfs_mode", "LT") == "CT":
        base_map.update(_CT_REGION_MAP)
    region_map = {**base_map, **(cfg.region_map or {})}
    fallback = region_map.get(cfg.country)
    if fallback and normalize_label(fallback) in norm_avail:
        chosen = norm_avail[normalize_label(fallback)]
        log.warning("Geography: country '%s' not in NGFS file. "
                    "Falling back to region '%s'.", cfg.country, chosen)
        return ngfs_long[ngfs_long["region"] == chosen].copy(), chosen

    # 3) no match
    raise ValueError(
        f"Country '{cfg.country}' not found in NGFS file and no region "
        f"fallback available. Available regions sample: "
        f"{sorted(available)[:15]}..."
    )
