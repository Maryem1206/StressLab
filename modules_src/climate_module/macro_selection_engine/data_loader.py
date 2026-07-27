"""
data_loader.py
==============
Load NGFS NiGEM Phase 5 files and user-provided historical files.

Fix v3 : filtre par pays ET par scénarios AVANT le melt.
Le melt sur le fichier complet (200 pays × 7 scénarios × 100 variables
× 29 années) produit ~40 millions de lignes → MemoryError sur machines
avec RAM limitée. Filtrer avant le melt réduit le volume d'un facteur ~470.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from statistics import mode as stat_mode
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import EngineConfig, get_logger

log = get_logger(__name__)

_KNOWN_CHANNELS = ("combined(no bus)", "combined", "physical", "transition")

# ─────────────────────────────────────────────────────────────────────
# Cached Excel reader — NGFS/GEM-E3 files run 20-50 MB and pandas'
# openpyxl engine takes 30-75s to parse them (measured: ~75s for a
# 300k-row GEM-E3 CT file), which used to be the dominant cost of every
# single Phase 1 run. The scenario data doesn't change between re-runs
# against the same upload, so re-parsing the XML every time was pure
# waste. Cached in-memory (this worker process) and on disk as Parquet
# (keyed by source mtime) so repeat runs — including after a server
# restart — skip re-parsing the Excel file.
# ─────────────────────────────────────────────────────────────────────
_EXCEL_READ_CACHE: dict = {}


def _read_excel_cached(path: str, sheet_name) -> pd.DataFrame:
    src = Path(path)
    mtime = src.stat().st_mtime
    key = (str(src), mtime, str(sheet_name))

    cached = _EXCEL_READ_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    parquet_path = src.with_name(src.name + ".cache.parquet")
    if parquet_path.exists() and parquet_path.stat().st_mtime >= mtime:
        try:
            df = pd.read_parquet(parquet_path)
            _EXCEL_READ_CACHE[key] = df
            log.info("Loaded from parquet cache: %s", parquet_path.name)
            return df.copy()
        except Exception as exc:
            log.warning("Parquet cache unreadable (%s) -- re-reading Excel.", exc)

    df = pd.read_excel(path, sheet_name=sheet_name)
    try:
        fd, tmp_path = tempfile.mkstemp(
            suffix=".parquet", dir=str(parquet_path.parent))
        os.close(fd)
        df.to_parquet(tmp_path)
        os.replace(tmp_path, parquet_path)
    except Exception as exc:
        log.warning("Could not write parquet cache for %s (%s).", src.name, exc)

    _EXCEL_READ_CACHE[key] = df
    return df.copy()

_DEFAULT_REGION_MAP = {
    "Tunisia": "Africa", "Morocco": "Africa",
    "Algeria": "Africa", "Egypt": "Africa",
    "Nigeria": "Africa", "Kenya": "Africa",
    "South Africa": "Africa",
    "Vietnam": "Asia", "Thailand": "Asia",
    "Romania": "Developing Europe", "Bulgaria": "Developing Europe",
}


def _normalize_ngfs_units(long: pd.DataFrame, baseline_scenario: str = "Baseline") -> pd.DataFrame:
    """
    Normalise toutes les déviations NGFS en Abs. difference.

    Le fichier NGFS NiGEM contient deux types d'unités pour les scénarios stressés :

      "Abs. difference"        → déviation absolue (ex: +0.54 pp de taux d'intérêt)
                                 → garder tel quel

      "% difference[, ...]"    → déviation relative en % du niveau Baseline
                                 (ex: -6.2% du PIB Baseline)
                                 → convertir : abs_diff(t) = baseline_level(t) × pct_diff(t) / 100

    Le scénario Baseline n'est pas modifié — il contient les niveaux absolus de référence.

    Paramètres
    ----------
    long              : DataFrame NGFS tidy (output de load_ngfs avant ce call)
    baseline_scenario : nom exact du scénario Baseline dans le fichier

    Retourne
    --------
    DataFrame avec les valeurs `% difference` converties en `Abs. difference`.
    La colonne `unit` est mise à jour pour refléter la conversion.
    """
    # Séparer Baseline (niveaux) et scénarios stressés (déviations)
    is_baseline = long["scenario"] == baseline_scenario
    baseline_df = long[is_baseline].copy()
    stressed_df = long[~is_baseline].copy()

    if baseline_df.empty:
        log.warning(
            "_normalize_ngfs_units: scénario '%s' introuvable — "
            "conversion %% difference impossible, données inchangées.",
            baseline_scenario,
        )
        return long

    # Identifier les lignes % difference dans les scénarios stressés
    is_pct = stressed_df["unit"].str.startswith("% difference", na=False)
    n_pct  = int(is_pct.sum())
    n_abs  = int((~is_pct & stressed_df["unit"].str.startswith("Abs.", na=False)).sum())

    if n_pct == 0:
        log.info(
            "_normalize_ngfs_units: aucune ligne '%% difference' trouvée "
            "(%d lignes 'Abs. difference'). Rien à convertir.",
            n_abs,
        )
        return long

    log.info(
        "_normalize_ngfs_units: %d lignes 'Abs. difference' (inchangées) | "
        "%d lignes '%% difference' à convertir en abs.",
        n_abs, n_pct,
    )

    # Construire la table de référence : (region, variable_base, year) → baseline_level
    # Le Baseline n'a pas de channel → channel = None
    baseline_levels = (
        baseline_df[baseline_df["channel"].isna()]
        .groupby(["region", "variable_base", "year"])["value"]
        .mean()  # moyenne si plusieurs modèles
    )

    # Appliquer la conversion sur les lignes % difference
    stressed_pct = stressed_df[is_pct].copy()

    def _convert_row(row):
        key = (row["region"], row["variable_base"], row["year"])
        if key in baseline_levels.index:
            baseline_val = baseline_levels[key]
            if pd.notna(baseline_val) and pd.notna(row["value"]):
                return baseline_val * row["value"] / 100.0
        # Fallback : pas de niveau Baseline trouvé → garder la valeur
        return row["value"]

    # Vectorisé via merge (plus rapide que apply row-by-row)
    baseline_ref = baseline_levels.reset_index().rename(columns={"value": "baseline_level"})

    stressed_pct = stressed_pct.merge(
        baseline_ref,
        on=["region", "variable_base", "year"],
        how="left",
    )

    # Conversion : abs_diff = baseline_level × pct_diff / 100
    converted_mask = stressed_pct["baseline_level"].notna() & stressed_pct["value"].notna()
    stressed_pct.loc[converted_mask, "value"] = (
        stressed_pct.loc[converted_mask, "baseline_level"]
        * stressed_pct.loc[converted_mask, "value"]
        / 100.0
    )

    n_converted  = int(converted_mask.sum())
    n_no_baseline = int((~converted_mask).sum())
    if n_no_baseline > 0:
        log.warning(
            "_normalize_ngfs_units: %d lignes '%% difference' sans niveau Baseline "
            "correspondant → valeurs inchangées (vérifier la cohérence du fichier).",
            n_no_baseline,
        )

    # Mettre à jour l'unité → "Abs. difference (converted from % diff)"
    stressed_pct.loc[converted_mask, "unit"] = "Abs. difference"

    # Supprimer la colonne temporaire baseline_level
    stressed_pct = stressed_pct.drop(columns=["baseline_level"])

    log.info(
        "_normalize_ngfs_units: %d/%d lignes '%%difference' converties en Abs. difference.",
        n_converted, n_pct,
    )

    # Reconstruire le DataFrame complet
    stressed_abs = stressed_df[~is_pct].copy()
    result = pd.concat([baseline_df, stressed_abs, stressed_pct], ignore_index=True)
    return result


def _split_channel(variable_name: str) -> Tuple[str, Optional[str]]:
    s = str(variable_name).strip()
    m = re.search(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*$", s)
    if not m:
        return s, None
    inner = m.group(1).strip()
    if inner in _KNOWN_CHANNELS:
        base = s[: m.start()].strip().rstrip(";").strip()
        return base, inner
    return s, None


def _parse_region(region_str: str) -> str:
    if "|" in str(region_str):
        return str(region_str).split("|", 1)[1].strip()
    return str(region_str).strip()


def load_ngfs(cfg: EngineConfig,
              scenarios: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Charge et structure le fichier NGFS NiGEM Phase 5.

    Optimisation mémoire (v3)
    -------------------------
    Filtre par pays et scénarios AVANT le melt pour éviter MemoryError.

    Parameters
    ----------
    cfg       : EngineConfig
    scenarios : scénarios à charger. Si None, charge tous les scénarios.
                Passez toujours la liste des scénarios nécessaires :
                ["Baseline", "Delayed transition", "Fragmented World"]
    """
    log.info("Loading NGFS file: %s", cfg.ngfs_path)
    df = _read_excel_cached(cfg.ngfs_path, cfg.ngfs_sheet)

    required = {
        cfg.ngfs_col_model, cfg.ngfs_col_scenario, cfg.ngfs_col_region,
        cfg.ngfs_col_variable, cfg.ngfs_col_unit,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"NGFS file missing required columns: {missing}")

    # ── Identifier les colonnes années ───────────────────────────────
    year_cols = [
        c for c in df.columns
        if str(c).strip().isdigit() and 1900 <= int(str(c).strip()) <= 2200
    ]
    if not year_cols:
        raise ValueError("No year columns detected in NGFS file.")
    log.info("Detected %d year columns: %s..%s",
             len(year_cols), year_cols[0], year_cols[-1])

    # ── FILTRE 1 : par pays/région (AVANT melt) ──────────────────────
    if cfg.country:
        region_col = cfg.ngfs_col_region
        region_mask = df[region_col].astype(str).str.contains(
            cfg.country, case=False, na=False, regex=False
        )
        if region_mask.any():
            df = df[region_mask].copy()
            log.info("Pre-filter by country '%s': %d rows kept.",
                     cfg.country, len(df))
        else:
            # Fallback vers région agrégée
            region_map = {**_DEFAULT_REGION_MAP, **(cfg.region_map or {})}
            fallback = region_map.get(cfg.country)
            if fallback:
                region_mask = df[region_col].astype(str).str.contains(
                    fallback, case=False, na=False, regex=False
                )
                if region_mask.any():
                    df = df[region_mask].copy()
                    log.info(
                        "Pre-filter: country '%s' → region '%s': %d rows.",
                        cfg.country, fallback, len(df)
                    )
                else:
                    log.warning(
                        "Pre-filter: '%s' and '%s' not found. "
                        "Loading all regions — this may be slow.",
                        cfg.country, fallback
                    )
            else:
                log.warning(
                    "Pre-filter: country '%s' not found in file. "
                    "Loading all regions — this may be slow.",
                    cfg.country
                )

    # ── FILTRE 2 : par scénarios (AVANT melt) ────────────────────────
    if scenarios:
        scn_mask = df[cfg.ngfs_col_scenario].isin(scenarios)
        n_before = len(df)
        df = df[scn_mask].copy()
        log.info("Pre-filter by %d scenarios: %d → %d rows.",
                 len(scenarios), n_before, len(df))

    if df.empty:
        raise ValueError(
            f"No rows after pre-filtering for country='{cfg.country}' "
            f"and scenarios={scenarios}. "
            f"Check your EngineConfig.country and scenario names."
        )

    log.info("Melting %d rows × %d year columns ...", len(df), len(year_cols))

    # ── Melt (sur le sous-ensemble filtré uniquement) ─────────────────
    id_cols = [
        cfg.ngfs_col_model, cfg.ngfs_col_scenario,
        cfg.ngfs_col_region, cfg.ngfs_col_variable, cfg.ngfs_col_unit,
    ]
    long = df.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    long["year"] = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    long = long.rename(columns={
        cfg.ngfs_col_model:    "model",
        cfg.ngfs_col_scenario: "scenario",
        cfg.ngfs_col_region:   "region_raw",
        cfg.ngfs_col_variable: "variable_full",
        cfg.ngfs_col_unit:     "unit",
    })

    # ── Parsing canal + région (list comprehensions, jamais Series.map) ──
    variable_list = long["variable_full"].tolist()
    region_list   = long["region_raw"].tolist()

    parsed = [_split_channel(v) for v in variable_list]
    long["variable_base"] = [t[0] for t in parsed]
    long["channel"]       = [t[1] for t in parsed]
    long["region"]        = [_parse_region(r) for r in region_list]

    log.info(
        "NGFS tidy: %d rows | %d vars | %d scenarios | %d regions",
        len(long),
        long["variable_base"].nunique(),
        long["scenario"].nunique(),
        long["region"].nunique(),
    )

    return long[[
        "model", "scenario", "region", "variable_base",
        "channel", "unit", "year", "value",
    ]]


# ─────────────────────────────────────────────────────────────────────
# multi_scenario.py doit passer scenarios= à load_ngfs.
# Fonction helper exportée pour que multi_scenario puisse l'utiliser.
# ─────────────────────────────────────────────────────────────────────

def load_ngfs_for_stress_test(cfg: EngineConfig,
                               scenario_map: dict) -> pd.DataFrame:
    """
    Charge le fichier NGFS en chargeant TOUS les scénarios disponibles.
    Le filtrage adverse/severe est fait en aval par le scenario_selector.

    Dispatch automatique LT (NiGEM) ou CT (GEM-E3) selon cfg.ngfs_mode.
    """
    if cfg.ngfs_mode == "CT":
        return load_ngfs_ct(cfg, scenario_map=None)
    return load_ngfs(cfg, scenarios=None)


# ─────────────────────────────────────────────────────────────────────
# CT (short-term) loader — GEM-E3 format IAMC
# ─────────────────────────────────────────────────────────────────────

def load_ngfs_ct(cfg: EngineConfig,
                 scenario_map: Optional[dict] = None) -> pd.DataFrame:
    """
    Charge et normalise un fichier GEM-E3 NGFS Short-Term (IAMC format).
    Si scenario_map=None, charge TOUS les scénarios disponibles.

    Format GEM-E3 :
        Model | Scenario | Region | Variable | Unit | 2021 | ... | 2030
        - Annuel, niveaux absolus
        - Pas de suffixe channel (combined/physical/transition)
        - Régions : noms directs ("Africa", "Middle East & Africa (R5)", etc.)

    Normalise vers le même schéma interne que load_ngfs() :
        model | scenario | region | variable_base | channel | unit | year | value

    Pour le CT, channel = None pour toutes les lignes (pas de risk channel).
    """
    path = cfg.ngfs_ct_path
    if not path:
        raise ValueError("EngineConfig.ngfs_ct_path must be set for CT mode.")

    log.info("Loading NGFS CT file (GEM-E3): %s", path)
    df = _read_excel_cached(path, cfg.ngfs_sheet)

    required = {"Model", "Scenario", "Region", "Variable", "Unit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GEM-E3 file missing columns: {missing}")

    # ── Identifier les colonnes années ────────────────────────────────
    year_cols = [
        c for c in df.columns
        if str(c).strip().isdigit() and 2000 <= int(str(c).strip()) <= 2100
    ]
    if not year_cols:
        raise ValueError("No year columns detected in GEM-E3 file.")
    log.info("CT: %d year columns: %s..%s", len(year_cols), year_cols[0], year_cols[-1])

    # ── Filtre par scénarios (None = tous) ───────────────────────────
    if scenario_map is not None:
        needed = [v for v in scenario_map.values() if v != "__pending__"]
        df = df[df["Scenario"].isin(needed)].copy()
        log.info("CT: filtered to %d scenarios → %d rows.", len(needed), len(df))
    else:
        log.info("CT: loading ALL scenarios → %d rows.", len(df))

    if df.empty:
        raise ValueError("No rows in GEM-E3 file after scenario filter.")

    # ── Filtre par région (pre-filter avant melt) ─────────────────────
    if cfg.country:
        region_col = "Region"
        # Essayer match direct
        mask = df[region_col].astype(str).str.contains(
            cfg.country, case=False, na=False, regex=False
        )
        if not mask.any():
            # Fallback vers région agrégée
            from .geo_selector import _CT_REGION_MAP, _DEFAULT_REGION_MAP
            ct_map = {**_DEFAULT_REGION_MAP, **_CT_REGION_MAP, **(cfg.region_map or {})}
            fallback = ct_map.get(cfg.country)
            if fallback:
                mask = df[region_col].astype(str).str.contains(
                    fallback, case=False, na=False, regex=False
                )
                if mask.any():
                    log.info("CT pre-filter: '%s' → '%s' (%d rows).",
                             cfg.country, fallback, mask.sum())
        if mask.any():
            df = df[mask].copy()
        else:
            log.warning("CT: country '%s' not found. Loading all regions.", cfg.country)

    # ── Melt ──────────────────────────────────────────────────────────
    id_cols = ["Model", "Scenario", "Region", "Variable", "Unit"]
    long = df.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    long["year"]  = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    long = long.rename(columns={
        "Model":    "model",
        "Scenario": "scenario",
        "Region":   "region",
        "Variable": "variable_base",
        "Unit":     "unit",
    })

    # CT : pas de channel (pas de suffixe combined/physical/transition)
    long["channel"] = None

    log.info(
        "CT tidy: %d rows | %d vars | %d scenarios | %d regions",
        len(long),
        long["variable_base"].nunique(),
        long["scenario"].nunique(),
        long["region"].nunique(),
    )

    return long[["model", "scenario", "region", "variable_base",
                  "channel", "unit", "year", "value"]]


# ─────────────────────────────────────────────────────────────────────
# Historical file loader
# ─────────────────────────────────────────────────────────────────────

def _detect_separator(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            counts = {
                ",":  line.count(","),
                ";":  line.count(";"),
                "\t": line.count("\t"),
                "|":  line.count("|"),
            }
            return max(counts, key=counts.get)
    return ","


def _find_header_row(path: str, sep: str) -> int:
    rows = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            fields = line.strip().split(sep)
            n = len([x for x in fields if x.strip()])
            if n > 0:
                rows.append((i, n))
    if not rows:
        return 0
    try:
        target_n = stat_mode([n for _, n in rows])
    except Exception:
        target_n = max(n for _, n in rows)
    for i, n in rows:
        if n == target_n:
            return i
    return 0


def load_historical(cfg: EngineConfig) -> pd.DataFrame:
    """Load user-provided historical macro+target file (wide, annual).
    If no file is provided or file doesn't exist, returns an empty DataFrame
    with just the year column — the Variable Resolver will fill it via API.
    """
    path = cfg.historical_path
    if not path:
        log.info("No historical file configured. Will rely on API resolution.")
        return pd.DataFrame({cfg.hist_year_col: []})

    import os
    if not os.path.exists(path):
        log.warning("Historical file not found: %s. Will rely on API resolution.", path)
        return pd.DataFrame({cfg.hist_year_col: []})

    log.info("Loading historical file: %s", path)

    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
        log.info("Loaded Excel: %d rows × %d cols.", *df.shape)
        return _validate_and_finalize(df, cfg)

    sep = _detect_separator(path)
    log.info("Detected separator: %r", sep)
    skiprows = _find_header_row(path, sep)
    if skiprows > 0:
        log.warning("Skipping %d metadata row(s) before header.", skiprows)

    try:
        df = pd.read_csv(
            path,
            sep=sep,
            skiprows=skiprows,
            encoding="utf-8-sig",
            on_bad_lines="warn",
            engine="python",
        )
    except Exception as e:
        raise ValueError(
            f"Could not parse '{path}'. Details: {e}\n"
            "Tip: save as UTF-8 CSV from Excel (File → Save As → CSV UTF-8)."
        ) from e

    df = df.dropna(how="all").dropna(axis=1, how="all")

    if sep == ";":
        for col in df.columns:
            if df[col].dtype == object:
                cleaned = (
                    df[col].astype(str)
                    .str.replace(r"\s", "", regex=True)
                    .str.replace("%", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
                as_num = pd.to_numeric(cleaned, errors="coerce")
                if as_num.notna().mean() > 0.5:
                    df[col] = as_num

    log.info("Loaded CSV: %d rows × %d cols (sep=%r, skip=%d).",
             df.shape[0], df.shape[1], sep, skiprows)
    return _validate_and_finalize(df, cfg)


# All lowercase aliases — matching done via .lower().strip() comparison
_YEAR_ALIASES = {
    # English
    "year", "yr", "y", "years", "date", "dates", "period", "periods",
    "time", "time_period", "fiscal_year", "fiscal year", "financial_year",
    "financial year", "reporting_year", "reporting year", "obs_year",
    "observation_year", "data_year", "calendar_year", "calendar year",
    # French
    "annee", "année", "an", "ans", "période", "periode",
    "exercice", "exercice_fiscal", "date_observation", "années",
    # Arabic transliteration
    "sana", "sanawat",
}

_PD_ALIASES = [
    # English variants
    "pd_pit", "pd", "pd_thr", "pd_avg",
    "default rate", "default_rate", "default-rate",
    "default ratio", "default_ratio",
    "probability of default", "probability_of_default",
    "prob_default", "prob default", "probability_default",
    "loss_rate", "loss rate", "loss-rate",
    "charge_off", "charge off", "charge-off", "chargeoff",
    "write_off", "write off", "write-off", "writeoff",
    "credit_loss", "credit loss", "credit_losses",
    "nonperforming", "non_performing", "non performing",
    "npls", "npl_ratio", "npl ratio", "npl_rate", "npl rate",
    "npl", "npe_ratio", "npe",
    "impairment_rate", "impairment rate",
    # French variants
    "taux_defaut", "taux défaut", "taux de défaut",
    "taux_défaut", "taux de defaut", "taux defaut",
    "taux_sinistre", "taux sinistre",
    "défaut", "defaut", "sinistralite", "sinistralite",
    "perte_sur_creances", "pertes sur créances",
    "taux_pertes", "taux de pertes",
    # Arabic transliteration
    "maadal_al_ta3athor", "nisbat_al_qard_al_muta3athor",
]


def _find_year_col(df: pd.DataFrame, expected: str) -> str:
    """Return actual column name matching the year concept (case-insensitive + aliases).
    Returns expected unchanged if no match found (caller handles the error).
    """
    cols_norm = {str(c).strip().lower(): str(c).strip() for c in df.columns}
    # 1. Exact match (case-insensitive)
    if expected.lower() in cols_norm:
        return cols_norm[expected.lower()]
    # 2. Alias match
    for alias in _YEAR_ALIASES:
        if alias in cols_norm:
            log.warning(
                "Year column '%s' not found; using '%s' as year column.",
                expected, cols_norm[alias],
            )
            return cols_norm[alias]
    return expected  # triggers ValueError below


def _find_target_column(df: pd.DataFrame, target: str, year_col: str = "year") -> str:
    """Return the actual column name that best matches target, or target if exact match."""
    if target in df.columns:
        return target
    cols_lower = {str(c).strip().lower(): str(c).strip() for c in df.columns}
    target_lower = target.lower().strip()
    if target_lower in cols_lower:
        return cols_lower[target_lower]
    for alias in _PD_ALIASES:
        if alias in cols_lower:
            log.warning(
                "Target column '%s' not found. Using '%s' as substitute.",
                target, cols_lower[alias],
            )
            return cols_lower[alias]
    # Last resort: pick first numeric non-year column
    for col in df.columns:
        if col == year_col:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().mean() > 0.7:
            log.warning(
                "Target column '%s' not found. Using first numeric column '%s' as substitute.",
                target, col,
            )
            return col
    return target  # will trigger the original ValueError below


def _validate_and_finalize(df: pd.DataFrame, cfg: EngineConfig) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]

    # Resolve year column — case-insensitive + exhaustive aliases
    actual_year = _find_year_col(df, cfg.hist_year_col)
    if actual_year not in df.columns:
        raise ValueError(
            f"Historical file must contain a year column (expected '{cfg.hist_year_col}').\n"
            f"Columns found: {list(df.columns)}\n"
            f"Accepted aliases: {sorted(_YEAR_ALIASES)}"
        )
    if actual_year != cfg.hist_year_col:
        df = df.rename(columns={actual_year: cfg.hist_year_col})

    # Auto-detect target column if exact name not found
    actual_target = _find_target_column(df, cfg.target_variable, year_col=cfg.hist_year_col)
    if actual_target not in df.columns:
        raise ValueError(
            f"Historical file must contain the target column "
            f"'{cfg.target_variable}'.\n"
            f"Columns found: {list(df.columns)}"
        )
    if actual_target != cfg.target_variable:
        df = df.rename(columns={actual_target: cfg.target_variable})

    df = df.copy()
    df[cfg.hist_year_col] = (
        pd.to_numeric(df[cfg.hist_year_col], errors="coerce")
        .astype("Int64")
    )
    df = df.dropna(subset=[cfg.hist_year_col])
    df[cfg.hist_year_col] = df[cfg.hist_year_col].astype(int)
    df = df.sort_values(cfg.hist_year_col).reset_index(drop=True)

    # Auto-convert percentage format to proportion (e.g. 2.30 or "2,30%" → 0.023)
    raw_target = df[cfg.target_variable].astype(str).str.strip()
    cleaned_target = (
        raw_target
        .str.replace("%", "", regex=False)
        .str.replace(r"\s", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    target_series = pd.to_numeric(cleaned_target, errors="coerce")
    # Write back the cleaned numeric values so downstream sees floats, not strings
    if target_series.notna().any():
        df[cfg.target_variable] = target_series
    if target_series.notna().any() and (target_series.dropna() > 1).all():
        log.warning(
            "Target column '%s' appears to be in percentage format "
            "(all values > 1, max=%.2f). Dividing by 100 to convert to proportion.",
            cfg.target_variable, float(target_series.max()),
        )
        df[cfg.target_variable] = target_series / 100.0

    log.info("Historical frame ready: %d rows, %d cols (target='%s').",
             len(df), df.shape[1], cfg.target_variable)
    return df
