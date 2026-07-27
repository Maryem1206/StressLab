"""
ngfs_liquidity_engine.py
========================
Projette LCR et NSFR sous les scénarios NGFS (LT/CT) en réutilisant
le LiquidityStressEngine existant (satellites + BCBS 238/295).

Pipeline :
  1. Charger Excel liquidité (historique bilan + satellites historiques)
  2. Fetch macro IMF WEO/WB  — même infra que le module crédit
  3. Calibrer les 5 satellites via LiquidityStressEngine.calibrate()
  4. Lire le fichier NGFS (LT/CT) en format tidy
  5. Mapper variable_base NGFS → hist_col IMF WEO via RESOLUTION_RULES
  6. Extraire trajectoires macro par scénario (baseline / adverse / sévère)
  7. compute_stress(ngfs_scenarios) → LCR/NSFR projetés sur horizon NGFS
  8. Retourner résultats structurés pour le dashboard
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

LOG = logging.getLogger("ngfs_liquidity_engine")

# ── sys.path : modules_src/ (PARENT de liquidity_module) + climate_module ────
# On ajoute modules_src/ (et non modules_src/liquidity_module) pour que les
# imports relatifs dans liquidity_stress_engine.py fonctionnent correctement.
_APP_ROOT    = Path(__file__).resolve().parent.parent.parent.parent
_MODULES_SRC = str(_APP_ROOT / "modules_src")          # ex. …/modules_src
_CLM_SRC     = str(_APP_ROOT / "modules_src" / "climate_module")
for _p in (_MODULES_SRC, _CLM_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# MACRO_VARS attendus par LiquidityStressEngine (définis dans liquidity_stress_engine.py)
_LIQUIDITY_MACRO_VARS = [
    "GDP_growth", "unemployment_rate", "cpi_inflation",
    "policy_rate", "exchange_rate",
]

# Mapping WEO hist_col → MACRO_VARS (identique au liquidity wrapper)
_WEO_TO_LIQ: Dict[str, str] = {
    "real_gdp_growth":       "GDP_growth",
    "gdp_growth":            "GDP_growth",
    "unemployment_rate":     "unemployment_rate",
    "cpi_inflation":         "cpi_inflation",
    "policy_rate":           "policy_rate",
    "exchange_rate_lcu_usd": "exchange_rate",
    "exchange_rate":         "exchange_rate",
}


def _prepare_macro_df(macro_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Réplique exactement la logique de _fetch_and_map_macro du liquidity wrapper :
      1. Normaliser l'index → year comme index entier
      2. Renommer colonnes WEO → MACRO_VARS
      3. Remplir les colonnes manquantes avec 0.0
      4. Interpoler les NaN, ffill / bfill / fillna(0.0)
      5. Retourner avec "year" comme colonne (reset_index)
    """
    # 1. Normaliser : s'assurer que l'index contient bien les années (et non 0,1,2…)
    #    fetch_credit_macro retourne year comme index ; mais par sécurité on gère
    #    aussi le cas où "year" serait une colonne.
    if "year" in macro_raw.columns and macro_raw.index.name != "year":
        raw = macro_raw.set_index("year")
    else:
        raw = macro_raw.copy()
        raw.index.name = "year"

    mapped = pd.DataFrame(index=raw.index)

    # 2. Mapper les colonnes disponibles
    for src, dst in _WEO_TO_LIQ.items():
        if src in raw.columns and dst not in mapped.columns:
            mapped[dst] = raw[src]

    # 3. Combler les colonnes manquantes avec 0.0
    for col in _LIQUIDITY_MACRO_VARS:
        if col not in mapped.columns:
            mapped[col] = 0.0
            LOG.warning("NGFS LCR/NSFR: colonne macro '%s' absente → 0", col)

    # 3b. Conserver TOUTES les autres variables macro brutes récupérées via
    # API (fetch_credit_macro) comme candidats supplémentaires du tournoi —
    # même univers macro que les modules crédit/marché, au lieu de
    # restreindre la liquidité aux 5 variables ci-dessus. Additif : les 5
    # noms requis par LiquidityDataLoader._validate_macro() restent présents
    # (mapping ci-dessus), la validation continue donc de passer ; on
    # élargit seulement le pool de candidats — SatelliteCalibrator utilise
    # déjà toutes les colonnes numériques de macro_df comme candidats
    # (satellite_calibrator.py, all_macro_cols = df_m.columns).
    _already_used_src = set(_WEO_TO_LIQ.keys())
    for col in raw.columns:
        if col in _already_used_src or col in mapped.columns:
            continue
        if pd.api.types.is_numeric_dtype(raw[col]):
            mapped[col] = raw[col]

    # 4. Interpoler + fill
    mapped = mapped.interpolate(method="linear", limit_direction="both")
    mapped = mapped.ffill().bfill().fillna(0.0)
    mapped = mapped.dropna(how="all")
    mapped.index = mapped.index.astype(int)
    mapped.index.name = "year"

    # 5. "year" comme colonne (attendu par LiquidityDataLoader)
    return mapped.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# MAPPING NGFS variable_base → WEO hist_col (via RESOLUTION_RULES)
# ─────────────────────────────────────────────────────────────────────────────

def _build_ngfs_to_weo_map() -> List[Tuple[List[str], List[str], str]]:
    """Charge les RESOLUTION_RULES du moteur climatique (fallback manuel)."""
    try:
        from macro_selection_engine.variable_resolver import RESOLUTION_RULES
        return [
            (r.ngfs_keywords, getattr(r, "ngfs_exclude", []), r.hist_col)
            for r in RESOLUTION_RULES
        ]
    except Exception as exc:
        LOG.warning("Cannot import RESOLUTION_RULES: %s — using fallback", exc)
        return [
            (["gdp growth", "real gdp growth", "gdp yoy"], [], "real_gdp_growth"),
            (["gdp per capita"], [], "gdp_per_capita"),
            (["unemployment"], [], "unemployment_rate"),
            (["inflation", "consumer price", "cpi"], ["food", "energy"], "cpi_inflation"),
            (["central bank rate", "policy rate", "short term interest",
              "short-term interest", "intervention rate"], [], "policy_rate"),
            (["lending rate", "long term interest", "bond yield"], [], "lending_rate"),
            (["real effective exchange rate", "reer"], [], "real_effective_exchange_rate"),
            (["exchange rate", "fx rate", "nominal exchange"],
             ["effective", "real effective"], "exchange_rate_lcu_usd"),
            (["current account"], [], "current_account_gdp"),
            (["government debt", "public debt", "sovereign debt"], [], "gov_debt_gdp"),
            (["fiscal balance", "budget balance"], [], "fiscal_balance_gdp"),
        ]


_NGFS_WEO_RULES = _build_ngfs_to_weo_map()


def _map_ngfs_variable(var_name: str) -> Optional[str]:
    """Mappe un nom de variable NGFS → WEO hist_col. None si inconnu."""
    v_lower = var_name.lower()
    for keywords, excludes, hist_col in _NGFS_WEO_RULES:
        if any(kw in v_lower for kw in keywords):
            if not any(ex in v_lower for ex in excludes):
                return hist_col
    return None


def _weo_to_liq_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomme les colonnes WEO hist_col → MACRO_VARS dans un DataFrame NGFS.
    Ne remplit PAS les colonnes manquantes (predict_fn gère les NaN par imputation).
    """
    return df.rename(columns=_WEO_TO_LIQ)


# ─────────────────────────────────────────────────────────────────────────────
# LECTURE DU FICHIER NGFS → DataFrames macro par scénario (colonnes WEO)
# ─────────────────────────────────────────────────────────────────────────────

def _read_ngfs_macro_scenarios(
    ngfs_path: str,
    ngfs_mode: str,
    country: str,
    baseline_scn: str,
    adverse_scn: str,
    severe_scn: str,
    mapping_yaml: str = "",
) -> Dict[str, pd.DataFrame]:
    """
    Retourne {alias: DataFrame(index=year, columns=MACRO_VARS)}.
    Aliases : "baseline", "adverse", "severe".
    Les colonnes sont les noms MACRO_VARS (après mapping WEO → liq).
    """
    try:
        from macro_selection_engine.utils import EngineConfig
        from macro_selection_engine.data_loader import (
            load_ngfs, load_ngfs_ct, _normalize_ngfs_units,
        )
    except ImportError as exc:
        raise RuntimeError(f"Cannot import climate data_loader: {exc}")

    needed_scens = {baseline_scn, adverse_scn, severe_scn}

    cfg_kw: Dict = dict(
        country           = country,
        baseline_scenario = baseline_scn,
        ngfs_mode         = ngfs_mode,
        ngfs_path         = ngfs_path if ngfs_mode == "LT" else "",
        ngfs_ct_path      = ngfs_path if ngfs_mode == "CT" else None,
        auto_accept_mapping   = True,
        auto_select_model     = True,
        auto_accept_scenarios = True,
    )
    if mapping_yaml:
        cfg_kw["mapping_yaml_path"] = mapping_yaml

    cfg = EngineConfig(**cfg_kw)

    tidy = (load_ngfs_ct(cfg, scenario_map=None)
            if ngfs_mode == "CT"
            else load_ngfs(cfg, scenarios=None))

    tidy = tidy[tidy["scenario"].isin(needed_scens)].copy()
    if tidy.empty:
        LOG.warning("NGFS: aucun scénario parmi %s trouvé", needed_scens)
        return {}

    # Le fichier NGFS stocke Baseline en niveau absolu (unit="%") mais les
    # scénarios stressés en écart vs Baseline (unit="Abs. difference" ou
    # "% difference"). La reconstruction en niveau absolu se fait juste
    # après le pivot, ci-dessous (voir ngfs_credit_engine.py, même logique).
    tidy = _normalize_ngfs_units(tidy, baseline_scenario=baseline_scn)

    # Mapper variable_base → WEO hist_col
    tidy["hist_col"] = tidy["variable_base"].apply(_map_ngfs_variable)
    tidy = tidy[tidy["hist_col"].notna()].copy()
    if tidy.empty:
        LOG.warning("NGFS: aucune variable macro mappable")
        return {}

    LOG.info("NGFS: %d var_base → %d colonnes WEO",
             tidy["variable_base"].nunique(), tidy["hist_col"].nunique())

    scen_map = {baseline_scn: "baseline", adverse_scn: "adverse", severe_scn: "severe"}
    result: Dict[str, pd.DataFrame] = {}

    for ngfs_scen, alias in scen_map.items():
        df_s = tidy[tidy["scenario"] == ngfs_scen]
        if df_s.empty:
            LOG.warning("NGFS: scénario '%s' absent", ngfs_scen)
            continue

        # Agréger par (year, hist_col) — mean si plusieurs régions
        df_agg = (df_s.groupby(["year", "hist_col"])["value"]
                      .mean()
                      .reset_index())
        pivoted = df_agg.pivot(index="year", columns="hist_col", values="value")
        pivoted.index = pivoted.index.astype(int)
        pivoted = pivoted.sort_index()
        pivoted.index.name = "year"
        pivoted.columns.name = None  # supprimer le nom de niveau "hist_col"

        # Renommer WEO hist_col → MACRO_VARS (GDP_growth, exchange_rate…)
        pivoted = pivoted.rename(columns=_WEO_TO_LIQ)
        # Les colonnes NGFS non mappées restent telles quelles ;
        # predict_fn du SatelliteCalibrator impute les drivers manquants par la moyenne
        # historique, donc des colonnes supplémentaires ou manquantes sont tolérées.

        # Reconstruction du niveau absolu : Baseline est déjà un niveau ;
        # adverse/severe sont des écarts (Abs. difference, normalisés
        # ci-dessus) et doivent être additionnés au niveau Baseline pour
        # retrouver la vraie trajectoire de la variable sous ce scénario.
        if alias != "baseline" and "baseline" in result:
            pivoted = pivoted.add(result["baseline"], fill_value=0.0)

        LOG.info("NGFS '%s': %d années × %d vars MACRO", alias,
                 len(pivoted), len(pivoted.columns))
        result[alias] = pivoted

    return result


def _ngfs_viable_columns(ngfs_scenarios: Dict[str, pd.DataFrame]) -> set:
    """
    Macro columns whose NGFS-mapped counterpart carries real signal in at
    least one non-baseline scenario (adverse/severe) — i.e. not flat/zero
    across the whole projection horizon.

    A variable that is all-zero (or entirely NaN) in the NGFS file cannot
    transmit any climate shock once replayed, no matter how well it
    correlates with LCR/NSFR historically — selecting it would waste a
    satellite slot on a dead input. Eliminated here, before calibration,
    rather than discovered after the fact.
    """
    viable: set = set()
    for alias, df in ngfs_scenarios.items():
        if alias == "baseline":
            continue
        for col in df.columns:
            series = df[col].dropna()
            if not series.empty and float(series.abs().max()) > 1e-9:
                viable.add(col)
    return viable


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def compute_ngfs_lcr_nsfr(
    liq_excel_path: str,
    country_iso2: str,
    ngfs_path: str,
    ngfs_mode: str,
    baseline_scn: str,
    adverse_scn: str,
    severe_scn: str,
    mapping_yaml: str = "",
    cache_dir: str = "data_cache",
    cache_ttl: int = 30,
    portfolio_type: str = "mixed",
    forced_sat_ranks: Optional[Dict[str, int]] = None,
) -> Optional[Dict]:
    """
    Calcule LCR & NSFR projetés sous 3 scénarios NGFS.
    Retourne None si les données sont insuffisantes.

    forced_sat_ranks : override manuel Step 3 ("Utiliser un rang") — passé
    tel quel à LiquidityStressEngine.calibrate(forced_sat_ranks=...), qui
    supporte déjà ce paramètre nativement (SatelliteCalibrator.calibrate_all).
    """
    # Import via le chemin package correct (_MODULES_SRC dans sys.path)
    from liquidity_module.liquidity_stress_engine import (
        LiquidityDataLoader, LiquidityStressEngine,
    )
    from ..core.imf_weo_fetcher import fetch_credit_macro

    # ── 1. Vérifier le fichier Excel ─────────────────────────────────────────
    if not liq_excel_path or not Path(liq_excel_path).exists():
        LOG.info("NGFS LCR/NSFR: fichier Excel absent — skip")
        return {"error": "Fichier Excel liquidité introuvable."}

    # ── 2. Fetch macro IMF WEO / WB ──────────────────────────────────────────
    try:
        macro_raw, _ = fetch_credit_macro(
            country        = country_iso2,
            start_year     = 1990,
            cache_dir      = cache_dir,
            cache_ttl_days = cache_ttl,
        )
    except Exception as exc:
        LOG.warning("NGFS LCR/NSFR: fetch_credit_macro failed: %s", exc, exc_info=True)
        return {"error": f"Échec récupération macro (WB/IMF) : {exc}"}

    if macro_raw is None or macro_raw.empty:
        LOG.warning("NGFS LCR/NSFR: macro vide pour '%s'", country_iso2)
        return {"error": f"Aucune donnée macro disponible pour '{country_iso2}'."}

    # Prépare le DataFrame macro (mapping + fill + interpolation)
    macro_df = _prepare_macro_df(macro_raw)

    # ── 2b. Lire NGFS AVANT la calibration des satellites ─────────────────────
    # Chargé ici (et non après calibrate()) pour pouvoir éliminer, avant que
    # le tournoi de calibration ne les choisisse, les variables candidates
    # dont l'équivalent NGFS est plat/nul sur tout l'horizon — un tel
    # candidat pourrait corréler bien avec LCR/NSFR historiquement mais ne
    # transmettra jamais de choc climatique une fois rejoué sur NGFS.
    try:
        ngfs_scenarios = _read_ngfs_macro_scenarios(
            ngfs_path    = ngfs_path,
            ngfs_mode    = ngfs_mode,
            country      = country_iso2,
            baseline_scn = baseline_scn,
            adverse_scn  = adverse_scn,
            severe_scn   = severe_scn,
            mapping_yaml = mapping_yaml,
        )
    except Exception as exc:
        LOG.warning("NGFS LCR/NSFR: NGFS read failed: %s", exc, exc_info=True)
        return {"error": f"Échec lecture fichier NGFS : {exc}"}

    if not ngfs_scenarios:
        LOG.warning("NGFS LCR/NSFR: aucun scénario NGFS extrait")
        return {"error": "Aucun scénario NGFS extrait du fichier."}

    # Ne jamais éliminer "year" (sinon l'alignement temporel bilan/macro dans
    # LiquidityStressEngine.calibrate() retombe sur un index positionnel
    # 0,1,2… qui ne recoupe plus jamais les vraies années du bilan — voir
    # incident "0 année commune bilan/macro") ni les 5 colonnes macro
    # OBLIGATOIRES exigées par LiquidityDataLoader._validate_macro()
    # (GDP_growth, unemployment_rate, cpi_inflation, policy_rate,
    # exchange_rate) — même sans équivalent NGFS exploitable, elles doivent
    # rester présentes (à 0.0, via _prepare_macro_df) pour que le chargeur ne
    # lève pas d'erreur. Seules les variables macro CANDIDATES additionnelles
    # (au-delà de ces 5 + year) sont éliminées si plates.
    ngfs_viable = (_ngfs_viable_columns(ngfs_scenarios)
                   | set(_LIQUIDITY_MACRO_VARS) | {"year"})
    _cols_before = list(macro_df.columns)
    _dropped_flat = [c for c in _cols_before if c not in ngfs_viable]
    if _dropped_flat:
        macro_df = macro_df[[c for c in _cols_before if c in ngfs_viable]]
        LOG.info(
            "NGFS LCR/NSFR: %d variable(s) éliminée(s) — plates/nulles dans "
            "le fichier NGFS, aucun choc transmissible: %s",
            len(_dropped_flat), _dropped_flat,
        )
    if macro_df.shape[1] == 0:
        LOG.warning(
            "NGFS LCR/NSFR: toutes les variables candidates sont plates/"
            "nulles dans le fichier NGFS — aucun choc climatique transmissible."
        )
        return {"error": "Toutes les variables macro sont plates/nulles dans "
                          "le fichier NGFS — aucun choc climatique transmissible."}

    # ── 3. Charger Excel liquidité + calibrer les satellites ──────────────────
    try:
        inputs = LiquidityDataLoader.load(liq_excel_path, macro_df)
    except Exception as exc:
        LOG.warning("NGFS LCR/NSFR: LiquidityDataLoader.load failed: %s",
                    exc, exc_info=True)
        return {"error": f"Fichier liquidité illisible : {exc}"}

    engine = LiquidityStressEngine(inputs, portfolio_type=portfolio_type)
    try:
        engine.calibrate(forced_sat_ranks=forced_sat_ranks or {})
    except Exception as exc:
        LOG.warning("NGFS LCR/NSFR: calibrate() failed: %s", exc, exc_info=True)
        return {"error": f"Échec calibration satellites liquidité : {exc}"}

    # Année de choc = dernière année du bilan liquidité réel + 1. Le fichier
    # NGFS démarre ses colonnes-années à 2022 (avant la fin de l'historique
    # bilan réel) — sans ce filtre, les trajectoires NSFR/LCR projetées
    # (Row 3 Bloc B, Row 5) démarreraient à tort en 2022 au lieu de la
    # première année réellement inconnue. Même correctif que
    # ngfs_credit_engine.py::compute_ngfs_pd_lgd().
    try:
        _liq_proj_start = int(inputs.time_series["year"].max()) + 1
        ngfs_scenarios = {
            alias: df[df.index >= _liq_proj_start]
            for alias, df in ngfs_scenarios.items()
        }
    except Exception as exc:
        LOG.warning(
            "NGFS LCR/NSFR: filtrage année de choc échoué (ignoré): %s", exc,
        )

    # ── 7. Projeter LCR/NSFR via compute_stress() ────────────────────────────
    # enforce_sign=True : LCR ne dépend que de haircut_add (HQLA) et
    # run_off_retail/corporate (NCO) ; NSFR ne dépend que d'asf_factor_*
    # (ASF). Avec les contraintes de signe actives (run-off/haircut ne
    # peuvent qu'augmenter, ASF ne peut que baisser), LCR et NSFR sont
    # structurellement garantis de ne jamais s'améliorer sous un scénario
    # de stress climatique — jamais juste flat, mais strictement ≤ baseline.
    # Précédemment enforce_sign=False (un scénario NGFS "adverse" peut
    # afficher une macro court-terme meilleure que le baseline) produisait
    # des LCR/NSFR qui s'amélioraient sous stress — économiquement
    # défendable au niveau macro brut, mais incohérent avec la sémantique
    # d'un stress test bancaire (un scénario "adverse"/"severe" doit
    # dégrader les ratios réglementaires, jamais les améliorer).
    try:
        risk_outputs = engine.compute_stress(ngfs_scenarios, enforce_sign=True)
    except Exception as exc:
        LOG.warning("NGFS LCR/NSFR: compute_stress failed: %s", exc, exc_info=True)
        return {"error": f"Échec projection LCR/NSFR : {exc}"}

    if not risk_outputs:
        LOG.warning("NGFS LCR/NSFR: compute_stress retourné vide")
        return {"error": "La projection LCR/NSFR n'a produit aucun résultat."}

    # ── 8. Structurer pour le dashboard ──────────────────────────────────────
    # Résumé satellites
    sat_summary: Dict[str, Dict] = {}
    if engine._sat_results:
        for var, sr in engine._sat_results.items():
            sat_summary[var] = {
                "family":   getattr(sr, "family",   "N/A"),
                "drivers":  list(getattr(sr, "drivers", [])),
                "r2":       round(float(getattr(sr, "r2",  0.0)), 3),
                "aic":      round(float(getattr(sr, "aic", 0.0)), 2),
                "signs_ok": bool(getattr(sr, "signs_ok", False)),
                "n_obs":    int(getattr(sr, "n_obs", 0)),
            }

    def _safe_float(v, fallback: float = 0.0) -> float:
        """Convertit en float JSON-sérialisable (élimine inf, nan)."""
        try:
            f = float(v)
            if f != f or abs(f) == float("inf"):  # nan or inf
                return fallback
            return round(f, 2)
        except (TypeError, ValueError):
            return fallback

    # Trajectoires LCR/NSFR par scénario
    scenarios_out: Dict[str, Dict] = {}
    for scen_id, ro in risk_outputs.items():
        ts = ro.time_series
        try:
            years = [int(y) for y in
                     (ts["year"].tolist() if "year" in ts.columns else list(ts.index))]
            lcr  = [_safe_float(v) for v in ts["lcr"].tolist()] \
                   if "lcr"  in ts.columns else []
            nsfr = [_safe_float(v) for v in ts["nsfr"].tolist()] \
                   if "nsfr" in ts.columns else []
        except Exception as exc:
            LOG.warning("NGFS LCR/NSFR: time_series '%s': %s", scen_id, exc)
            years, lcr, nsfr = [], [], []

        scenarios_out[scen_id] = {"years": years, "lcr": lcr, "nsfr": nsfr}

    # Métriques clés depuis RiskOutput.metrics + propriétés
    def _min_val(scen: str, col: str) -> Optional[float]:
        vals = [v for v in scenarios_out.get(scen, {}).get(col, [])
                if v is not None]
        return round(float(min(vals)), 2) if vals else None

    baseline_ro = risk_outputs.get("baseline")
    metrics = {
        "lcr_baseline":  _safe_float(
            baseline_ro.metrics.get("lcr_baseline")) if baseline_ro else 0.0,
        "nsfr_baseline": _safe_float(
            baseline_ro.metrics.get("nsfr_baseline")) if baseline_ro else 0.0,
        "lcr_min_adv":     _min_val("adverse", "lcr"),
        "lcr_min_sev":     _min_val("severe",  "lcr"),
        "nsfr_min_adv":    _min_val("adverse", "nsfr"),
        "nsfr_min_sev":    _min_val("severe",  "nsfr"),
        "breach_year_adv": (int(risk_outputs["adverse"].breach_year)
                            if "adverse" in risk_outputs
                            and risk_outputs["adverse"].breach_year is not None
                            else None),
        "breach_year_sev": (int(risk_outputs["severe"].breach_year)
                            if "severe" in risk_outputs
                            and risk_outputs["severe"].breach_year is not None
                            else None),
    }

    LOG.info(
        "NGFS LCR/NSFR OK — baseline LCR=%.1f%% NSFR=%.1f%%"
        " | adv LCR_min=%s sev LCR_min=%s",
        metrics["lcr_baseline"], metrics["nsfr_baseline"],
        metrics["lcr_min_adv"],  metrics["lcr_min_sev"],
    )

    # ── Séries historiques LCR/NSFR observées (pour MultiTargetSatelliteFactory) ─
    # Si le fichier Excel contient des colonnes _lcr_obs / _nsfr_obs (LCR et NSFR
    # pré-calculés par la banque), on les expose comme séries historiques.
    # Sinon : None (la factory ignorera silencieusement ces cibles).
    _lcr_hist  = None
    _nsfr_hist = None
    try:
        ts = inputs.time_series
        # LiquidityDataLoader._read_time_series() does df.reset_index(drop=True)
        # after parsing, so ts's own index is a plain 0,1,2… row position, NOT
        # the "year" column — indexing by year here (when available) so the
        # resulting series carries real calendar years, not row positions.
        # Without this, lcr_hist_by_year/nsfr_hist_by_year below end up keyed
        # 0..14 instead of 2010..2024, and the dashboard's Row 3 Bloc B chart
        # plots history on a bogus 0-2000 x-axis instead of years.
        ts_by_year = ts.set_index("year") if "year" in ts.columns else ts
        if "_lcr_obs" in ts_by_year.columns:
            _s = ts_by_year["_lcr_obs"].dropna()
            if not _s.empty:
                _lcr_hist = _s.rename("lcr")
        if "_nsfr_obs" in ts_by_year.columns:
            _s = ts_by_year["_nsfr_obs"].dropna()
            if not _s.empty:
                _nsfr_hist = _s.rename("nsfr")
        if _lcr_hist is not None:
            LOG.info(
                "NGFS LCR/NSFR: exposition séries historiques observées — "
                "LCR %d pts, NSFR %s pts",
                len(_lcr_hist),
                len(_nsfr_hist) if _nsfr_hist is not None else "N/A",
            )
    except Exception as _he:
        LOG.debug("NGFS LCR/NSFR: extraction hist obs échouée (ignoré): %s", _he)

    # ── Raw macro trajectories, keyed by actual NGFS scenario name ───────────
    # Same fix as ngfs_credit_engine.py: ClimateMacroAdapter.compute_delta()
    # reads ngfs_projections["macro_trajectories"] for Layer-1 macro deltas.
    # Exposed here too as a fallback source (climate/wrapper.py tries
    # ngfs_credit first, then ngfs_liquidity).
    _alias_to_scenario = {
        "baseline": baseline_scn, "adverse": adverse_scn, "severe": severe_scn,
    }
    # Année de choc = même convention que ngfs_credit_engine.py : le fichier
    # NGFS démarre ses colonnes-années à 2022, mais le bilan liquidité réel
    # va jusqu'à sa dernière année observée. Réutiliser les années NGFS
    # antérieures à ça créerait le même décrochage visuel que pour le
    # crédit (courbe "Adverse/Severe" démarrant en plein milieu de
    # l'historique réel avec des valeurs NGFS divergentes).
    try:
        _macro_proj_start = int(inputs.time_series["year"].max()) + 1
    except Exception:
        _macro_proj_start = 0  # dégradation : ne filtre rien si indisponible
    macro_trajectories: Dict[str, Dict[int, Dict[str, float]]] = {}
    for _alias, _ngfs_df in ngfs_scenarios.items():
        _scen_name = _alias_to_scenario.get(_alias)
        if not _scen_name:
            continue
        # fallback=None (not the default 0.0): a missing/NaN NGFS value means
        # "no data", not "value is exactly zero" — see ngfs_credit_engine.py
        # for the bug this caused (bogus multi-hundred-percent deltas when a
        # scenario lacked data for a given year, silently zero-filled).
        macro_trajectories[_scen_name] = {
            int(_yr): {str(_c): _safe_float(_v, fallback=None) for _c, _v in _row.items()}
            for _yr, _row in _ngfs_df.to_dict(orient="index").items()
            if int(_yr) >= _macro_proj_start
        }

    # lcr_hist_by_year / nsfr_hist_by_year: JSON-serialisable {année: valeur}
    # dérivées de _lcr_hist / _nsfr_hist (des pd.Series conservées ci-dessous
    # pour MultiTargetSatelliteFactory, qui les lit en mémoire avant la
    # sérialisation JSON). Ajout additif — même correctif que
    # "macro_hist_by_year" dans ngfs_credit_engine.py : json.dumps(...,
    # default=str) transforme silencieusement une pd.Series en son repr
    # texte, la rendant inutilisable côté dashboard (Row 3 Bloc B).
    _lcr_hist_by_year = ({int(k): _safe_float(v, fallback=None)
                          for k, v in _lcr_hist.items()}
                         if _lcr_hist is not None else {})
    _nsfr_hist_by_year = ({int(k): _safe_float(v, fallback=None)
                           for k, v in _nsfr_hist.items()}
                          if _nsfr_hist is not None else {})

    return {
        "satellites":    sat_summary,
        "scenarios":     scenarios_out,
        "metrics":       metrics,
        "ngfs_mode":     ngfs_mode,
        "macro_trajectories": macro_trajectories,
        # ── Données historiques exposées pour MultiTargetSatelliteFactory ──────
        # macro_df_hist : DataFrame macro WEO historique (index=année int)
        # lcr_hist      : pd.Series LCR observés (%, index=année) — None si absent
        # nsfr_hist     : pd.Series NSFR observés (%, index=année) — None si absent
        "macro_df_hist": macro_df,
        "lcr_hist":      _lcr_hist,
        "nsfr_hist":     _nsfr_hist,
        "lcr_hist_by_year":  _lcr_hist_by_year,
        "nsfr_hist_by_year": _nsfr_hist_by_year,
        # tournament_leaderboards : {var_name: [candidats classés]} pour les
        # 5 satellites de composants bilan (run_off_retail/corporate,
        # haircut_add, asf_factor_retail/corporate) — déjà sérialisable
        # (SatelliteCalibrator._calibrate_tournament construit une version
        # JSON-safe). Additif, pour l'onglet Liquidité de l'Étape 3.
        "tournament_leaderboards": getattr(
            engine._calibrator, "tournament_leaderboards", {}
        ),
    }


def calibrate_ngfs_liquidity_satellites_only(
    liq_excel_path: str,
    country_iso2: str,
    ngfs_path: str,
    ngfs_mode: str,
    baseline_scn: str,
    adverse_scn: str,
    severe_scn: str,
    mapping_yaml: str = "",
    cache_dir: str = "data_cache",
    cache_ttl: int = 30,
    portfolio_type: str = "mixed",
    forced_sat_ranks: Optional[Dict[str, int]] = None,
) -> Optional[Dict]:
    """
    Calibration SEULE des 5 satellites de composants bilan liquidité
    (run_off_retail/corporate, haircut_add, asf_factor_retail/corporate) —
    sans compute_stress() (pas de projection multi-scénarios/années).

    Additif : mêmes étapes 1-3 que compute_ngfs_lcr_nsfr() (fetch macro,
    pré-filtre NGFS, chargement Excel + calibrate()), mais s'arrête avant
    la projection coûteuse. Conçu pour tourner en Phase 1 (Étape 3, onglet
    "Liquidité") afin d'exposer le tournoi de calibration sans attendre le
    lancement complet — cette fonction ne modifie pas compute_ngfs_lcr_nsfr,
    utilisée telle quelle en Phase 2 pour la projection complète.

    Retourne {"satellites": ..., "tournament_leaderboards": ...} ou None
    si les données sont insuffisantes.
    """
    from liquidity_module.liquidity_stress_engine import (
        LiquidityDataLoader, LiquidityStressEngine,
    )
    from ..core.imf_weo_fetcher import fetch_credit_macro

    if not liq_excel_path or not Path(liq_excel_path).exists():
        LOG.info("NGFS LCR/NSFR (calibration seule): fichier Excel absent — skip")
        return {"error": "Fichier Excel liquidité introuvable."}

    try:
        macro_raw, _ = fetch_credit_macro(
            country        = country_iso2,
            start_year     = 1990,
            cache_dir      = cache_dir,
            cache_ttl_days = cache_ttl,
        )
    except Exception as exc:
        LOG.warning("NGFS LCR/NSFR (calibration seule): fetch_credit_macro "
                    "failed: %s", exc, exc_info=True)
        return {"error": f"Échec récupération macro (WB/IMF) : {exc}"}

    if macro_raw is None or macro_raw.empty:
        LOG.warning("NGFS LCR/NSFR (calibration seule): macro vide pour '%s'",
                    country_iso2)
        return {"error": f"Aucune donnée macro disponible pour '{country_iso2}'."}

    macro_df = _prepare_macro_df(macro_raw)

    try:
        ngfs_scenarios = _read_ngfs_macro_scenarios(
            ngfs_path    = ngfs_path,
            ngfs_mode    = ngfs_mode,
            country      = country_iso2,
            baseline_scn = baseline_scn,
            adverse_scn  = adverse_scn,
            severe_scn   = severe_scn,
            mapping_yaml = mapping_yaml,
        )
    except Exception as exc:
        LOG.warning("NGFS LCR/NSFR (calibration seule): NGFS read failed: %s",
                    exc, exc_info=True)
        return {"error": f"Échec lecture fichier NGFS : {exc}"}

    if not ngfs_scenarios:
        LOG.warning("NGFS LCR/NSFR (calibration seule): aucun scénario NGFS extrait")
        return {"error": "Aucun scénario NGFS extrait du fichier."}

    # Ne jamais éliminer "year" ni les 5 colonnes macro obligatoires — voir
    # commentaire équivalent dans compute_ngfs_lcr_nsfr() ci-dessus.
    ngfs_viable = (_ngfs_viable_columns(ngfs_scenarios)
                   | set(_LIQUIDITY_MACRO_VARS) | {"year"})
    _cols_before = list(macro_df.columns)
    macro_df = macro_df[[c for c in _cols_before if c in ngfs_viable]]
    if macro_df.shape[1] == 0:
        LOG.warning(
            "NGFS LCR/NSFR (calibration seule): toutes les variables "
            "candidates sont plates/nulles dans le fichier NGFS."
        )
        return {"error": "Toutes les variables macro sont plates/nulles dans "
                          "le fichier NGFS — aucun choc climatique transmissible."}

    try:
        inputs = LiquidityDataLoader.load(liq_excel_path, macro_df)
    except Exception as exc:
        LOG.warning("NGFS LCR/NSFR (calibration seule): LiquidityDataLoader.load "
                    "failed: %s", exc, exc_info=True)
        return {"error": f"Fichier liquidité illisible : {exc}"}

    engine = LiquidityStressEngine(inputs, portfolio_type=portfolio_type)
    try:
        engine.calibrate(forced_sat_ranks=forced_sat_ranks or {})
    except Exception as exc:
        LOG.warning("NGFS LCR/NSFR (calibration seule): calibrate() failed: %s",
                    exc, exc_info=True)
        return {"error": f"Échec calibration satellites liquidité : {exc}"}

    sat_summary: Dict[str, Dict] = {}
    if engine._sat_results:
        for var, sr in engine._sat_results.items():
            sat_summary[var] = {
                "family":   getattr(sr, "family",   "N/A"),
                "drivers":  list(getattr(sr, "drivers", [])),
                "r2":       round(float(getattr(sr, "r2",  0.0)), 3),
                "aic":      round(float(getattr(sr, "aic", 0.0)), 2),
                "signs_ok": bool(getattr(sr, "signs_ok", False)),
                "n_obs":    int(getattr(sr, "n_obs", 0)),
            }

    return {
        "satellites": sat_summary,
        "tournament_leaderboards": getattr(
            engine._calibrator, "tournament_leaderboards", {}
        ),
    }
