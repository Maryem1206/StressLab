"""
ngfs_credit_engine.py
=====================
Projects PD and LGD_PIT under NGFS scenarios (LT/CT) by reusing the
credit satellite calibration pipeline without any refit on NGFS data.

Pipeline (TYPE B, mirrors ngfs_liquidity_engine.py):
  1. Load PD+EAD historical file (uploaded dataset)
  2. Fetch conventional macro history via WEO/WB API (fetch_credit_macro)
  3. Run macro pre-tests + feature engineering + satellite tournament ONCE
     → produces best (satellite model coefficients, μ, σ) + Frye-Jacobs (k, ρ)
  4. Read NGFS file → WEO-aligned trajectories per scenario
  5. Project PD_PIT = _predict_pd(_project_features(ngfs_macro, best)) — no refit
  6. Compute LGD_PIT = _frye_jacobs_lgd(pd_pit, rho, k) — no refit
  7. Return structured result for dashboard (symmetric with ngfs_liquidity_engine)

The satellite coefficients calibrated in step 3 are stored in immutable dicts
and never overwritten.  Steps 5-6 are pure functions (no fitting, no side-effects).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("ngfs_credit_engine")

# ── sys.path : expose modules_src/ for macro_selection_engine imports ────────
_APP_ROOT    = Path(__file__).resolve().parent.parent.parent.parent
_MODULES_SRC = str(_APP_ROOT / "modules_src")
_CLM_SRC     = str(_APP_ROOT / "modules_src" / "climate_module")
for _p in (_MODULES_SRC, _CLM_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Column aliases: hist_col (RESOLUTION_RULES) → CREDIT_MACRO_UNIVERSE name ─
# After fixing variable_resolver.py (hist_col renames for reer/gov_debt/
# fiscal_balance), only the gdp_growth alias remains necessary.
_HIST_TO_CREDIT: Dict[str, str] = {
    "gdp_growth": "real_gdp_growth",
}


def _build_ngfs_to_credit_map() -> List[Tuple[List[str], List[str], str]]:
    """
    Build keyword-match rules to map NGFS variable names → CREDIT_MACRO_UNIVERSE
    column names, loading them from RESOLUTION_RULES when possible.
    """
    try:
        from macro_selection_engine.variable_resolver import RESOLUTION_RULES
        rules = []
        for r in RESOLUTION_RULES:
            credit_col = _HIST_TO_CREDIT.get(r.hist_col, r.hist_col)
            rules.append((r.ngfs_keywords, getattr(r, "ngfs_exclude", []), credit_col))
        return rules
    except Exception as exc:
        LOG.warning("Cannot import RESOLUTION_RULES: %s — using built-in fallback", exc)
        return [
            (["gdp growth", "real gdp growth", "gdp yoy"],       [],                         "real_gdp_growth"),
            (["unemployment"],                                    [],                         "unemployment_rate"),
            (["inflation", "consumer price", "cpi"],             ["food", "energy"],         "cpi_inflation"),
            (["central bank rate", "policy rate", "short term interest",
              "short-term interest", "intervention rate"],        [],                         "policy_rate"),
            (["lending rate", "long term interest", "bond yield"], [],                       "lending_rate"),
            (["real effective exchange rate", "reer"],            [],                         "real_effective_xrate"),
            (["exchange rate", "fx rate", "nominal exchange"],   ["effective", "real eff"],  "exchange_rate_lcu_usd"),
            (["current account"],                                 [],                         "current_account_gdp"),
            (["government debt", "public debt", "sovereign debt"], [],                       "gov_debt_gdp"),
            (["fiscal balance", "budget balance"],                [],                         "fiscal_balance_gdp"),
        ]


_NGFS_CREDIT_RULES = _build_ngfs_to_credit_map()


def _map_ngfs_variable(var_name: str) -> Optional[str]:
    """NGFS variable label → CREDIT_MACRO_UNIVERSE column name. None if unknown."""
    v = var_name.lower()
    for keywords, excludes, col in _NGFS_CREDIT_RULES:
        if any(kw in v for kw in keywords):
            if not any(ex in v for ex in excludes):
                return col
    return None


def _read_ngfs_credit_scenarios(
    ngfs_path: str,
    ngfs_mode: str,
    country: str,
    baseline_scn: str,
    adverse_scn: str,
    severe_scn: str,
    mapping_yaml: str = "",
) -> Dict[str, pd.DataFrame]:
    """
    Load NGFS file and return {alias: DataFrame(index=year, columns=credit_col)}.
    Aliases: "baseline", "adverse", "severe".
    """
    try:
        from macro_selection_engine.utils import EngineConfig
        from macro_selection_engine.data_loader import (
            load_ngfs, load_ngfs_ct, _normalize_ngfs_units,
        )
    except ImportError as exc:
        raise RuntimeError(f"Cannot import climate data_loader: {exc}")

    needed = {baseline_scn, adverse_scn, severe_scn}

    cfg_kw: Dict = dict(
        country               = country,
        baseline_scenario     = baseline_scn,
        ngfs_mode             = ngfs_mode,
        ngfs_path             = ngfs_path if ngfs_mode == "LT" else "",
        ngfs_ct_path          = ngfs_path if ngfs_mode == "CT" else None,
        auto_accept_mapping   = True,
        auto_select_model     = True,
        auto_accept_scenarios = True,
    )
    if mapping_yaml:
        cfg_kw["mapping_yaml_path"] = mapping_yaml

    cfg  = EngineConfig(**cfg_kw)
    tidy = (load_ngfs_ct(cfg, scenario_map=None)
            if ngfs_mode == "CT"
            else load_ngfs(cfg, scenarios=None))

    tidy = tidy[tidy["scenario"].isin(needed)].copy()
    if tidy.empty:
        LOG.warning("NGFS crédit: aucun scénario parmi %s trouvé dans le fichier", needed)
        return {}

    # Le fichier NGFS stocke Baseline en niveau absolu (unit="%") mais les
    # scénarios stressés en écart vs Baseline (unit="Abs. difference" ou
    # "% difference"). Sans cette normalisation, un écart de +0.24 pt de
    # taux directeur serait lu comme "le taux vaut 0.24%" au lieu de
    # "15.75% + 0.24% = 15.99%" — la reconstruction en niveau absolu se
    # fait juste après le pivot, ci-dessous.
    tidy = _normalize_ngfs_units(tidy, baseline_scenario=baseline_scn)

    tidy["credit_col"] = tidy["variable_base"].apply(_map_ngfs_variable)
    tidy = tidy[tidy["credit_col"].notna()].copy()
    if tidy.empty:
        LOG.warning("NGFS crédit: aucune variable mappable vers CREDIT_MACRO_UNIVERSE")
        return {}

    LOG.info("NGFS crédit: %d var_base → %d colonnes crédit",
             tidy["variable_base"].nunique(), tidy["credit_col"].nunique())

    scen_map = {baseline_scn: "baseline", adverse_scn: "adverse", severe_scn: "severe"}
    result: Dict[str, pd.DataFrame] = {}

    for ngfs_scen, alias in scen_map.items():
        df_s = tidy[tidy["scenario"] == ngfs_scen]
        if df_s.empty:
            LOG.warning("NGFS crédit: scénario '%s' absent", ngfs_scen)
            continue

        df_agg  = (df_s.groupby(["year", "credit_col"])["value"]
                       .mean().reset_index())
        pivoted = df_agg.pivot(index="year", columns="credit_col", values="value")
        pivoted.index = pivoted.index.astype(int)
        pivoted = pivoted.sort_index()
        pivoted.index.name  = "year"
        pivoted.columns.name = None

        # Reconstruction du niveau absolu : Baseline est déjà un niveau ;
        # adverse/severe sont des écarts (Abs. difference, normalisés
        # ci-dessus) et doivent être additionnés au niveau Baseline pour
        # retrouver la vraie trajectoire de la variable sous ce scénario.
        if alias != "baseline" and "baseline" in result:
            pivoted = pivoted.add(result["baseline"], fill_value=0.0)

        LOG.info("NGFS crédit '%s': %d années × %d colonnes macro",
                 alias, len(pivoted), len(pivoted.columns))
        result[alias] = pivoted

    return result


def _safe_float(v, fallback: float = 0.0) -> float:
    try:
        f = float(v)
        return fallback if (f != f or abs(f) == float("inf")) else round(f, 6)
    except (TypeError, ValueError):
        return fallback


def _ngfs_viable_columns(ngfs_scenarios: Dict[str, "pd.DataFrame"]) -> set:
    """
    Historical columns whose NGFS-mapped counterpart carries real signal in
    at least one non-baseline scenario (adverse/severe) — i.e. not flat/zero
    across the whole projection horizon.

    A variable that is all-zero (or entirely NaN) in the NGFS file cannot
    transmit any climate shock once replayed, no matter how well it
    correlates with PD historically — selecting it would waste a model slot
    on a dead input. Eliminated here, before the satellite tournament runs,
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
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def compute_ngfs_pd_lgd(
    upload_paths: Dict[str, str],
    country_iso2: str,
    ngfs_path: str,
    ngfs_mode: str,
    baseline_scn: str,
    adverse_scn: str,
    severe_scn: str,
    lgd_ttc: float = 0.45,
    history_start: int = 1990,
    mapping_yaml: str = "",
    cache_dir: str = "data_cache",
    cache_ttl: int = 30,
) -> Optional[Dict]:
    """
    Compute PD_PIT and LGD_PIT trajectories under NGFS scenarios without
    refitting the satellite model.

    Mirrors compute_ngfs_lcr_nsfr() from ngfs_liquidity_engine.py.
    Returns None on any failure — the caller (climate/wrapper.py) treats
    this as non-blocking and continues without NGFS credit projections.

    Returns
    -------
    {
      "model":     { family, combo, r2, n_obs, verdict, n_fail, n_warn, n_pass },
      "scenarios": { "baseline": {"years":[], "pd":[], "lgd":[]}, "adverse": …, "severe": … },
      "metrics":   { pd_ttc, lgd_ttc, rho, fj_k, pd_*_mean, pd_*_peak },
      "ngfs_mode": "LT" | "CT",
    }
    """

    # ── Lazy imports — credit module functions ───────────────────────────────
    try:
        from app.modules.credit.wrapper import (
            _load_credit_dataset,
            _generate_features,
            _top_k_dedup,
            _run_tournament,
            _project_features,
            _predict_pd,
            _asset_correlation_irb,
            _calibrate_fj_k,
            _frye_jacobs_lgd,
            _build_capital_params,
        )
        from app.modules.core.imf_weo_fetcher import fetch_credit_macro, universe_index
        from app.modules.core.macro_pre_tests import run_macro_pre_tests
        from app.modules.credit.capital_engine import CreditCapitalEngine, load_capital_history
    except ImportError as exc:
        LOG.warning("NGFS PD/LGD: import crédit échoué: %s", exc)
        return None

    # ── 1. Charger PD + EAD historiques ─────────────────────────────────────
    try:
        pd_series, ead_series, last_pd_year = _load_credit_dataset(upload_paths)
    except Exception as exc:
        LOG.warning("NGFS PD/LGD: _load_credit_dataset échoué: %s", exc)
        return None

    pd_ttc = float(pd_series.mean())
    LOG.info("NGFS PD/LGD: PD_TTC=%.4f, last_pd_year=%d", pd_ttc, last_pd_year)

    # ── 2. Fetch macro WEO conventionnel ─────────────────────────────────────
    try:
        macro_full, _ = fetch_credit_macro(
            country       = country_iso2,
            start_year    = history_start,
            cache_dir     = cache_dir,
            cache_ttl_days = cache_ttl,
        )
    except Exception as exc:
        LOG.warning("NGFS PD/LGD: fetch_credit_macro échoué: %s", exc)
        return None

    if macro_full is None or macro_full.empty:
        LOG.warning("NGFS PD/LGD: macro vide pour '%s'", country_iso2)
        return None

    macro_hist = macro_full.loc[macro_full.index <= last_pd_year].copy()

    # ── 3. Aligner PD × macro ────────────────────────────────────────────────
    target = pd_series.rename("pd")
    joined = macro_hist.join(target, how="inner").dropna(subset=["pd"])
    if len(joined) < 5:
        LOG.warning("NGFS PD/LGD: seulement %d années communes PD×macro (min 5)",
                    len(joined))
        return None

    macro_aligned  = joined.drop(columns=["pd"])
    target_aligned = joined["pd"]
    n_obs          = len(target_aligned)
    uni_idx        = universe_index()
    LOG.info("NGFS PD/LGD: %d années alignées PD×macro (%d–%d)",
             n_obs, int(joined.index.min()), int(joined.index.max()))

    # ── 4. Pré-tests macro ───────────────────────────────────────────────────
    # Mirrors credit/wrapper.py's STEP 0.5: a pre-test failure must not let
    # the pipeline silently fall through to fitting on unfiltered/unvetted
    # macro variables. Fail safe (return None) instead, consistent with the
    # "no accepted variables" case just below.
    try:
        pre = run_macro_pre_tests(
            macro_df     = macro_aligned,
            pd_series    = target_aligned,
            universe_idx = uni_idx,
        )
    except Exception as exc:
        LOG.warning("NGFS PD/LGD: run_macro_pre_tests échoué: %s", exc)
        return None
    if not pre.accepted:
        LOG.warning("NGFS PD/LGD: aucune variable macro acceptée par les pré-tests")
        return None
    macro_aligned = macro_aligned[pre.accepted]
    LOG.info("NGFS PD/LGD: pré-tests → %d vars acceptées", len(pre.accepted))

    # ── 4b. Lire trajectoires NGFS AVANT le tournoi ──────────────────────────
    # Chargé ici (et non après le tournoi) pour pouvoir éliminer, avant la
    # sélection de variables, toute variable candidate dont l'équivalent NGFS
    # est plat/nul sur tout l'horizon : un tel candidat pourrait être choisi
    # par le tournoi (bonne corrélation historique WB) mais ne transmettra
    # jamais de choc climatique réel une fois rejoué sur les trajectoires NGFS.
    try:
        ngfs_scenarios = _read_ngfs_credit_scenarios(
            ngfs_path   = ngfs_path,
            ngfs_mode   = ngfs_mode,
            country     = country_iso2,
            baseline_scn = baseline_scn,
            adverse_scn  = adverse_scn,
            severe_scn   = severe_scn,
            mapping_yaml = mapping_yaml,
        )
    except Exception as exc:
        LOG.warning("NGFS PD/LGD: lecture NGFS échouée: %s", exc)
        return None

    if not ngfs_scenarios:
        LOG.warning("NGFS PD/LGD: aucun scénario NGFS extrait")
        return None

    # ── 4c. Recalage niveau réel — élimine la discontinuité de raccord ───────
    # Le niveau "Baseline" du fichier NGFS vient d'un modèle indépendant
    # (NiGEM) qui n'a aucune raison de tomber exactement sur la dernière
    # valeur macro RÉELLEMENT observée (WEO) à l'année de raccord — d'où un
    # saut visuel (et un vrai décalage de niveau) entre la courbe historique
    # et les 3 courbes projetées (baseline/adverse/severe) sur les graphs
    # satellites. Correction : un décalage additif constant par variable,
    # calé sur last_pd_year (dernière année avant projection), appliqué
    # UNIFORMÉMENT aux 3 scénarios — ne change donc AUCUN écart
    # adverse/severe vs baseline (les deltas utilisés par Layer 1/Layer 2
    # sont des différences entre scénarios à année égale, invariantes par
    # translation), seulement le niveau absolu affiché/reconstruit.
    # Chaque variable a sa propre dernière année observée (ex. lending_rate
    # s'arrête en 2024 quand unemployment_rate va jusqu'à 2031 via les
    # projections WEO du FMI) — ancrer TOUTES les variables sur la même
    # année fixe (last_pd_year) manquerait celles dont l'historique s'arrête
    # plus tôt. On cherche donc, par variable, la dernière année ≤
    # last_pd_year effectivement disponible dans macro_full.
    _anchor_ceiling = int(last_pd_year)
    if "baseline" in ngfs_scenarios:
        _bl_df = ngfs_scenarios["baseline"]
        for _col in _bl_df.columns:
            if _col not in macro_full.columns:
                continue
            _hist_col = macro_full[_col].dropna()
            _hist_col = _hist_col[_hist_col.index <= _anchor_ceiling]
            if _hist_col.empty:
                continue
            _anchor_year = int(_hist_col.index.max())
            if _anchor_year not in _bl_df.index:
                continue
            _hist_val = _hist_col.loc[_anchor_year]
            _ngfs_val = _bl_df.at[_anchor_year, _col]
            if pd.isna(_hist_val) or pd.isna(_ngfs_val):
                continue
            _offset = float(_hist_val) - float(_ngfs_val)
            if abs(_offset) < 1e-9:
                continue
            for _alias in ngfs_scenarios:
                if _col in ngfs_scenarios[_alias].columns:
                    ngfs_scenarios[_alias][_col] = ngfs_scenarios[_alias][_col] + _offset
            LOG.debug(
                "NGFS crédit: recalage '%s' à l'année %d — offset=%.4f "
                "(hist=%.4f, ngfs_baseline=%.4f)",
                _col, _anchor_year, _offset, _hist_val, _ngfs_val,
            )

    ngfs_viable = _ngfs_viable_columns(ngfs_scenarios)
    _cols_before = list(macro_aligned.columns)
    _dropped_flat = [c for c in _cols_before if c not in ngfs_viable]
    if _dropped_flat:
        macro_aligned = macro_aligned[[c for c in _cols_before if c in ngfs_viable]]
        LOG.info(
            "NGFS PD/LGD: %d variable(s) éliminée(s) — plates/nulles dans le "
            "fichier NGFS, aucun choc transmissible: %s",
            len(_dropped_flat), _dropped_flat,
        )
    if macro_aligned.shape[1] == 0:
        LOG.warning(
            "NGFS PD/LGD: toutes les variables candidates sont plates/nulles "
            "dans le fichier NGFS — aucun choc climatique transmissible."
        )
        return None

    # ── 5. Feature engineering + tournoi satellite (UNE SEULE FOIS) ─────────
    try:
        feature_df, all_specs = _generate_features(
            macro_aligned, target_aligned, uni_idx)
        k_eff              = min(15, max(n_obs - 2, 3))
        topk_df, topk_specs = _top_k_dedup(feature_df, all_specs, k=k_eff)
        rho                = _asset_correlation_irb(pd_ttc)
        best, _leaderboard = _run_tournament(
            topk_df, target_aligned, topk_specs, n_obs, rho=rho)
    except Exception as exc:
        LOG.warning("NGFS PD/LGD: calibration satellite échouée: %s", exc)
        return None

    best_combo = best["combo"]
    best_specs = {f: topk_specs[f] for f in best_combo}

    # ── 6. Calibration Frye-Jacobs (une seule fois, sur historique) ──────────
    fj_k = _calibrate_fj_k(pd_ttc, lgd_ttc, rho)
    LOG.info("NGFS PD/LGD: ρ=%.4f k=%.4f (PD_TTC=%.4f LGD_TTC=%.2f)",
             rho, fj_k, pd_ttc, lgd_ttc)

    # ── 8. Projeter PD_PIT + LGD_PIT SANS refit ─────────────────────────────
    # Les coefficients de best, best_specs, rho et fj_k sont des objets
    # immuables (dicts + floats) issus de l'étape 5-6.
    # _project_features et _predict_pd sont des fonctions pures.
    scenarios_out: Dict[str, Dict] = {}

    # Collected for the capital engine (step 9) — same loop, no extra cost.
    _pd_arrays_ngfs:  Dict[str, np.ndarray] = {}
    _lgd_arrays_ngfs: Dict[str, np.ndarray] = {}
    _horizon_ngfs:    Optional[List[int]]   = None

    needed_parents = {best_specs[f].parent_var for f in best_combo}

    # Année de choc = dernière année historique réelle + 1 (ex. 2026 si
    # l'historique PD va jusqu'à 2025). Le fichier NGFS démarre ses
    # colonnes-années à 2022 (avant même la fin de l'historique réel) — sans
    # ce filtre, TOUT l'horizon d'orchestration (crédit/liquidité/marché,
    # Rows 3/4/5 du dashboard) inclut à tort 2022-2025, années déjà connues
    # historiquement, faussant les projections marché/liquidité affichées
    # dès le début de l'horizon au lieu de démarrer en 2026.
    _proj_start = int(last_pd_year) + 1

    for alias, ngfs_df in ngfs_scenarios.items():
        ngfs_df = ngfs_df[ngfs_df.index >= _proj_start]
        horizon = list(ngfs_df.index)
        if not horizon:
            LOG.warning(
                "NGFS crédit '%s': aucune année >= %d dans le fichier NGFS "
                "— scénario ignoré.", alias, _proj_start,
            )
            continue
        try:
            # Toute colonne manquante dans le DataFrame NGFS est remplie avec NaN.
            # Dans _predict_pd, np.nan_to_num(X_raw, nan=0.0) impute la valeur
            # manquante par 0 en espace z-score, soit la moyenne historique.
            for col in needed_parents:
                if col not in ngfs_df.columns:
                    ngfs_df = ngfs_df.copy()
                    ngfs_df[col] = float("nan")
                    LOG.info("NGFS crédit '%s': '%s' absent → NaN (moyenne hist.)",
                             alias, col)

            # Ancrage : les transforms "growth"/"lag1" (.diff()/.shift(1) dans
            # _apply_transform) ont besoin d'une valeur à t-1 pour calculer la
            # 1ère année projetée (ex. 2026). ngfs_df démarre pile à cette
            # année-là (_proj_start), donc sans point d'ancrage .diff() vaut
            # NaN pour CETTE seule année → np.nan_to_num() le remplace
            # silencieusement par 0 (z-score moyen), IDENTIQUE pour
            # baseline/adverse/severe — ce qui efface tout signal de scénario
            # uniquement sur la 1ère année.
            #
            # L'ancre est prise sur la trajectoire NGFS elle-même à l'année
            # précédente (_proj_start - 1), PAS sur l'historique réel : les
            # deux sources (bilan banque vs fichier NGFS/NiGEM) ne se
            # raccordent pas forcément exactement, et utiliser l'historique
            # réel comme ancre introduit un saut de raccordement artificiel
            # (constaté empiriquement : PD retombait au plancher 1e-5 pour
            # les 3 scénarios en 2026, identique, donc toujours sans signal
            # de scénario — juste un défaut différent). L'ancrage intra-NGFS
            # évite ce saut : les deux points du .diff() viennent du même
            # modèle/vintage.
            anchor_src = ngfs_scenarios[alias]
            anchor_year = _proj_start - 1
            if anchor_year in anchor_src.index:
                anchor_vals = {
                    col: (float(anchor_src.at[anchor_year, col])
                          if col in anchor_src.columns
                          and pd.notna(anchor_src.at[anchor_year, col])
                          else float("nan"))
                    for col in ngfs_df.columns
                }
                anchor_row = pd.DataFrame([anchor_vals], index=[anchor_year])
                ngfs_df_anchored = pd.concat([anchor_row, ngfs_df])
            else:
                ngfs_df_anchored = ngfs_df

            feature_proj = _project_features(ngfs_df_anchored, best_combo, best_specs, horizon)
            pd_pit       = _predict_pd(feature_proj, best)
            lgd_pit      = _frye_jacobs_lgd(pd_pit, rho, fj_k)

            scenarios_out[alias] = {
                "years": horizon,
                "pd":    [_safe_float(v) for v in pd_pit.tolist()],
                "lgd":   [_safe_float(v) for v in lgd_pit.tolist()],
            }
            # Retain raw arrays for capital engine (step 9).
            _pd_arrays_ngfs[alias]  = pd_pit
            _lgd_arrays_ngfs[alias] = lgd_pit
            if _horizon_ngfs is None:
                _horizon_ngfs = horizon

            LOG.info(
                "NGFS PD/LGD '%s': %d années | PD_mean=%.4f PD_peak=%.4f | "
                "LGD_mean=%.4f",
                alias, len(horizon),
                float(pd_pit.mean()), float(pd_pit.max()),
                float(lgd_pit.mean()),
            )
        except Exception as exc:
            LOG.warning("NGFS PD/LGD: projection '%s' échouée: %s", alias, exc)

    if not scenarios_out:
        LOG.warning("NGFS PD/LGD: aucune projection NGFS réussie")
        return None

    # ── 9. Capital ASRF / RWA / ratios sous scénario NGFS ────────────────────
    # Non-bloquant : si le moteur échoue, scenarios_out reste valide sans capital.
    # Paramètres construits via _build_capital_params (fonction partagée avec
    # credit/wrapper.py) → EAD growth rate identique entre les deux chemins.
    # _calibrate_severity utilise _pd_arrays_ngfs : sévérité calibrée sur les
    # PD climatiques de CE scénario NGFS, pas sur les PD conventionnelles.
    capital_hist_by_year: Dict[int, Dict[str, Optional[float]]] = {}
    # capital_trajectories : {metric: {"historical"/"baseline"/"adverse"/"severe": {x,y}}}
    # — exposé pour ClimateEngineAdapter (injection directe NGFS), qui construit
    # ses PlatformResult sans rappeler CreditModuleWrapper.run().
    capital_trajectories: Dict[str, Any] = {}
    if len(_pd_arrays_ngfs) == 3 and _horizon_ngfs is not None:
        try:
            _cap_path   = upload_paths.get("capital", "")
            _capital_df = load_capital_history(_cap_path)
            _hist_lgd   = _frye_jacobs_lgd(
                pd_series.sort_index().values.astype(float), rho, fj_k
            )
            _ead_M      = float(ead_series.iloc[-1])

            _cap_params = _build_capital_params(ead_series, {})
            _cap_engine = CreditCapitalEngine(_cap_params)

            _capital_out = _cap_engine.run(
                pd_hist    = pd_series,
                lgd_hist   = _hist_lgd,
                pd_stress  = _pd_arrays_ngfs,
                lgd_stress = _lgd_arrays_ngfs,
                ead        = _ead_M,
                horizon    = _horizon_ngfs,
                capital_df = _capital_df,
            )
            capital_trajectories = _capital_out.capital_trajectories

            def _col_list(df: "pd.DataFrame", col: str) -> Optional[List[float]]:
                if col not in df.columns:
                    return None
                return [_safe_float(v) for v in df[col].tolist()]

            def _first_breach(df: "pd.DataFrame") -> Optional[int]:
                for _, row in df.sort_values("year").iterrows():
                    if (row.get("car_flag") == "BREACH"
                            or row.get("leverage_flag") == "BREACH"):
                        return int(row["year"])
                return None

            for _alias in list(scenarios_out.keys()):
                _sub = _capital_out.stress_df[
                    _capital_out.stress_df["scenario"] == _alias
                ].copy()
                if _sub.empty:
                    continue
                scenarios_out[_alias].update({
                    "k_required":    _col_list(_sub, "k_required"),
                    "rwa_stressed":  _col_list(_sub, "rwa_stressed"),
                    "car":           _col_list(_sub, "car"),
                    "tier1_ratio":   _col_list(_sub, "tier1_ratio"),
                    "cet1_ratio":    _col_list(_sub, "cet1_ratio"),
                    "leverage_ratio": _col_list(_sub, "leverage_ratio"),
                    "breach_year":   _first_breach(_sub),
                })

            # Ratios historiques observés (CAR/Tier1/CET1/leverage) — déjà
            # calculés par CreditCapitalEngine.compute_historical() mais
            # jusqu'ici jamais exposés au-delà de l'objet en mémoire. Ajout
            # additif pour la Row 5 du dashboard (charts CAR/Tier1 4-séries).
            _hist_df = _capital_out.historical_df
            if _hist_df is not None and not _hist_df.empty:
                for _, _hrow in _hist_df.iterrows():
                    capital_hist_by_year[int(_hrow["year"])] = {
                        "car":            _safe_float(_hrow.get("car"), fallback=None),
                        "tier1_ratio":    _safe_float(_hrow.get("tier1_ratio"), fallback=None),
                        "cet1_ratio":     _safe_float(_hrow.get("cet1_ratio"), fallback=None),
                        "leverage_ratio": _safe_float(_hrow.get("leverage_ratio"), fallback=None),
                    }

            LOG.info(
                "NGFS capital OK — capital_df=%s ead=%.2f M",
                _capital_df is not None, _ead_M,
            )
        except Exception as _exc:
            LOG.warning(
                "NGFS capital: moteur capital ignoré (non-bloquant): %s", _exc
            )

    def _mean(alias: str) -> Optional[float]:
        vals = scenarios_out.get(alias, {}).get("pd", [])
        return round(float(np.mean(vals)), 6) if vals else None

    def _peak(alias: str) -> Optional[float]:
        vals = scenarios_out.get(alias, {}).get("pd", [])
        return round(float(max(vals)), 6) if vals else None

    LOG.info(
        "NGFS PD/LGD OK — baseline_mean=%s adverse_mean=%s severe_mean=%s",
        _mean("baseline"), _mean("adverse"), _mean("severe"),
    )

    # ── Raw macro trajectories, keyed by actual NGFS scenario name ───────────
    # ClimateMacroAdapter.compute_delta() looks up ngfs_projections["macro_trajectories"]
    # to compute Layer-1 macro deltas (GDP, inflation, FX, spread, oil, unemployment)
    # for the ClimateOrchestrator heatmap/KPIs. Without this key, compute_delta()
    # silently falls back to a 2-field PD/LCR-only proxy (_from_computed_results),
    # leaving every other heatmap column at 0.00 regardless of scenario severity.
    _alias_to_scenario = {
        "baseline": baseline_scn, "adverse": adverse_scn, "severe": severe_scn,
    }
    # Année de choc : last_pd_year (ex. 2025), PAS last_pd_year + 1. Le
    # fichier NGFS démarre ses colonnes-années à 2022, mais l'historique réel
    # (PD, macro observée) va jusqu'à last_pd_year -- réutiliser les années
    # NGFS antérieures (2022 à last_pd_year-1) créerait toujours
    # l'incohérence visuelle documentée ci-dessous (courbe "Adverse/Severe"
    # démarrant en 2022 avec des valeurs qui divergent de la macro RÉELLEMENT
    # observée sur ces années déjà passées), donc ces années restent
    # exclues. MAIS last_pd_year lui-même (ex. 2025) doit être le premier
    # point de la trajectoire NGFS, pas last_pd_year+1 : le "décrochage
    # visuel" entre historique et projection est désormais géré côté
    # affichage (_clmBridgeSeries dans dashboard.html, Row 2) qui tronque
    # l'historique strictement avant la première année NGFS disponible et
    # raccorde la dernière valeur historique réelle au premier point projeté
    # -- il n'est donc plus nécessaire de sacrifier l'année last_pd_year
    # pour éviter le doublon visuel.
    _macro_proj_start = int(last_pd_year)
    macro_trajectories: Dict[str, Dict[int, Dict[str, float]]] = {}
    for _alias, _ngfs_df in ngfs_scenarios.items():
        _scen_name = _alias_to_scenario.get(_alias)
        if not _scen_name:
            continue
        # fallback=None (not the default 0.0): a missing/NaN NGFS value for a
        # given (scenario, year, variable) means "no data", not "value is
        # exactly zero". ClimateMacroAdapter._from_macro_trajectories() only
        # computes a delta when both baseline and stressed values are not
        # None — filling gaps with 0.0 instead produced bogus deltas (e.g.
        # a real baseline oil_price of 97.19 vs a missing/zero-filled
        # stressed value looked like a -97% shock that never happened).
        macro_trajectories[_scen_name] = {
            int(_yr): {str(_c): _safe_float(_v, fallback=None) for _c, _v in _row.items()}
            for _yr, _row in _ngfs_df.to_dict(orient="index").items()
            if int(_yr) >= _macro_proj_start
        }

    return {
        "model": {
            "family":  best["family"],
            "combo":   best_combo,
            "r2":      round(best["r2"], 4),
            "n_obs":   best["n_obs"],
            "verdict": best.get("verdict", "N/A"),
            "n_fail":  best.get("n_fail", 0),
            "n_warn":  best.get("n_warn", 0),
            "n_pass":  best.get("n_pass", 0),
        },
        "scenarios": scenarios_out,
        "metrics": {
            "pd_ttc":            round(pd_ttc, 6),
            "lgd_ttc":           round(lgd_ttc, 4),
            "rho":               round(rho, 6),
            "fj_k":              round(fj_k, 6),
            "pd_baseline_mean":  _mean("baseline"),
            "pd_adverse_mean":   _mean("adverse"),
            "pd_severe_mean":    _mean("severe"),
            "pd_baseline_peak":  _peak("baseline"),
            "pd_adverse_peak":   _peak("adverse"),
            "pd_severe_peak":    _peak("severe"),
        },
        "ngfs_mode": ngfs_mode,
        "macro_trajectories": macro_trajectories,
        # ── Données historiques exposées pour MultiTargetSatelliteFactory ──────
        # macro_df_hist : DataFrame WEO historique complet (toutes années disponibles)
        #                 Colonnes = CREDIT_MACRO_UNIVERSE, index = année int
        # pd_series_hist: pd.Series historique de PD (fraction), index = année int
        #                 Alignée avec macro_df_hist (intersection des années communes)
        # NOTE: ces deux clés restent des objets pandas bruts — climate/wrapper.py
        # les lit en mémoire (_macro_df_hist = _ngfs_credit.get("macro_df_hist"))
        # AVANT toute sérialisation JSON, pour MultiTargetSatelliteFactory. Ne pas
        # les changer de type ici : ça casserait ce chemin de consommation.
        "macro_df_hist":  macro_full,
        "pd_series_hist": target_aligned,
        # macro_hist_by_year : même structure que macro_trajectories
        # ({année: {variable: valeur}}) mais pour l'historique observé — ajouté
        # car macro_df_hist (DataFrame brut) est perdu au moment de la
        # sérialisation JSON du run (converti en simple chaîne de caractères
        # par le fallback default=str, donc inexploitable par le dashboard).
        # Utilisé par le Row 2 du dashboard climat (séries "historique observé").
        "macro_hist_by_year": {
            int(_yr): {str(_c): _safe_float(_v, fallback=None) for _c, _v in _row.items()}
            for _yr, _row in macro_full.to_dict(orient="index").items()
        },
        # capital_hist_by_year : ratios réglementaires (CAR/Tier1/CET1/leverage)
        # observés historiquement — utilisé par la Row 5 du dashboard climat
        # (charts CAR/Tier1 4-séries). Vide si le moteur capital n'a pas tourné.
        "capital_hist_by_year": capital_hist_by_year,
        "capital_trajectories": capital_trajectories,
    }
