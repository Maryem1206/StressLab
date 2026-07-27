"""
Liquidity Risk Wrapper — intégration plateforme multi-risques.

Pipeline :
    1. Charger liquidity_input.xlsx depuis les uploads
    2. Fetch macro via API (WorldBank / IMF WEO) — réutilise l'infra du credit module
    3. Mapper les colonnes macro vers les noms attendus par le moteur liquidité
    4. Construire 3 scénarios (baseline / adverse / severe) via crise historique
    5. Calibrer les satellites (forward stepwise sign-constrained)
    6. Calculer LCR (BCBS 238) et NSFR (BCBS 295) stressés
    7. Validation automatique (5 batteries de tests)
    8. Convertir RiskOutput → PlatformResult

Références :
    BCBS 238 (2013) — Liquidity Coverage Ratio
    BCBS 295 (2014) — Net Stable Funding Ratio
    Burnham & Anderson (2002) — AIC tolerance ΔAIC ≤ 2
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..base import PlatformModule, PlatformResult

LOG = logging.getLogger("liquidity_wrapper")
warnings.filterwarnings("ignore")

# ─── Colonnes macro attendues par le moteur liquidité ─────────────────────────
# Mapping : nom dans le fetcher credit → nom attendu par le moteur liquidité
_MACRO_COLUMN_MAP = {
    "real_gdp_growth":       "GDP_growth",
    "unemployment_rate":     "unemployment_rate",
    "cpi_inflation":         "cpi_inflation",
    "policy_rate":           "policy_rate",
    "exchange_rate_lcu_usd": "exchange_rate",
}

_LIQUIDITY_MACRO_VARS = [
    "GDP_growth", "unemployment_rate", "cpi_inflation",
    "policy_rate", "exchange_rate",
]

# Profil de decay :
#   T+1 : pré-choc — conditions inchangées
#   T+2 : choc plein (pic)
#   T+3 : atténuation (70%)
#   T+4 : retour progressif (30%)
#   T+5+ : décroissance linéaire vers 0 puis maintien à 0
_DECAY_TABLE = {0: 0.0, 1: 1.0, 2: 0.7, 3: 0.3}

def _decay(i: int) -> float:
    if i in _DECAY_TABLE:
        return _DECAY_TABLE[i]
    # Au-delà de T+4 : décroissance linéaire depuis 0.3 vers 0 sur 2 ans puis 0
    return max(0.0, 0.3 - 0.15 * (i - 3))


def _compute_crisis_macro_deltas(
    crisis: Any,
    macro_hist_idx: pd.DataFrame,
) -> Dict[str, float]:
    """
    Calcule les chocs macro d'une crise historique sur les variables liquidité.

    Appelle compute_historical_shocks() du catalogue sur les données macro
    déjà converties en format liquidité (GDP_growth, policy_rate, etc.).
    Retourne des deltas absolus par variable, utilisables directement
    dans _build_one_scenario().

    Si une variable est absente des données historiques (ex. données manquantes
    pour la période pré-crise), son delta est 0 et un warning est loggé.
    """
    from ..core.historical_scenarios import compute_historical_shocks
    raw_deltas = compute_historical_shocks(macro_hist_idx, crisis)
    deltas: Dict[str, float] = {}
    for var in _LIQUIDITY_MACRO_VARS:
        if var in raw_deltas:
            deltas[var] = float(raw_deltas[var])
        else:
            deltas[var] = 0.0
            LOG.warning(
                "Crise '%s' : variable '%s' absente des données historiques "
                "(fenêtre %d–%d manquante) — delta = 0.",
                crisis.name, var, crisis.t_pre_start, crisis.t_peak,
            )
    return deltas


def _shocks_to_macro_deltas(
    shocks: Dict[str, float],
    last_macro_row: "pd.Series",
) -> Dict[str, float]:
    """
    Convertit les chocs génériques du ScenarioManager en deltas absolus
    sur les variables macro du moteur liquidité.

    Clés d'entrée (scenario_manager.py) :
      gdp_delta          : variation fraction (ex. -0.06 = -6 pp de croissance)
      unemployment_delta : variation fraction (ex.  0.04 = +4 pp de chômage)
      inflation_delta    : variation fraction (ex.  0.05 = +5 pp d'inflation)
      rate_bps           : variation en points de base (ex. 300 = +3 pp de taux)
      fx_pct             : fraction, négatif = dépréciation de la monnaie locale
                           (ex. -0.20 → taux de change LCU/USD augmente de 20%)
      oil_pct            : variation fraction du prix du pétrole (proxy non utilisé
                           directement — real_estate_price reste inchangé)
    """
    base_fx = float(last_macro_row.get("exchange_rate", 1.0))
    return {
        "GDP_growth":        shocks.get("gdp_delta", 0.0) * 100,
        "unemployment_rate": shocks.get("unemployment_delta", 0.0) * 100,
        "cpi_inflation":     shocks.get("inflation_delta", 0.0) * 100,
        "policy_rate":       shocks.get("rate_bps", 0.0) / 100,
        "exchange_rate":     -shocks.get("fx_pct", 0.0) * base_fx,
        "real_estate_price": 0.0,  # pas de mapping direct oil_pct → prix immobilier
    }


def _build_one_scenario(
    macro_hist: pd.DataFrame,
    macro_deltas: Dict[str, float],
    horizon_years: List[int],
    projected_baseline: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Construit un DataFrame macro projeté pour UN niveau de sévérité.

    Formule : valeur_projetée(t) = base(t) + delta * decay(t)
    base(t)  = projected_baseline[var][t] si disponible (AR/VAR/VECM via macro_projector)
             = last_observed[var]          sinon (flat fallback)
    decay : {T+1: 0, T+2: 1.0, T+3: 0.7, T+4: 0.3} — montée puis atténuation
    """
    mac = macro_hist.copy()
    if "year" in mac.columns:
        mac = mac.set_index("year")
    last_row = mac.iloc[-1]

    rows = {}
    for i, yr in enumerate(horizon_years):
        d = _decay(i)
        row = {}
        # 5 vars macro stressées (chocs + decay appliqués sur le baseline projeté)
        for var in _LIQUIDITY_MACRO_VARS:
            if (projected_baseline is not None
                    and yr in projected_baseline.index
                    and var in projected_baseline.columns):
                base_val = float(projected_baseline.loc[yr, var])
            else:
                base_val = float(last_row.get(var, 0.0))
            delta = macro_deltas.get(var, 0.0)
            row[var] = base_val + delta * d
        # Toutes les autres colonnes macro : projection plate (dernière valeur observée)
        for col in last_row.index:
            if col not in row:
                try:
                    row[col] = float(last_row[col])
                except (ValueError, TypeError):
                    pass
        rows[yr] = row

    df = pd.DataFrame(rows).T
    df.index.name = "year"
    return df


_LIQUIDITY_FILE_KEYS = (
    "liquidity", "liquidity_input", "liq", "lcr", "nsfr",
    "liquidity_data", "bilan_liquidite", "liquidite",
)

_LIQUIDITY_FILE_KEYWORDS = (
    "liquidity", "liquidite", "liq_input", "lcr_nsfr",
    "liq", "lcr", "nsfr", "bilan_liq",
)


def _find_liquidity_file(upload_paths: Dict, uploads_dir: Path) -> Optional[Path]:
    """Détecte le fichier de données liquidité dans les uploads.

    Stratégie :
      1. Clé exacte dans upload_paths (liste _LIQUIDITY_FILE_KEYS).
      2. Scan du dossier uploads : le stem du fichier contient au moins
         un des mots-clés de _LIQUIDITY_FILE_KEYWORDS.
    """
    # 1. Clé dans upload_paths
    for key in _LIQUIDITY_FILE_KEYS:
        if key in upload_paths:
            p = Path(upload_paths[key])
            if p.exists():
                return p

    # 2. Scan par nom de fichier
    if uploads_dir.exists():
        for p in sorted(uploads_dir.iterdir()):
            if p.suffix.lower() not in (".xlsx", ".xls"):
                continue
            stem_lower = p.stem.lower()
            if any(kw in stem_lower for kw in _LIQUIDITY_FILE_KEYWORDS):
                return p

    return None


def _earliest_crisis_year() -> int:
    """Retourne le t_pre_start minimum sur tout le CRISIS_CATALOG."""
    try:
        from ..core.historical_scenarios import CRISIS_CATALOG
        return min(c.t_pre_start for c in CRISIS_CATALOG.values()) - 2
    except Exception:
        return 1985


def _fetch_and_map_macro(
    country_iso2: str,
    cache_dir: str = "data_cache",
    cache_ttl: int = 30,
    start_year: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Fetch macro via l'infra du credit module et mapper les colonnes.

    start_year remonte jusqu'au minimum t_pre_start du CRISIS_CATALOG
    pour couvrir les fenêtres pré-crise de tous les scénarios historiques.
    """
    if start_year is None:
        start_year = _earliest_crisis_year()
    try:
        from ..core.imf_weo_fetcher import fetch_credit_macro
        macro_full, report = fetch_credit_macro(
            country=country_iso2, start_year=start_year,
            cache_dir=cache_dir, cache_ttl_days=cache_ttl,
        )
        if macro_full.empty:
            return None

        # Mapper les 5 colonnes requises par le moteur
        mapped = pd.DataFrame(index=macro_full.index)
        for src_col, dst_col in _MACRO_COLUMN_MAP.items():
            if src_col in macro_full.columns:
                mapped[dst_col] = macro_full[src_col]

        # Combler les colonnes requises manquantes avec 0
        for col in _LIQUIDITY_MACRO_VARS:
            if col not in mapped.columns:
                mapped[col] = 0.0
                LOG.warning("Colonne macro '%s' non disponible via API -- initialisee a 0", col)

        # Propager TOUTES les colonnes numeriques supplementaires de l API
        # (exclure les colonnes source deja renommees ci-dessus)
        _src_mapped = set(_MACRO_COLUMN_MAP.keys())
        for col in macro_full.columns:
            if col in _src_mapped:
                continue
            if col in mapped.columns:
                continue
            if pd.api.types.is_numeric_dtype(macro_full[col]):
                mapped[col] = macro_full[col]
        LOG.info(
            "Macro pool etendu : %d colonnes transmises au calibrateur "
            "(5 obligatoires + %d supplementaires de l API).",
            len(mapped.columns),
            len(mapped.columns) - len(_LIQUIDITY_MACRO_VARS),
        )

        # Interpoler les NaN (colonnes presentes mais vides) puis forward/back fill
        mapped = mapped.interpolate(method="linear", limit_direction="both")
        mapped = mapped.ffill().bfill().fillna(0.0)

        mapped = mapped.dropna(how="all")
        mapped.index.name = "year"
        return mapped.reset_index()

    except Exception as e:
        LOG.warning("Fetch macro API échoué (%s) — fallback sur stub", e)
        return None






def get_projection_series(charts_data: dict, variable: str = "nsfr"):
    """
    Adapts already-computed liquidity trajectories into a ProjectionSeries,
    for the climate dashboard's Row 3 (NSFR) and Row 5 (LCR) charts.

    Additive only — reads charts_data a LiquidityModuleWrapper.run() call
    already produced (lcr_trajectories / nsfr_trajectories); does not touch
    any existing method.

    Parameters
    ----------
    charts_data : the "charts_data" dict of a liquidity PlatformResult.
    variable    : "lcr" or "nsfr".

    Returns
    -------
    ProjectionSeries | None (None if the underlying trajectory is absent).
    """
    from ..base import projection_series_from_trajectory_dict

    key = "lcr_trajectories" if variable == "lcr" else "nsfr_trajectories"
    traj = (charts_data or {}).get(key, {})
    return projection_series_from_trajectory_dict(traj, variable.upper(), "%")


class LiquidityModuleWrapper(PlatformModule):
    module_id = "liquidity"
    module_label = "Risque de Liquidité"
    is_placeholder = False

    def run(self) -> Dict[str, PlatformResult]:
        p = self.params

        # ══════════════════════════════════════════════════════════════════
        # STEP 0 — IMPORT DYNAMIQUE DU MOTEUR LIQUIDITÉ
        # ══════════════════════════════════════════════════════════════════
        modules_src = p.get("modules_src_dir", "modules_src")
        liq_src = Path(modules_src) / "liquidity_module"
        if str(liq_src.parent) not in sys.path:
            sys.path.insert(0, str(liq_src.parent))

        try:
            from liquidity_module.liquidity_stress_engine import (
                LiquidityDataLoader, LiquidityStressEngine, run_liquidity_stress,
            )
            from liquidity_module.validation_engine import validate_results
        except ImportError as e:
            LOG.error("Import du moteur liquidité échoué: %s", e)
            raise RuntimeError(
                f"Module liquidité introuvable dans {liq_src}. "
                f"Vérifier modules_src/liquidity_module/. Erreur: {e}"
            )

        # ══════════════════════════════════════════════════════════════════
        # STEP 1 — CHARGER LIQUIDITY_INPUT.XLSX
        # ══════════════════════════════════════════════════════════════════
        uploads_dir = Path(p.get("uploads_dir", "uploads"))
        liq_file = _find_liquidity_file(self.upload_paths, uploads_dir)

        if liq_file is None:
            raise FileNotFoundError(
                "Fichier de données liquidité introuvable dans les uploads. "
                f"Noms acceptés (stem du fichier) : {_LIQUIDITY_FILE_KEYWORDS}. "
                f"Clés upload acceptées : {_LIQUIDITY_FILE_KEYS}."
            )

        LOG.info("Fichier liquidité : %s", liq_file)

        # ══════════════════════════════════════════════════════════════════
        # STEP 2 — FETCH MACRO (API ou fallback stub)
        # ══════════════════════════════════════════════════════════════════
        country_iso2 = str(p.get("country_iso2", "EG")).upper()
        cache_dir = str(p.get("cache_dir", "data_cache"))

        macro_hist = _fetch_and_map_macro(
            country_iso2, cache_dir,
            start_year=_earliest_crisis_year(),
        )
        if macro_hist is None or macro_hist.empty:
            raise RuntimeError(
                f"Données macro indisponibles pour le pays '{country_iso2}'. "
                "Vérifiez la connexion à l'API FMI/Banque Mondiale ou le code ISO-2 du pays."
            )

        LOG.info("Macro source: API (%d obs, années %d–%d)",
                 len(macro_hist),
                 int(macro_hist["year"].min()),
                 int(macro_hist["year"].max()))


        # ══════════════════════════════════════════════════════════════════
        # STEP 2.5 — PRE-TESTS MACRO (filtre pertinence + qualité)
        # ══════════════════════════════════════════════════════════════════
        # Exécuté sur toutes les vars macro fetchées avant la projection
        # des satellites. Blocs 0-4 actifs ; Bloc 5 sauté (pas de cible PD).
        _pt_res = None
        try:
            from ..core.macro_pre_tests import run_macro_pre_tests, PreTestConfig
            _mac_idx = macro_hist.set_index('year') if 'year' in macro_hist.columns else macro_hist
            _pt_cfg  = PreTestConfig(hard_filter_irrelevant=True, flag_unknown=True)
            _pt_res  = run_macro_pre_tests(
                macro_df=_mac_idx,
                target_series=None,
                config=_pt_cfg,
                module='liquidity',
            )
            LOG.info(
                'Macro pre-tests liquidité : %d/%d vars acceptées, %d rejetées '
                '(%d hors périmètre stress-test), %d transformées',
                _pt_res.report.get('n_accepted', 0),
                _pt_res.report.get('n_total', 0),
                _pt_res.report.get('n_rejected', 0),
                _pt_res.report.get('n_irrelevant', 0),
                _pt_res.report.get('n_transformed', 0),
            )
        except Exception as _pt_err:
            LOG.warning('Macro pre-tests liquidité échoués (non bloquant): %s', _pt_err)

        # ── Appliquer le filtre Bloc 0 sur macro_hist ─────────────────────────
        # Retirer les vars non pertinentes (population, etc.) du pool macro
        # AVANT de les passer au satellite calibrator.
        # Les 5 vars obligatoires liquidité + "year" sont toujours conservées.
        if _pt_res is not None and _pt_res.accepted:
            _always_keep = set(_LIQUIDITY_MACRO_VARS) | {"year"}
            _accepted_set = set(_pt_res.accepted) | _always_keep
            _drop_cols = [c for c in macro_hist.columns if c not in _accepted_set]
            if _drop_cols:
                macro_hist = macro_hist.drop(columns=_drop_cols)
                LOG.info(
                    "Filtre Bloc 0 appliqué : %d colonnes non pertinentes "
                    "retirées du pool macro avant calibration satellite (ex: %s).",
                    len(_drop_cols), ", ".join(_drop_cols[:5]),
                )


        # ── Bloc 5 : Granger causalité avec run_off_retail comme cible ───────
        # Complète les Blocs 0-4 déjà exécutés. Cible = taux de run-off retail
        # lu directement depuis l'Excel bilan (colonne run_off_retail).
        try:
            import pandas as _pd_b5
            _df_b5 = _pd_b5.read_excel(liq_file, sheet_name=0)
            _col_b5 = next(
                (c for c in _df_b5.columns
                 if str(c).strip().lower() in (
                     "run_off_retail", "run_off", "runoff_retail", "runoff"
                 )),
                None,
            )
            if _col_b5 is not None:
                _year_col = next(
                    (c for c in _df_b5.columns
                     if str(c).strip().lower() in ("year", "annee", "année", "date")),
                    None,
                )
                if _year_col:
                    _b5_ser = (
                        _df_b5[[_year_col, _col_b5]]
                        .dropna()
                        .assign(yr=lambda d: d[_year_col].astype(int))
                        .set_index("yr")[_col_b5]
                        .rename("run_off_retail")
                    )
                    if len(_b5_ser) >= 8:
                        from ..core.macro_pre_tests import run_macro_pre_tests, PreTestConfig
                        _pt_cfg5 = PreTestConfig(
                            hard_filter_irrelevant=False, flag_unknown=False, min_obs=8,
                        )
                        _mac_b5 = (
                            macro_hist.set_index("year")
                            if "year" in macro_hist.columns else macro_hist
                        )
                        _pt5_res = run_macro_pre_tests(
                            macro_df      = _mac_b5,
                            target_series = _b5_ser,
                            config        = _pt_cfg5,
                            module        = 'liquidity',
                        )
                        if _pt5_res.accepted:
                            _b5_always = set(_LIQUIDITY_MACRO_VARS) | {"year"}
                            _b5_keep   = set(_pt5_res.accepted) | _b5_always
                            _b5_drop   = [c for c in macro_hist.columns
                                          if c not in _b5_keep]
                            if _b5_drop:
                                macro_hist = macro_hist.drop(columns=_b5_drop)
                                LOG.info(
                                    "Bloc 5 liquidité (Granger→run_off_retail) : "
                                    "%d vars non causales retirées du pool (ex: %s).",
                                    len(_b5_drop), ", ".join(_b5_drop[:5]),
                                )
        except Exception as _b5_liq_err:
            LOG.warning("Bloc 5 liquidité échoué (non bloquant): %s", _b5_liq_err)

        # ══════════════════════════════════════════════════════════════════
        # STEP 3 — CONSTRUIRE LES SCÉNARIOS
        # ══════════════════════════════════════════════════════════════════
        # horizon_start/horizon_end DOIVENT venir des params (définis dans dashboard.py).
        # Ne pas utiliser macro_hist.year.max() comme base : l'API IMF retourne des
        # prévisions jusqu'en 2031, ce qui décalerait le choc à 2032.
        # Fallback d'urgence : année courante + 1 si les params sont absents.
        import datetime as _dt
        _current_year = _dt.datetime.now().year
        horizon_start = int(p.get("horizon_start", _current_year - 1))
        horizon_end   = int(p.get("horizon_end",   horizon_start + 4))
        horizon = list(range(horizon_start, horizon_end))

        scenario_levels = self.scenario.get("levels", {})
        adv_cfg = scenario_levels.get("adverse", {})

        # ── Detect idiosyncratic scenario (PATH D) ───────────────────────────
        has_idio_shocks = any(k in adv_cfg for k in
            ("hqla_coef", "outflow_coef", "asf_coef", "rsf_coef",
             "npl_add_pp", "capital_coef", "rwa_coef"))

        has_macro_shocks = any(k in adv_cfg for k in
            ("gdp_delta", "rate_bps", "unemployment_delta", "inflation_delta", "fx_pct"))

        mac_idx = macro_hist.set_index("year") if "year" in macro_hist.columns else macro_hist
        last_macro_row = mac_idx.iloc[-1]

        scenarios: Dict = {}

        # ── STEP 2.7 : Projection macro baseline (AR / VAR / VECM) ──────────
        # Calcule une trajectoire projetée pour les 5 vars liquidité sur l'horizon.
        # Utilisée comme base dans _build_one_scenario() à la place de last_row flat.
        _proj_baseline: Optional[pd.DataFrame] = None
        try:
            from ..core.macro_projector import project_macro as _proj_macro_liq
            _mac_idx_proj = (
                macro_hist.set_index("year") if "year" in macro_hist.columns
                else macro_hist
            )
            _liq_proj_vars = [v for v in _LIQUIDITY_MACRO_VARS
                              if v in _mac_idx_proj.columns
                              and not _mac_idx_proj[v].dropna().empty]
            if _liq_proj_vars:
                _liq_proj_res = _proj_macro_liq(
                    historical    = _mac_idx_proj,
                    variables     = _liq_proj_vars,
                    horizon_years = horizon,
                )
                _proj_baseline = _liq_proj_res.projected
                LOG.info(
                    "macro_projector liquidité : %d vars projetées via %s (horizon=%s)",
                    len(_liq_proj_vars), _liq_proj_res.model_used, horizon,
                )
        except Exception as _liq_proj_err:
            LOG.warning(
                "macro_projector liquidité échoué (non bloquant) — "
                "fallback projection plate : %s", _liq_proj_err
            )

        if has_idio_shocks:
            # PATH D: Idiosyncratic — engine runs on zero-macro for all levels;
            # ratio shocks are applied after the engine returns results.
            _zero_deltas = {v: 0.0 for v in _LIQUIDITY_MACRO_VARS}
            _zero_scn = _build_one_scenario(
                macro_hist, _zero_deltas, horizon,
                projected_baseline=_proj_baseline,
            )
            scenarios = {"baseline": _zero_scn, "adverse": _zero_scn, "severe": _zero_scn}
            LOG.info(
                "Scénario IDIOSYNCRATIQUE liquidité: satellite macro bypassed — "
                "LCR/NSFR seront ajustés directement post-engine."
            )

        else:
            # ── Résoudre le scénario crise ────────────────────────────────────
            # Priorité PATH B/C > PATH A
            catalog_crisis_name = (
                str(self.scenario.get("crisis_name", ""))
                or str(p.get("crisis_name", ""))
            ).strip()

            crisis_deltas_1x: Optional[Dict[str, float]] = None
            if catalog_crisis_name:
                try:
                    from ..core.historical_scenarios import get_crisis
                    crisis = get_crisis(catalog_crisis_name)
                    mac_idx_year = (
                        macro_hist.set_index("year")
                        if "year" in macro_hist.columns
                        else macro_hist
                    )
                    crisis_deltas_1x = _compute_crisis_macro_deltas(crisis, mac_idx_year)
                    LOG.info(
                        "PATH B/C: chocs liquidité calculés depuis données API pour "
                        "'%s' (%s) — fenêtre [%d–%d] → pic %d. "
                        "Deltas: GDP=%+.2f pp, chômage=%+.2f pp, CPI=%+.2f pp, "
                        "taux=%+.2f pp, FX=%+.2f.",
                        catalog_crisis_name, crisis.description,
                        crisis.t_pre_start, crisis.t_pre_end, crisis.t_peak,
                        crisis_deltas_1x.get("GDP_growth", 0.0),
                        crisis_deltas_1x.get("unemployment_rate", 0.0),
                        crisis_deltas_1x.get("cpi_inflation", 0.0),
                        crisis_deltas_1x.get("policy_rate", 0.0),
                        crisis_deltas_1x.get("exchange_rate", 0.0),
                    )
                except KeyError:
                    from ..core.historical_scenarios import CRISIS_CATALOG
                    LOG.warning(
                        "PATH B/C: '%s' absent du CRISIS_CATALOG "
                        "(disponibles : %s) — fallback PATH A.",
                        catalog_crisis_name, ", ".join(CRISIS_CATALOG.keys()),
                    )
                except Exception as e:
                    LOG.error(
                        "PATH B/C: calcul des chocs pour '%s' échoué (%s) "
                        "— données API insuffisantes pour la fenêtre de crise? "
                        "Fallback PATH A.",
                        catalog_crisis_name, e,
                    )

            for level in ("baseline", "adverse", "severe"):
                if level == "baseline":
                    level_deltas = {v: 0.0 for v in _LIQUIDITY_MACRO_VARS}

                elif crisis_deltas_1x is not None:
                    # PATH B/C prioritaire : données historiques réelles pays/crise
                    scale = 0.5 if level == "adverse" else 1.0
                    level_deltas = {k: v * scale for k, v in crisis_deltas_1x.items()}

                elif has_macro_shocks:
                    # PATH A fallback : chocs pré-définis (scénarios sans entrée catalogue)
                    level_shocks = scenario_levels.get(level, {})
                    level_deltas = _shocks_to_macro_deltas(level_shocks, last_macro_row)

                else:
                    LOG.error(
                        "Liquidité : aucun choc pour le niveau '%s' "
                        "(pas de crisis_name valide, pas de macro_shocks). "
                        "Vérifier la configuration du scénario.",
                        level,
                    )
                    level_deltas = {v: 0.0 for v in _LIQUIDITY_MACRO_VARS}

                scenarios[level] = _build_one_scenario(
                    macro_hist, level_deltas, horizon,
                    projected_baseline=_proj_baseline,
                )

        LOG.info("Horizon: %s | Scenario: %s", horizon, self.scenario.get("label", "N/A"))

        # ══════════════════════════════════════════════════════════════════
        # STEP 4 — RUN LIQUIDITY STRESS ENGINE
        # ══════════════════════════════════════════════════════════════════
        bank_name      = str(p.get("bank_name", ""))
        country        = str(p.get("country", ""))
        portfolio_type = str(p.get("portfolio_type", "mixed"))

        # Balance sheet stress deltas (optionnel — override des défauts du moteur)
        _bs_liab  = p.get("bs_stress_delta_liab")   # ex. {"baseline":0,"adverse":-0.02,"severe":-0.05}
        _bs_asset = p.get("bs_stress_delta_asset")   # ex. {"baseline":0,"adverse":+0.02,"severe":+0.04}

        LOG.info(
            "Liquidity engine params: portfolio_type='%s' | "
            "bs_stress_liab=%s | bs_stress_asset=%s",
            portfolio_type, _bs_liab, _bs_asset,
        )

        # For idiosyncratic PATH D the balance-sheet stress deltas must be zeroed out.
        # Keeping the default deltas (liab -1%/-3%, asset +1%/+3%) causes HQLA items
        # (L1, L2a) and deposit liabilities to shrink at the same rate, but since
        # HQLA << deposits for most banks, the NCO contraction dominates and LCR
        # paradoxically *rises* under stress. All ratio stress is handled by the
        # post-engine multipliers in STEP 4b below.
        _eff_bs_liab  = ({"baseline": 0.0, "adverse": 0.0, "severe": 0.0}
                         if has_idio_shocks else _bs_liab)
        _eff_bs_asset = ({"baseline": 0.0, "adverse": 0.0, "severe": 0.0}
                         if has_idio_shocks else _bs_asset)

        _forced_sat_ranks = {
            k: int(v) for k, v in (p.get("forced_sat_ranks") or {}).items()
        }

        liq_outputs, engine = run_liquidity_stress(
            input_path=str(liq_file),
            macro_hist=macro_hist,
            scenarios=scenarios,
            bank_name=bank_name,
            country=country,
            verbose=False,
            return_engine=True,
            portfolio_type=portfolio_type,
            bs_stress_delta_liab=_eff_bs_liab,
            bs_stress_delta_asset=_eff_bs_asset,
            forced_sat_ranks=_forced_sat_ranks,
        )

        # ══════════════════════════════════════════════════════════════════
        # STEP 4b — IDIOSYNCRATIC: apply direct ratio shocks post-engine
        # ══════════════════════════════════════════════════════════════════
        # The engine ran on zero-macro scenarios for all levels.
        # Now we apply the user-calibrated balance-sheet multipliers:
        #   LCR_stressed  = LCR_base  × (1 − hqla_coef)  / (1 + outflow_coef)
        #   NSFR_stressed = NSFR_base × (1 − asf_coef)   / (1 + rsf_coef)
        # This is applied with the standard decay profile over the horizon.
        if has_idio_shocks:
            for level in ("adverse", "severe"):
                shocks      = scenario_levels.get(level, {})
                hqla_coef   = float(shocks.get("hqla_coef",   0.0))
                outflow_cf  = float(shocks.get("outflow_coef", 0.0))
                asf_coef    = float(shocks.get("asf_coef",     0.0))
                rsf_coef    = float(shocks.get("rsf_coef",     0.0))

                if all(c == 0.0 for c in (hqla_coef, outflow_cf, asf_coef, rsf_coef)):
                    continue

                # Full-stress multipliers (applied at peak, i.e. T+2)
                lcr_full_mult  = (1.0 - hqla_coef)  / max(1.0 + outflow_cf, 1e-6)
                nsfr_full_mult = (1.0 - asf_coef)   / max(1.0 + rsf_coef,   1e-6)

                out = liq_outputs[level]
                ts  = out.time_series.copy()

                # Apply multipliers with the standard decay profile so the shock
                # peaks at T+2 (shock year) and attenuates afterwards, matching
                # the behaviour of macro-driven scenarios.
                # effective_mult(i) = 1 − decay(i) × (1 − full_mult)
                #   i=0 → decay=0.0 → mult=1.0  (no change pre-shock)
                #   i=1 → decay=1.0 → mult=full_mult (peak stress)
                #   i=2 → decay=0.7 → partial recovery
                for i, idx in enumerate(ts.index):
                    d = _decay(i)
                    yr_lcr_mult  = 1.0 - d * (1.0 - lcr_full_mult)
                    yr_nsfr_mult = 1.0 - d * (1.0 - nsfr_full_mult)
                    if "lcr"  in ts.columns:
                        ts.at[idx, "lcr"]  = ts.at[idx, "lcr"]  * yr_lcr_mult
                    if "nsfr" in ts.columns:
                        ts.at[idx, "nsfr"] = ts.at[idx, "nsfr"] * yr_nsfr_mult

                out.time_series = ts

                if "lcr"  in ts.columns:
                    out.metrics["lcr_stressed"]  = round(float(ts["lcr"].min()),  4)
                if "nsfr" in ts.columns:
                    out.metrics["nsfr_stressed"] = round(float(ts["nsfr"].min()), 4)
                out.metrics["hqla_delta"] = -hqla_coef
                out.metrics["nco_delta"]  = outflow_cf

                lcr_bl = out.metrics.get("lcr_baseline", 100.0)
                lcr_st = out.metrics.get("lcr_stressed", 100.0)
                out.loss = max(0.0, (lcr_bl - lcr_st) * 0.01)

                # Detect breach year post-adjustment
                if "lcr" in ts.columns and "year" in ts.columns:
                    breach = next(
                        (int(r["year"]) for _, r in ts.iterrows() if r["lcr"] < 100),
                        None,
                    )
                    out.metrics["breach_year"] = breach

                LOG.info(
                    "Idiosyncratic liq [%s]: hqla=%.3f outflow=%.3f asf=%.3f rsf=%.3f"
                    " → LCR×%.3f NSFR×%.3f (stressed_lcr=%.1f%% stressed_nsfr=%.1f%%)",
                    level, hqla_coef, outflow_cf, asf_coef, rsf_coef,
                    lcr_mult, nsfr_mult,
                    out.metrics.get("lcr_stressed", float("nan")),
                    out.metrics.get("nsfr_stressed", float("nan")),
                )

        # ══════════════════════════════════════════════════════════════════
        # STEP 5 — VALIDATION AUTOMATIQUE
        # ══════════════════════════════════════════════════════════════════
        try:
            report = validate_results(
                liq_outputs, engine, macro_hist,
                bank_name=bank_name, country=country,
            )
            validation_summary = {
                "passed": report.passed,
                "total": report.total_tests,
                "pass_count": report.passed_tests,
                "fail_count": report.failed_tests,
                "warn_count": report.warnings,
            }
            LOG.info(
                "Validation: %d/%d PASS | %d WARN | %d FAIL → %s",
                report.passed_tests, report.total_tests,
                report.warnings, report.failed_tests,
                "✅ VALIDÉ" if report.passed else "❌ NON VALIDÉ",
            )
        except Exception as e:
            LOG.warning("Validation échouée: %s", e)
            validation_summary = {"passed": None, "error": str(e)}

        # ══════════════════════════════════════════════════════════════════
        # STEP 6 — CONVERSION RiskOutput → PlatformResult
        # ══════════════════════════════════════════════════════════════════
        results: Dict[str, PlatformResult] = {}

        # Collecter les trajectoires pour charts_data (toutes les sévérités)
        all_lcr = {}
        all_nsfr = {}

        # Série historique (bilan + macro réels, pas de stress)
        try:
            hist = engine.compute_historical_ratios()
            hist_x = [str(y) for y in hist["years"]]
            all_lcr["historical"]  = {"x": hist_x, "y": hist["lcr"]}
            all_nsfr["historical"] = {"x": hist_x, "y": hist["nsfr"]}
        except Exception as e:
            LOG.warning("Série historique LCR/NSFR non disponible: %s", e)

        for level, out in liq_outputs.items():
            ts = out.time_series

            proj_years = ts["year"].astype(int).tolist() if "year" in ts.columns else horizon
            proj_x    = [str(y) for y in proj_years]
            lcr_proj  = ts["lcr"].tolist()  if "lcr"  in ts.columns else []
            nsfr_proj = ts["nsfr"].tolist() if "nsfr" in ts.columns else []

            all_lcr[level]  = {"x": proj_x, "y": lcr_proj}
            all_nsfr[level] = {"x": proj_x, "y": nsfr_proj}

        # Construire les calibration results pour charts
        try:
            cal_report = engine.calibration_report()
            model_table = cal_report.to_dict(orient="records")
        except Exception:
            model_table = []

        # Rich per-satellite diagnostics: drivers, R², AIC, p-values, sign checks
        satellite_table = []
        _sat_vreports: list = []
        try:
            from app.modules.core.post_estimation_validator import (
                PostEstimationValidator, RiskOutput,
            )
            for var, sat in engine._sat_results.items():
                # Pull pre-computed verdict from tournament winner entry (rank 0)
                _sat_lb = getattr(
                    getattr(engine, "_calibrator", None),
                    "tournament_leaderboards", {},
                ).get(var, [])
                _winner_lb = _sat_lb[0] if _sat_lb else {}

                satellite_table.append({
                    "satellite":     var,
                    "family":        sat.family,
                    "drivers":       sat.drivers,
                    "coefficients":  {k: round(float(v), 4) for k, v in sat.coefficients.items()},
                    "std_errors":    {k: round(float(v), 4) for k, v in sat.std_errors.items()},
                    "pvalues":       {k: round(float(v), 4) for k, v in sat.pvalues.items()},
                    "r2":            round(float(sat.r2), 4),
                    "aic":           round(float(sat.aic), 2),
                    "bic":           round(float(sat.bic), 2),
                    "n_obs":         int(sat.n_obs),
                    "converged":     bool(sat.converged),
                    "signs_ok":      bool(sat.signs_ok),
                    "sign_checks":   {k: bool(v) for k, v in sat.sign_checks.items()},
                    "years":         sat.obs_years or [],
                    "actual":        [round(float(v), 6) for v in (sat.actual_values or [])],
                    "fitted":        [round(float(v), 6) for v in (sat.fitted_values or [])],
                    "verdict":       _winner_lb.get("verdict", "VALIDATED_WITH_WARNINGS"),
                    "n_fail":        _winner_lb.get("n_fail", 0),
                    "n_warn":        _winner_lb.get("n_warn", 0),
                    "n_pass":        _winner_lb.get("n_pass", 0),
                })
                # ── Post-Estimation Validation (per satellite) ────────────
                try:
                    _yt = np.asarray(sat.actual_values or [], dtype=float)
                    _yp = np.asarray(sat.fitted_values or [], dtype=float)
                    if len(_yt) > 0 and len(_yp) > 0:
                        _ro = RiskOutput(
                            module     = "liquidity",
                            model_type = sat.family,
                            y_true     = _yt,
                            y_pred     = _yp,
                            residuals  = _yt - _yp,
                            n_obs      = int(sat.n_obs),
                            k_params   = len(sat.drivers),
                            bounds     = (0.0, 1.0),
                        )
                        _vr = PostEstimationValidator(_ro).run()
                        _sat_vreports.append({
                            "satellite": var,
                            "report":    _vr.to_dict(),
                        })
                        LOG.info(
                            "PostEstimation [liquidity/%s/%s]: %d P %d W %d F — %s",
                            var, sat.family,
                            _vr.n_pass, _vr.n_warn, _vr.n_fail, _vr.verdict,
                        )
                        for _tr in _vr.results:
                            if _tr.status in ("WARN", "FAIL"):
                                LOG.warning(
                                    "  [%s] %s %s (%s): %s",
                                    _tr.status, _tr.test_id, _tr.test_name,
                                    var, _tr.message,
                                )
                except Exception as _ve:
                    LOG.warning("PostEstimationValidator (liquidity/%s): %s", var, _ve)
        except Exception as _e:
            LOG.warning("Satellite table build failed: %s", _e)
            satellite_table = []

        # Full tournament leaderboard (all combinations tested, ranked, per satellite)
        tournament_table: List[Dict] = []
        try:
            calibrator = engine._calibrator
            for var_name, lb in calibrator.tournament_leaderboards.items():
                tournament_table.extend(lb)
        except Exception as _e:
            LOG.warning("Tournament table build failed: %s", _e)
            tournament_table = []

        for level, out in liq_outputs.items():
            ts = out.time_series
            m = out.metrics

            results[level] = PlatformResult(
                module_id=self.module_id,
                module_label=self.module_label,
                scenario_id=level,
                total_loss=out.loss,
                kpis={
                    "lcr_baseline":  m.get("lcr_baseline", 0),
                    "lcr_stressed":  m.get("lcr_stressed", 0),
                    "nsfr_baseline": m.get("nsfr_baseline", 0),
                    "nsfr_stressed": m.get("nsfr_stressed", 0),
                    "breach_year":   m.get("breach_year"),
                    "hqla_delta":    m.get("hqla_delta", 0),
                    "nco_delta":     m.get("nco_delta", 0),
                },
                time_series=ts,
                charts_data={
                    "lcr_trajectories":   all_lcr,
                    "nsfr_trajectories":  all_nsfr,
                    "model_table":        model_table,
                    "satellite_table":    satellite_table,
                    "tournament_table":   tournament_table,
                    "validation_reports": _sat_vreports,
                    "selected_sat_ranks": _forced_sat_ranks,
                },
                metadata={
                    "crisis_name":     self.scenario.get("label", "N/A"),
                    "crisis_deltas":   scenario_levels,
                    "country_iso2":    country_iso2,
                    "bank_name":       bank_name,
                    "macro_source":    "api",
                    "horizon":         horizon,
                    "rsf_weights":     m.get("rsf_weights", {}),
                    "portfolio_type":  portfolio_type,
                    "rsf_path":        out.metadata.get("rsf_path", "unknown"),
                    "validation":      validation_summary,
                },
            )

        return results

    def get_kpis(self, results: Dict) -> Dict[str, Any]:
        out = {}
        for level, r in results.items():
            for k, v in r.kpis.items():
                out[f"{level}_{k}"] = v
        return out



