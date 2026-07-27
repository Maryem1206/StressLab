"""Climate / ESG module wrapper.
Exposes:
  - model_table         : full model comparison table (for Step 3a UI)
  - scenario_combos     : all ranked pairs (adverse, severe, score) (for Step 3b UI)
  - forced_model_rank   : int — force a specific model (0-based)
  - ngfs_adverse_forced / ngfs_severe_forced — force scenario pair
  - ngfs_mode           : "LT" | "CT"
"""
from __future__ import annotations
import logging, sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from ..base import PlatformModule, PlatformResult

LOG = logging.getLogger("climate_wrapper")
_SRC = str(Path(__file__).resolve().parent.parent.parent.parent /
           "modules_src" / "climate_module")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

NGFS_SCENARIOS_LT = [
    "Baseline", "Net Zero 2050", "Below 2°C", "Delayed Transition",
    "Divergent Net Zero", "Nationally Determined Contributions (NDCs)",
    "Current Policies", "Fragmented World",
]
# CT (Current Transition, GEM-E3-based) scenarios use a completely different
# naming scheme from LT (NiGEM-based) — these are FAMILY PREFIXES, not
# exhaustive scenario identifiers. Real files carry geographic variants
# (DAPS_AFR_R, DAPS_ASIA, DAPS_EUR, ...) and stochastic runs (DIRE_run1, ...) —
# see scenario_family_filter.py, which collapses these down to one
# representative per family at runtime. The actual scenario names always come
# from the uploaded GEM-E3 file, never from a hardcoded list.
NGFS_SCENARIOS_CT = ["Baseline", "DAPS", "DIRE", "HWTP", "SWUC"]


class ClimateModuleWrapper(PlatformModule):
    module_id     = "climate"
    module_label  = "Risque Climatique / ESG"
    is_placeholder = False

    def run(self) -> Dict[str, PlatformResult]:
        try:
            from macro_selection_engine import EngineConfig, run_three_scenarios
        except ImportError as e:
            LOG.warning("Cannot import climate engine: %s", e)
            return self._err(str(e))

        p = self.params

        # ── Mode LT / CT ─────────────────────────────────────────────
        ngfs_mode = str(p.get("ngfs_mode", "LT")).upper()
        if ngfs_mode not in ("LT", "CT"):
            ngfs_mode = "LT"

        # ── NGFS file paths ──────────────────────────────────────────
        ngfs_path    = self.upload_paths.get("ngfs", "")
        ngfs_ct_path = self.upload_paths.get("ngfs_ct", "")

        if ngfs_mode == "LT" and not (ngfs_path and Path(ngfs_path).exists()):
            return self._err("Fichier NGFS LT manquant.")
        if ngfs_mode == "CT" and not (ngfs_ct_path and Path(ngfs_ct_path).exists()):
            return self._err("Fichier NGFS CT manquant.")

        # ── Scenario pair — ScenarioManager drives the selection ────────
        # Priority: explicit Phase-3b param override
        #        >  ScenarioManager-configured scenario (levels.adverse/severe)
        #        >  engine auto-selection (fallback when both are absent)
        levels = self.scenario.get("levels", {})
        baseline_scn = levels.get("baseline", {}).get("ngfs_scenario", "Baseline")

        ngfs_adverse_forced = (
            p.get("ngfs_adverse_forced", "")
            or levels.get("adverse", {}).get("ngfs_scenario", "")
        )
        ngfs_severe_forced = (
            p.get("ngfs_severe_forced", "")
            or levels.get("severe", {}).get("ngfs_scenario", "")
        )
        LOG.info(
            "ClimateWrapper: scenario_label=%r baseline=%r "
            "adverse=%r severe=%r (source=%s)",
            self.scenario.get("label", "?"),
            baseline_scn, ngfs_adverse_forced, ngfs_severe_forced,
            "param_override" if p.get("ngfs_adverse_forced")
            else ("scenario_manager" if ngfs_adverse_forced else "engine_auto"),
        )

        # ── Forced model rank ────────────────────────────────────────
        _raw_forced_model_rank = p.get("forced_model_rank", None)
        forced_model_rank = _raw_forced_model_rank
        if forced_model_rank is not None:
            try:   forced_model_rank = int(forced_model_rank)
            except: forced_model_rank = None
        LOG.info(
            "ClimateWrapper: forced_model_rank raw=%r (type=%s) → parsed=%r",
            _raw_forced_model_rank, type(_raw_forced_model_rank).__name__,
            forced_model_rank,
        )

        # ── Validate historical file before launching engine ─────────
        target_var  = p.get("target_variable", "Default rate")
        hist_path   = self.upload_paths.get(
                        "historical", str(Path(_SRC) / "mon_historique.csv"))
        _hist_ok = self._check_historical_file(hist_path, target_var)
        if _hist_ok is not None:
            return self._err(_hist_ok)

        # ── EngineConfig ─────────────────────────────────────────────
        kw = dict(
            country            = p.get("country", "Egypt"),
            baseline_scenario  = baseline_scn,
            risk_channel       = p.get("risk_channel", "combined"),
            target_variable    = target_var,
            risk_type          = "credit",
            historical_path    = hist_path,
            mapping_yaml_path  = self.upload_paths.get(
                                    "mapping",
                                    str(Path(_SRC) / "mapping.yaml")),
            output_path        = "/tmp/climate_result.json",
            auto_accept_mapping   = True,
            auto_select_model     = True,
            auto_accept_scenarios = True,
            min_abs_correlation= float(p.get("min_abs_correlation", 0.10)),
            vif_threshold      = float(p.get("vif_threshold", 10.0)),
            pvalue_threshold   = float(p.get("pvalue_threshold", 0.20)),
            max_combo_size     = int(p.get("max_combo_size", 3)),
            max_candidate_vars = int(p.get("max_candidate_vars", 6)),
            top_n_models       = int(p.get("top_n_models", 10)),
            ngfs_mode          = ngfs_mode,
            ngfs_path          = ngfs_path,
        )
        if ngfs_mode == "CT":
            kw["ngfs_ct_path"] = ngfs_ct_path
        if ngfs_adverse_forced:
            kw["forced_adverse_scenario"] = ngfs_adverse_forced
        if ngfs_severe_forced:
            kw["forced_severe_scenario"]  = ngfs_severe_forced
        if forced_model_rank is not None:
            kw["forced_model_rank"] = forced_model_rank

        # ── Cross-risk unanimous variable restriction (climate only) ──────
        # Restrict Stage 2's candidate historical columns to variables
        # independently accepted by PD (crédit), AND at least one liquidity
        # target (LCR/NSFR), AND at least one market beta factor — so the
        # same macro shock story drives credit/liquidity/market stress under
        # a given NGFS scenario, instead of each module picking its own
        # best-fit variables in isolation. Best-effort: any failure here
        # (missing files, insufficient history) degrades to no restriction,
        # never blocks the climate run.
        try:
            from macro_selection_engine.cross_risk_selector import (
                select_unanimous_variables,
            )
            from app.modules.credit.wrapper import _load_credit_dataset
            from app.modules.core.imf_weo_fetcher import fetch_credit_macro

            _cr_country = (
                p.get("country") or p.get("country_iso2")
                or p.get("country_code") or "EG"
            )
            _COUNTRY_TO_ISO2_CR = {
                "egypt": "EG", "morocco": "MA", "tunisia": "TN", "nigeria": "NG",
                "south africa": "ZA", "senegal": "SN", "kenya": "KE", "ghana": "GH",
                "france": "FR", "germany": "DE", "united kingdom": "GB",
            }
            if len(_cr_country) > 3:
                _cr_country = _COUNTRY_TO_ISO2_CR.get(
                    _cr_country.lower(), _cr_country[:2].upper())

            _pd_series, _, _last_pd_year = _load_credit_dataset(self.upload_paths)
            _macro_full, _ = fetch_credit_macro(
                country        = _cr_country,
                start_year     = int(p.get("history_start", 1990)),
                cache_dir      = p.get("cache_dir", "data_cache"),
                cache_ttl_days = int(p.get("cache_ttl_days", 30)),
            )
            _macro_hist = _macro_full.loc[_macro_full.index <= _last_pd_year]

            _liq_hist  = _extract_liquidity_history(self.upload_paths, p)
            _beta_hist = _extract_market_betas(self.upload_paths, p)

            _restrict_vars = None
            if _macro_hist is not None and not _macro_hist.empty and not _pd_series.empty:
                _restrict_vars = select_unanimous_variables(
                    macro_df          = _macro_hist,
                    pd_series         = _pd_series,
                    liquidity_targets = _liq_hist,
                    market_targets    = _beta_hist,
                )
            if _restrict_vars:
                kw["restrict_candidate_vars"] = _restrict_vars
            else:
                LOG.info(
                    "ClimateWrapper: sélection cross-risque vide ou non "
                    "calculable — pas de restriction appliquée."
                )
        except Exception as _cre:
            LOG.warning(
                "ClimateWrapper: sélection cross-risque échouée (ignorée): %s",
                _cre,
            )

        # ── Multi-risk scenario pair selection: calibrate liquidity engine
        # and market beta1 satellite (best-effort, non-blocking) so the
        # adverse/severe pair is chosen using credit + liquidity + market
        # signals jointly instead of credit PD alone. Any failure here
        # degrades to fewer risks inside run_three_scenarios() itself.
        _liquidity_engine_for_selection = _calibrate_liquidity_engine(self.upload_paths, p)
        _beta1_sat, _last_beta1 = _calibrate_market_beta1(self.upload_paths, p)

        try:
            cfg = EngineConfig(**kw)
            LOG.info(
                "ClimateWrapper: cfg built — forced_model_rank=%r "
                "forced_adverse_scenario=%r forced_severe_scenario=%r",
                getattr(cfg, "forced_model_rank", "MISSING_FIELD"),
                getattr(cfg, "forced_adverse_scenario", "MISSING_FIELD"),
                getattr(cfg, "forced_severe_scenario", "MISSING_FIELD"),
            )
            raw = run_three_scenarios(
                cfg,
                liquidity_engine=_liquidity_engine_for_selection,
                market_beta1_satellite=_beta1_sat,
                market_last_beta1=_last_beta1,
            )
        except TypeError as e:
            # EngineConfig may not accept forced_* — retry without them
            LOG.warning("Retry without forced scenarios: %s", e)
            kw.pop("forced_adverse_scenario", None)
            kw.pop("forced_severe_scenario",  None)
            try:
                cfg = EngineConfig(**kw)
                raw = run_three_scenarios(
                    cfg,
                    liquidity_engine=_liquidity_engine_for_selection,
                    market_beta1_satellite=_beta1_sat,
                    market_last_beta1=_last_beta1,
                )
            except Exception as e2:
                return self._err(str(e2))
        except Exception as e:
            return self._err(str(e))

        # ── Apply forced model rank ──────────────────────────────────
        if forced_model_rank is not None:
            model_table = raw.get("model_comparison_table", [])
            if model_table and 0 <= forced_model_rank < len(model_table):
                raw["selected_model"] = model_table[forced_model_rank]
                LOG.info("Forced model rank %d: %s",
                         forced_model_rank,
                         raw["selected_model"].get("family")
                         or raw["selected_model"].get("modele", "?"))
            insample_fits = raw.get("insample_fits", [])
            if insample_fits and 0 <= forced_model_rank < len(insample_fits):
                raw["insample_fit"] = insample_fits[forced_model_rank]
                LOG.info("Forced insample_fit to rank %d", forced_model_rank)

        selected_rank_0idx = forced_model_rank if forced_model_rank is not None else 0
        lightweight = bool(p.get("_climate_lightweight_phase1", False))
        return self._parse(raw, ngfs_mode, selected_rank_0idx, lightweight=lightweight)

    # ── Parse raw output ─────────────────────────────────────────────
    def _parse(self, raw, ngfs_mode="LT", selected_rank_0idx=0, lightweight=False):
        results = {}
        p = self.params
        sm = raw.get("selected_model") or {}
        ss = raw.get("scenario_selection", {})

        # ── Model table (Step 3a) ────────────────────────────────────
        model_table  = raw.get("model_comparison_table", [])
        insample_fit = raw.get("insample_fit", {})

        # ── Stationarity report (ADF/KPSS, Stage 2) ───────────────────
        # Reflects the REAL, verified stationarity of the series each
        # variable's model actually uses: select_best_transforms() picks the
        # transform by correlation with the target, then tests ADF/KPSS on
        # that exact series, differencing iteratively (up to d=2) if it is
        # not stationary. Not a pre-selection assumption on raw levels.
        pretest_report = raw.get("stage2_historical", {}).get("stationarity_report", {})

        # ── Scenario combos (Step 3b) ────────────────────────────────
        # "ranking" is list of {adverse, severe, score}
        scenario_combos = ss.get("ranking", [])
        auto_adverse    = ss.get("selected_adverse", "")
        auto_severe     = ss.get("selected_severe",  "")
        auto_score      = ss.get("score", None)
        model_used      = ss.get("model_used", "N/A")

        # Fallback: if the engine didn't report selected scenarios,
        # use what ScenarioManager configured (levels.adverse/severe)
        _scen_levels = self.scenario.get("levels", {})
        if not auto_adverse:
            auto_adverse = (
                p.get("ngfs_adverse_forced", "")
                or _scen_levels.get("adverse", {}).get("ngfs_scenario", "")
            )
        if not auto_severe:
            auto_severe = (
                p.get("ngfs_severe_forced", "")
                or _scen_levels.get("severe", {}).get("ngfs_scenario", "")
            )
        if auto_adverse or auto_severe:
            LOG.info(
                "ClimateWrapper._parse: using adverse=%r severe=%r",
                auto_adverse, auto_severe,
            )

        # Build PD trajectories per level
        all_pd = {}
        for lvl in ("baseline", "adverse", "severe"):
            pts = raw.get("projections", {}).get(lvl, {}).get("path", [])
            all_pd[lvl] = {
                "x": [p["year"]  for p in pts if p.get("value") is not None],
                "y": [p["value"] for p in pts if p.get("value") is not None],
            }

        # ── Historical PD for chart display ──────────────────────────
        hist_x, hist_y = [], []
        try:
            hist_path = self.upload_paths.get(
                "historical", str(Path(_SRC) / "mon_historique.csv"))
            target_var = p.get("target_variable", "Default rate")
            _head = Path(hist_path).read_text(encoding="latin-1")[:500]
            _sep  = ";" if _head.count(";") > _head.count(",") else ","
            _hdf  = pd.read_csv(hist_path, sep=_sep, encoding="latin-1")
            _hdf.columns = [str(c).strip() for c in _hdf.columns]
            # Year column
            _yr = next((c for c in _hdf.columns if "year" in c.lower()), _hdf.columns[0])
            # PD column: 1) exact match on target_variable
            #            2) fuzzy match on target_variable words + common patterns
            #            3) last resort: second column
            _pdcol = None
            if target_var in _hdf.columns:
                _pdcol = target_var
            else:
                _kws = [w.lower() for w in target_var.split() if len(w) > 2]
                _kws += ["default", "pd", "taux", "rate", "prob", "défaut"]
                _pdcol = next(
                    (c for c in _hdf.columns if c != _yr and any(k in c.lower() for k in _kws)),
                    _hdf.columns[1] if len(_hdf.columns) > 1 else None,
                )
            if _pdcol:
                _hdf = _hdf[[_yr, _pdcol]].copy()
                # Handle French decimal format (comma as decimal separator, e.g. "0,023")
                for _col in (_yr, _pdcol):
                    _hdf[_col] = pd.to_numeric(
                        _hdf[_col].astype(str).str.replace(",", ".", regex=False),
                        errors="coerce",
                    )
                _hdf = _hdf.dropna()
                hist_x = [int(v)   for v in _hdf[_yr].tolist()]
                hist_y = [float(v) for v in _hdf[_pdcol].tolist()]
                LOG.info("Historical PD: %d points from column '%s'", len(hist_x), _pdcol)
            else:
                LOG.warning("Historical PD: no suitable column found (cols: %s)",
                            list(_hdf.columns))
        except Exception as _he:
            LOG.warning("Historical PD read failed: %s", _he)
        all_pd["historical"] = {"x": hist_x, "y": hist_y}

        # ── Post-Estimation Validation ────────────────────────────────────
        _clim_vreport_dict: dict = {}
        try:
            import numpy as _np
            from app.modules.core.post_estimation_validator import (
                PostEstimationValidator, RiskOutput,
            )
            # Fallback: si insample_fit direct est vide, essayer insample_fits[0]
            _isf = insample_fit or (raw.get("insample_fits") or [{}])[0]
            _yt   = _np.asarray(_isf.get("actual", []), dtype=float)
            _yp   = _np.asarray(_isf.get("fitted", []), dtype=float)
            _fam  = _isf.get("family", "")
            _vars = _isf.get("vars", [])
            _scen = {
                k: _np.asarray(all_pd[k]["y"], dtype=float)
                for k in ("baseline", "adverse", "severe")
                if all_pd.get(k, {}).get("y")
            }
            if len(_yt) > 0 and len(_yp) > 0 and _fam:
                _ro = RiskOutput(
                    module           = "climate",
                    model_type       = _fam,
                    y_true           = _yt,
                    y_pred           = _yp,
                    residuals        = _yt - _yp,
                    n_obs            = len(_yt),
                    k_params         = len(_vars),
                    scenario_results = _scen if len(_scen) == 3 else None,
                )
                _vr = PostEstimationValidator(_ro).run()
                if _vr.verdict == "REJECTED" and _vr.n_fail <= 1:
                    # Cohérent avec le fallback appliqué au tri des 4 candidats
                    # (multi_scenario.py) : UN SEUL test en échec (n_fail==1)
                    # sur 7-9 ne doit pas afficher un verdict "REJETÉ" bloquant
                    # pour le modèle déjà sélectionné par l'utilisateur. Au-delà
                    # (n_fail>=2), le verdict REJETÉ est conservé tel quel — il
                    # ne doit jamais être masqué à l'utilisateur.
                    LOG.warning(
                        "PostEstimation [climate/%s]: verdict REJECTED "
                        "(n_fail=%d) dégradé en VALIDATED_WITH_WARNINGS.",
                        _fam, _vr.n_fail,
                    )
                    _vr.verdict = "VALIDATED_WITH_WARNINGS"
                elif _vr.verdict == "REJECTED":
                    LOG.warning(
                        "PostEstimation [climate/%s]: verdict REJECTED "
                        "(n_fail=%d) conservé — trop de tests en échec pour "
                        "être dégradé.", _fam, _vr.n_fail,
                    )
                _clim_vreport_dict = _vr.to_dict()
                LOG.info(
                    "PostEstimation [climate/%s]: %d P %d W %d F — %s",
                    _fam, _vr.n_pass, _vr.n_warn, _vr.n_fail, _vr.verdict,
                )
                for _tr in _vr.results:
                    if _tr.status in ("WARN", "FAIL"):
                        LOG.warning(
                            "  [%s] %s %s: %s",
                            _tr.status, _tr.test_id, _tr.test_name, _tr.message,
                        )
                # Mettre à jour l'entrée du modèle sélectionné avec le verdict 9/9
                if 0 <= selected_rank_0idx < len(model_table):
                    _sel = model_table[selected_rank_0idx]
                    _sel["verdict"]       = _vr.verdict
                    _sel["n_fail"]        = _vr.n_fail
                    _sel["n_warn"]        = _vr.n_warn
                    _sel["n_pass"]        = _vr.n_pass
                    _sel["verdict_scope"] = "9/9 (inclut B3.3 monotonie des scénarios — Phase 1)"
        except Exception as _ve:
            LOG.warning("PostEstimationValidator (climate) failed: %s", _ve)
        # ─────────────────────────────────────────────────────────────────

        # ── NGFS → LCR/NSFR (si fichier liquidité disponible) ────────────────
        _ngfs_liq: dict = {}
        if not lightweight:
            try:
                _liq_path = (
                    self.upload_paths.get("liquidity") or
                    self.upload_paths.get("liquidity_input") or
                    self.upload_paths.get("liq_input") or
                    ""
                )
                # country : params > session Flask > défaut ISO2
                _country = (
                    p.get("country") or
                    p.get("country_iso2") or
                    p.get("country_code") or
                    "EG"
                )
                # Convertir nom complet → ISO2 si besoin
                _COUNTRY_TO_ISO2 = {
                    "egypt": "EG", "morocco": "MA", "tunisia": "TN",
                    "algeria": "DZ", "nigeria": "NG", "south africa": "ZA",
                    "senegal": "SN", "kenya": "KE", "ghana": "GH",
                    "france": "FR", "germany": "DE", "united kingdom": "GB",
                }
                if len(_country) > 3:
                    _country = _COUNTRY_TO_ISO2.get(
                        _country.lower(), _country[:2].upper())

                _map_yaml = self.upload_paths.get("mapping", "")
                if _liq_path and Path(_liq_path).exists() \
                        and auto_adverse and auto_severe:
                    LOG.info(
                        "NGFS LCR/NSFR: liq_path=%s country=%s",
                        _liq_path, _country,
                    )
                    _active_ngfs_path = (
                        self.upload_paths.get("ngfs", "")
                        if ngfs_mode == "LT"
                        else self.upload_paths.get("ngfs_ct", "")
                    )
                    _baseline_scn = (
                        self.scenario.get("levels", {})
                                     .get("baseline", {})
                                     .get("ngfs_scenario", "Baseline")
                    )
                    from .ngfs_liquidity_engine import compute_ngfs_lcr_nsfr
                    _ngfs_liq = compute_ngfs_lcr_nsfr(
                        liq_excel_path = _liq_path,
                        country_iso2   = _country,
                        ngfs_path      = _active_ngfs_path,
                        ngfs_mode      = ngfs_mode,
                        baseline_scn   = _baseline_scn,
                        adverse_scn    = auto_adverse,
                        severe_scn     = auto_severe,
                        mapping_yaml   = _map_yaml,
                        cache_dir      = "data_cache",
                        cache_ttl      = 30,
                        forced_sat_ranks = p.get("forced_liq_sat_ranks") or {},
                    ) or {}
                    LOG.info(
                        "NGFS LCR/NSFR: %d scénarios calculés",
                        len(_ngfs_liq.get("scenarios", {})),
                    )
                else:
                    _why = (
                        "fichier liquidité introuvable" if not _liq_path
                        else "fichier liquidité introuvable sur le disque" if not Path(_liq_path).exists()
                        else "scénarios adverse/severe non résolus"
                    )
                    LOG.info(
                        "NGFS LCR/NSFR: fichier liquidité absent ou introuvable "
                        "(liq_path=%r, upload_keys=%s)",
                        _liq_path, list(self.upload_paths.keys()),
                    )
                    _ngfs_liq = {"error": _why.capitalize() + "."}
            except Exception as _le:
                LOG.warning(
                    "NGFS LCR/NSFR computation failed: %s", _le, exc_info=True)
                _ngfs_liq = {"error": f"Échec inattendu module liquidité : {_le}"}
        # ─────────────────────────────────────────────────────────────────

        # ── Calibration seule liquidité + marché (Phase 1 / Étape 3) ─────────
        # Additif : expose les tournois satellites liquidité (5 composants
        # bilan) et marché (β₁/β₂/β₃) pendant la Phase 1 légère, sans lancer
        # la projection multi-scénarios coûteuse (compute_stress / run_stressed)
        # — nécessaire pour que l'onglet Liquidité/Marché de l'Étape 3 affiche
        # un vrai tournoi interactif avant le lancement du run complet.
        _liq_calib_only: dict = {}
        _mkt_satellites: dict = {}
        if lightweight:
            try:
                _liq_path_calib = (
                    self.upload_paths.get("liquidity") or
                    self.upload_paths.get("liquidity_input") or
                    self.upload_paths.get("liq_input") or
                    ""
                )
                if _liq_path_calib and Path(_liq_path_calib).exists() \
                        and auto_adverse and auto_severe:
                    from .ngfs_liquidity_engine import (
                        calibrate_ngfs_liquidity_satellites_only,
                    )
                    _baseline_scn_calib = (
                        self.scenario.get("levels", {})
                                     .get("baseline", {})
                                     .get("ngfs_scenario", "Baseline")
                    )
                    _active_ngfs_path_calib = (
                        self.upload_paths.get("ngfs", "")
                        if ngfs_mode == "LT"
                        else self.upload_paths.get("ngfs_ct", "")
                    )
                    # Convertir nom complet → ISO2 si besoin (même mapping que
                    # le bloc liquidité complet plus haut) — fetch_credit_macro
                    # exige un code ISO2/ISO3, pas un nom de pays type "Egypt".
                    _country_calib = (
                        p.get("country") or p.get("country_iso2")
                        or p.get("country_code") or "EG"
                    )
                    _COUNTRY_TO_ISO2_CALIB = {
                        "egypt": "EG", "morocco": "MA", "tunisia": "TN",
                        "algeria": "DZ", "nigeria": "NG", "south africa": "ZA",
                        "senegal": "SN", "kenya": "KE", "ghana": "GH",
                        "france": "FR", "germany": "DE", "united kingdom": "GB",
                    }
                    if len(_country_calib) > 3:
                        _country_calib = _COUNTRY_TO_ISO2_CALIB.get(
                            _country_calib.lower(), _country_calib[:2].upper())
                    _liq_calib_only = calibrate_ngfs_liquidity_satellites_only(
                        liq_excel_path = _liq_path_calib,
                        country_iso2   = _country_calib,
                        ngfs_path      = _active_ngfs_path_calib,
                        ngfs_mode      = ngfs_mode,
                        baseline_scn   = _baseline_scn_calib,
                        adverse_scn    = auto_adverse,
                        severe_scn     = auto_severe,
                        mapping_yaml   = self.upload_paths.get("mapping", ""),
                        forced_sat_ranks = p.get("forced_liq_sat_ranks") or {},
                    ) or {}
                    LOG.info(
                        "Phase1 liquidité: %d satellite(s) calibré(s)",
                        len(_liq_calib_only.get("satellites", {})),
                    )
            except Exception as _lce:
                LOG.warning(
                    "Phase1 liquidité: calibration seule échouée (non-bloquant): %s",
                    _lce, exc_info=True,
                )
                _liq_calib_only = {"error": f"Échec inattendu module liquidité : {_lce}"}

            try:
                from .engine_adapters import MarketEngineAdapter
                # skip_fhs=True: this probe only reads _pf["satellites"] (the
                # fitted IR satellite candidates, for the Step 3 Marché tab
                # preview) below — it never calls run_stressed(), so it never
                # touches risk_measure/fhs. Without skip_fhs, this silently
                # paid for a full FHS Monte Carlo bootstrap (GARCH fit × 3
                # series + 10k-simulation repricing) on every single Phase 1
                # run — the actual cause of Phase 1 runs taking 45+ minutes
                # with the log going silent right after yield-curve alignment,
                # long before the credit tournament (Phase 1's real output)
                # even started.
                _mkt_adapter = MarketEngineAdapter(self.upload_paths, p, {}, skip_fhs=True)
                _pf = getattr(_mkt_adapter, "_prefitted", None)
                if _pf is not None:
                    _sats = _pf.get("satellites") or {}
                    _mkt_satellites = {
                        str(k): {
                            "candidats": (bs.leaderboard or {}).get("candidats", []),
                            "modele_final": (bs.leaderboard or {}).get("modele_final", {}),
                        }
                        for k, bs in _sats.items()
                    }
                    LOG.info(
                        "Phase1 marché: %d satellite(s) β calibré(s)",
                        len(_mkt_satellites),
                    )
            except Exception as _mce:
                LOG.warning(
                    "Phase1 marché: calibration seule échouée (non-bloquant): %s",
                    _mce, exc_info=True,
                )
        # ─────────────────────────────────────────────────────────────────

        # ── NGFS → PD/LGD crédit (si fichier crédit uploadé) ─────────────────
        _ngfs_credit: dict = {}
        if not lightweight:
            try:
                _credit_path = (
                    self.upload_paths.get("credit") or
                    self.upload_paths.get("credit_input") or
                    ""
                )
                _credit_country = (
                    p.get("country") or
                    p.get("country_iso2") or
                    p.get("country_code") or
                    "EG"
                )
                _COUNTRY_TO_ISO2_CR = {
                    "egypt": "EG", "morocco": "MA", "tunisia": "TN",
                    "algeria": "DZ", "nigeria": "NG", "south africa": "ZA",
                    "senegal": "SN", "kenya": "KE", "ghana": "GH",
                    "france": "FR", "germany": "DE", "united kingdom": "GB",
                }
                if len(_credit_country) > 3:
                    _credit_country = _COUNTRY_TO_ISO2_CR.get(
                        _credit_country.lower(), _credit_country[:2].upper())

                if auto_adverse and auto_severe:
                    LOG.info(
                        "NGFS PD/LGD: country=%s ngfs_mode=%s",
                        _credit_country, ngfs_mode,
                    )
                    _active_ngfs_path = (
                        self.upload_paths.get("ngfs", "")
                        if ngfs_mode == "LT"
                        else self.upload_paths.get("ngfs_ct", "")
                    )
                    _baseline_scn_cr = (
                        self.scenario.get("levels", {})
                                     .get("baseline", {})
                                     .get("ngfs_scenario", "Baseline")
                    )
                    _lgd_ttc = float(p.get("lgd_ttc", 0.45))
                    _hist_start = int(p.get("history_start", 1990))
                    from .ngfs_credit_engine import compute_ngfs_pd_lgd
                    _ngfs_credit = compute_ngfs_pd_lgd(
                        upload_paths  = self.upload_paths,
                        country_iso2  = _credit_country,
                        ngfs_path     = _active_ngfs_path,
                        ngfs_mode     = ngfs_mode,
                        baseline_scn  = _baseline_scn_cr,
                        adverse_scn   = auto_adverse,
                        severe_scn    = auto_severe,
                        lgd_ttc       = _lgd_ttc,
                        history_start = _hist_start,
                        mapping_yaml  = _map_yaml,
                        cache_dir     = "data_cache",
                        cache_ttl     = 30,
                    ) or {}
                    LOG.info(
                        "NGFS PD/LGD: %d scénarios calculés",
                        len(_ngfs_credit.get("scenarios", {})),
                    )
                else:
                    LOG.info(
                        "NGFS PD/LGD: scénarios adverse/severe non encore "
                        "sélectionnés (auto_adverse=%r auto_severe=%r)",
                        auto_adverse, auto_severe,
                    )
            except Exception as _ce:
                LOG.warning(
                    "NGFS PD/LGD computation failed: %s", _ce, exc_info=True)
        # ─────────────────────────────────────────────────────────────────

        # ── Réconcilie all_pd (Row 3 "Crédit" chart + KPIs baseline/adverse/
        # severe de CE wrapper) avec les trajectoires NGFS déjà corrigées
        # (compute_ngfs_pd_lgd) ────────────────────────────────────────────
        # all_pd a été construit plus haut depuis raw["projections"] — un
        # moteur multi-scénarios générique (pas de reconstruction niveau
        # absolu NGFS ni d'ancrage première-année), qui produit encore des
        # trajectoires baseline/adverse/severe quasi identiques. _ngfs_credit
        # (ci-dessus) contient déjà les trajectoires corrigées utilisées avec
        # succès par climate_stress_results (Row 1) et le tableau KPI sous ce
        # même graph — on les réutilise ici pour que le graph Row 3 Bloc A et
        # les KPIs avg_pd/peak_pd de ce wrapper cessent de diverger de la
        # source de vérité déjà validée.
        _ngfs_scen_for_chart = _ngfs_credit.get("scenarios", {}) if _ngfs_credit else {}
        if _ngfs_scen_for_chart:
            for lvl in ("baseline", "adverse", "severe"):
                _s = _ngfs_scen_for_chart.get(lvl)
                if _s and _s.get("years") and _s.get("pd"):
                    all_pd[lvl] = {"x": list(_s["years"]), "y": list(_s["pd"])}
            LOG.info(
                "ClimateWrapper: all_pd (Row 3 + KPIs) réconcilié avec les "
                "trajectoires NGFS corrigées (%d scénarios)",
                len(_ngfs_scen_for_chart),
            )

        # ── Climate Orchestrator — transmission vers les autres modules ──────────
        # Called after all NGFS computations complete so ngfs_credit/ngfs_liq
        # are available.  Non-blocking: any failure stores an error key and the
        # existing dashboard sections are unaffected.
        _climate_stress_results: dict = {}
        if not lightweight:
            try:
                from .climate_orchestrator_runner import run_climate_orchestration

                _orch_config = {
                    "upload_paths": self.upload_paths,
                    "params":       p,
                    "scenario":     self.scenario,
                }

                _macro_df_hist = _ngfs_credit.get("macro_df_hist")
                if _macro_df_hist is None:
                    _macro_df_hist = _ngfs_liq.get("macro_df_hist")

                _target_series: dict = {}

                _pd_hist = _ngfs_credit.get("pd_series_hist")
                if _pd_hist is not None and not _pd_hist.empty:
                    _target_series["pd"] = _pd_hist
                    LOG.info(
                        "ClimateWrapper: PD historique exposé au satellite — "
                        "%d pts (%d–%d)",
                        len(_pd_hist), int(_pd_hist.index.min()),
                        int(_pd_hist.index.max()),
                    )
                elif hist_x:
                    import pandas as _pd_mod
                    _pd_hist_fb = _pd_mod.Series(
                        hist_y, index=hist_x, dtype=float, name="pd"
                    )
                    _pd_hist_fb.index = _pd_hist_fb.index.astype(int)
                    _target_series["pd"] = _pd_hist_fb
                    LOG.info(
                        "ClimateWrapper: PD historique fallback depuis hist_x/y"
                        " — %d pts", len(hist_x),
                    )

                _lcr_hist = _ngfs_liq.get("lcr_hist")
                if _lcr_hist is not None and not _lcr_hist.empty:
                    _target_series["lcr"] = _lcr_hist
                    LOG.info(
                        "ClimateWrapper: LCR historique observé exposé — %d pts",
                        len(_lcr_hist),
                    )
                _nsfr_hist = _ngfs_liq.get("nsfr_hist")
                if _nsfr_hist is not None and not _nsfr_hist.empty:
                    _target_series["nsfr"] = _nsfr_hist
                    LOG.info(
                        "ClimateWrapper: NSFR historique observé exposé — %d pts",
                        len(_nsfr_hist),
                    )

                _beta_series = _extract_market_betas(self.upload_paths, p)
                if _beta_series:
                    _target_series.update(_beta_series)
                    LOG.info(
                        "ClimateWrapper: betas NS historiques exposés — "
                        "cibles=%s", list(_beta_series.keys()),
                    )

                _orch_ngfs = {
                    "ngfs_credit":       _ngfs_credit,
                    "ngfs_liquidity":    _ngfs_liq,
                    "baseline_scenario": (
                        self.scenario.get("levels", {})
                        .get("baseline", {})
                        .get("ngfs_scenario", "Baseline")
                    ),
                    "adverse_scenario":  auto_adverse,
                    "severe_scenario":   auto_severe,
                    "ngfs_mode":         ngfs_mode,
                    "macro_trajectories": (
                        _ngfs_credit.get("macro_trajectories")
                        or _ngfs_liq.get("macro_trajectories")
                    ),
                    "macro_df":      _macro_df_hist,
                    "target_series": _target_series,
                }
                _climate_stress_results = run_climate_orchestration(
                    config           = _orch_config,
                    ngfs_projections = _orch_ngfs,
                    institution_data = {},
                )
                LOG.info(
                    "ClimateOrchestrator: %d scenarios computed",
                    len((_climate_stress_results.get("results") or {}).keys()),
                )

                # ── Séries de projection sVaR/sES (Row 5 dashboard) ───────────
                # Additif : agrège les rows déjà sérialisées par
                # ClimateStressResult.to_dashboard_dict() sur tout l'horizon —
                # aucun nouvel appel moteur, aucune donnée historique fabriquée
                # (historical_available reste False, voir macro_delta.py).
                try:
                    from .macro_delta import get_market_projection_series
                    _csr_results = _climate_stress_results.get("results") or {}
                    _svar_series = get_market_projection_series(
                        _csr_results, auto_adverse, auto_severe,
                        metric="sVaR 99% (M)",
                    )
                    _ses_series = get_market_projection_series(
                        _csr_results, auto_adverse, auto_severe,
                        metric="sES 97.5% (M)",
                    )
                    _climate_stress_results["market_svar_series"] = (
                        _svar_series.to_dict() if _svar_series else {}
                    )
                    _climate_stress_results["market_ses_series"] = (
                        _ses_series.to_dict() if _ses_series else {}
                    )
                except Exception as _mse:
                    LOG.warning(
                        "ClimateOrchestrator: séries sVaR/sES échouées "
                        "(non-bloquant): %s", _mse)
            except Exception as _oe:
                LOG.warning(
                    "ClimateOrchestrator failed (non-blocking): %s", _oe)
                _climate_stress_results = {"error": str(_oe), "results": {}}
        # ─────────────────────────────────────────────────────────────────────

        for lvl in ("baseline", "adverse", "severe"):
            x, y = all_pd[lvl]["x"], all_pd[lvl]["y"]
            ts   = pd.DataFrame({"year": x, "pd": y, "loss": y})
            avg  = float(sum(y) / len(y)) if y else 0.0

            kpis = {
                "avg_pd":        round(avg, 4),
                "peak_pd":       round(max(y, default=0), 4),
                "model_family":  sm.get("family") or sm.get("modele", "N/A"),
                "model_r2":      round(float(sm.get("R2") or sm.get("r2_or_pseudo") or sm.get("r2") or 0), 3),
                "model_vars":    (lambda v: [x.strip() for x in v.split(",") if x.strip()]
                                  if isinstance(v, str) else list(v or [])
                                 )(sm.get("variables", "")),
                "ngfs_adverse":  auto_adverse,
                "ngfs_severe":   auto_severe,
                "ngfs_mode":     ngfs_mode,
            }

            results[lvl] = PlatformResult(
                module_id    = self.module_id,
                module_label = self.module_label,
                scenario_id  = lvl,
                total_loss   = avg,
                kpis         = kpis,
                time_series  = ts,
                charts_data  = {
                    "pd_trajectories":       all_pd,
                    "selected_model":        sm,
                    "model_table":           model_table,
                    "scenario_combos":       scenario_combos,
                    "auto_adverse":          auto_adverse,
                    "auto_severe":           auto_severe,
                    "auto_score":            auto_score,
                    "model_used":            model_used,
                    "insample_fit":          insample_fit,
                    "selected_rank_0idx":    selected_rank_0idx,
                    "pretest_report":        pretest_report,
                    "validation_report":     _clim_vreport_dict,
                    "ngfs_liquidity":        _ngfs_liq,
                    "ngfs_credit":           _ngfs_credit,
                    # Tournois de calibration seule (Phase 1 / Étape 3) —
                    # liquidité (5 satellites composants bilan) et marché
                    # (β₁/β₂/β₃) — vides hors mode lightweight.
                    "liq_calib_tournaments": _liq_calib_only.get(
                        "tournament_leaderboards", {}),
                    "liq_calib_error":       _liq_calib_only.get("error", ""),
                    "mkt_beta_satellites":   _mkt_satellites,
                    # Stored only in baseline to avoid tripling the payload
                    "climate_stress_results": (
                        _climate_stress_results if lvl == "baseline" else {}
                    ),
                },
                metadata = {
                    "scenario_selection": ss,
                    "ngfs_mode":          ngfs_mode,
                    "model_table":        model_table,
                    "scenario_combos":    scenario_combos,
                    "model_used":         model_used,
                },
            )
        return results

    # ── Pre-flight: validate historical CSV has a PD column ──────────
    _PD_COL_PATTERNS = [
        "default", "pd", "taux", "rate", "prob", "défaut", "defaut",
        "npl", "loss", "default rate",
    ]

    def _check_historical_file(self, path: str, target_var: str):
        """Return None if OK, or an error string if the file is missing/invalid."""
        if not path or not Path(path).exists():
            return (
                f"Fichier historique introuvable : '{path}'. "
                "Veuillez uploader votre fichier historique (CSV avec colonnes "
                f"'year' et '{target_var}')."
            )
        try:
            _head = Path(path).read_text(encoding="latin-1")[:500]
            _sep  = ";" if _head.count(";") > _head.count(",") else ","
            _hdf  = pd.read_csv(path, sep=_sep, nrows=3, encoding="latin-1")
            _hdf.columns = [str(c).strip() for c in _hdf.columns]
            cols_lower = {c.lower(): c for c in _hdf.columns}
            # 1) exact match
            if target_var in _hdf.columns or target_var.lower() in cols_lower:
                return None
            # 2) pattern match
            found = any(
                any(pat in c.lower() for pat in self._PD_COL_PATTERNS)
                for c in _hdf.columns
            )
            if found:
                return None
            # 3) any numeric non-year column
            yr_col = next((c for c in _hdf.columns if "year" in c.lower()), None)
            for c in _hdf.columns:
                if c == yr_col:
                    continue
                if pd.to_numeric(_hdf[c], errors="coerce").notna().any():
                    return None
            return (
                f"Aucune colonne PD reconnue dans '{Path(path).name}'. "
                f"Colonnes trouvées : {list(_hdf.columns)}. "
                f"Renommez votre colonne PD en '{target_var}' (ou 'pd', 'taux_defaut', etc.)."
            )
        except Exception as e:
            return f"Impossible de lire le fichier historique : {e}"

    def _err(self, msg):
        return {lvl: PlatformResult(
            self.module_id, self.module_label, lvl, 0.0,
            {"error": msg},
            pd.DataFrame(columns=["year", "pd", "loss"]),
            {}, {}, [msg]) for lvl in ("baseline", "adverse", "severe")}

    def get_kpis(self, results):
        out = {}
        for lvl, r in results.items():
            for k, v in r.kpis.items():
                out[f"{lvl}_{k}"] = v
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Helper module-level : extraction des betas Nelson-Siegel historiques
# ─────────────────────────────────────────────────────────────────────────────

def _extract_market_betas(
    upload_paths: dict,
    params: dict,
) -> dict:
    """
    Charge les données de courbe des taux et ajuste le modèle Nelson-Siegel pour
    obtenir les séries historiques de β₁, β₂, β₃ (index = année int).

    Retourne {beta1: pd.Series, beta2: pd.Series, beta3: pd.Series} si les
    données sont disponibles, {} sinon (dégradation silencieuse).

    Les betas sont des moyennes annuelles des observations mensuelles/quotidiennes
    du NS fitting pour être cohérents avec le pas annuel du socle macro.
    """
    try:
        from ..market.yield_curve_loader import YieldCurveLoader
        from ..market.nelson_siegel import NelsonSiegelFitter
        from ..market.wrapper import MarketModuleWrapper

        # Résoudre le chemin du fichier Excel marché
        _mkt_wrapper = MarketModuleWrapper(upload_paths, params, scenario={})
        excel_path   = _mkt_wrapper._resolve_excel(params or {})

        loader = YieldCurveLoader(
            excel_path     = excel_path,
            country        = params.get("country", "EG"),
            cache_dir      = params.get("cache_dir", "data_cache"),
            cache_ttl_days = int(params.get("cache_ttl_days", 30)),
        )
        data     = loader.load()
        yield_df = data.yield_df

        if yield_df.empty or yield_df.dropna(how="all").shape[0] < 24:
            LOG.info("_extract_market_betas: données courbe insuffisantes")
            return {}

        ns_lambda, _ = NelsonSiegelFitter.search_lambda(yield_df)
        fitter       = NelsonSiegelFitter(lambda_=ns_lambda, min_maturities=3)
        betas_df     = fitter.fit(yield_df)   # index = date, colonnes = beta1/2/3

        if betas_df.dropna().shape[0] < 12:
            LOG.info("_extract_market_betas: trop peu de betas NS (%d)", len(betas_df))
            return {}

        # Agréger en moyennes annuelles (NS fitting peut être mensuel ou quotidien)
        betas_df.index = pd.to_datetime(betas_df.index, errors="coerce")
        betas_annual   = (
            betas_df.dropna()
                    .groupby(betas_df.dropna().index.year)
                    .mean()
        )
        betas_annual.index = betas_annual.index.astype(int)

        result = {}
        for col in ("beta1", "beta2", "beta3"):
            if col in betas_annual.columns:
                result[col] = betas_annual[col].rename(col)

        LOG.info(
            "_extract_market_betas: %d séries NS extraites (%d années)",
            len(result), len(betas_annual),
        )
        return result

    except Exception as exc:
        LOG.info("_extract_market_betas: non disponible (%s) — betas ignorés", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Helper module-level : extraction des séries LCR/NSFR historiques observées
# ─────────────────────────────────────────────────────────────────────────────

def _extract_liquidity_history(
    upload_paths: dict,
    params: dict,
) -> dict:
    """
    Charge le fichier Excel liquidité et extrait les séries historiques
    observées de LCR/NSFR (colonnes _lcr_obs/_nsfr_obs), sans lancer la
    calibration complète des satellites (coûteuse, inutile ici).

    Retourne {"lcr": pd.Series, "nsfr": pd.Series} pour les colonnes
    disponibles (index = année int), {} sinon (dégradation silencieuse).
    """
    try:
        # Import via le chemin package correct (déclenche l'ajout de
        # modules_src/ à sys.path, comme dans ngfs_liquidity_engine.py).
        from . import ngfs_liquidity_engine as _  # noqa: F401
        from liquidity_module.liquidity_stress_engine import LiquidityDataLoader

        liq_path = (
            upload_paths.get("liquidity")
            or upload_paths.get("liquidity_input")
            or upload_paths.get("liq_input")
            or ""
        )
        if not liq_path or not Path(liq_path).exists():
            LOG.info("_extract_liquidity_history: fichier liquidité absent")
            return {}

        ts = LiquidityDataLoader._read_time_series(liq_path)

        result: dict = {}
        for key, col in (("lcr", "_lcr_obs"), ("nsfr", "_nsfr_obs")):
            if col in ts.columns:
                s = pd.to_numeric(ts[col], errors="coerce").dropna()
                if not s.empty:
                    s.index = s.index.astype(int)
                    result[key] = s.rename(key)

        LOG.info(
            "_extract_liquidity_history: %d série(s) extraite(s): %s",
            len(result), list(result.keys()),
        )
        return result

    except Exception as exc:
        LOG.info("_extract_liquidity_history: non disponible (%s) — ignoré", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers module-level : calibration (pas juste extraction d'historique) des
# moteurs liquidité/marché, pour la sélection multi-risque du couple de
# scénarios (adverse, severe) — voir multi_risk_matrix.py.
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate_liquidity_engine(upload_paths: dict, params: dict):
    """
    Charge + calibre un LiquidityStressEngine complet (satellites LCR/NSFR
    sur historique), pour que la sélection de scénario puisse rejouer ces
    satellites déjà calibrés sur tous les scénarios NGFS candidats (voir
    multi_risk_matrix.build_liquidity_matrix). Dégradation silencieuse
    (retourne None) si le fichier liquidité est absent ou la calibration
    échoue — n'a jamais bloqué le run climat.
    """
    try:
        from .ngfs_liquidity_engine import _prepare_macro_df
        from liquidity_module.liquidity_stress_engine import (
            LiquidityDataLoader, LiquidityStressEngine,
        )
        from app.modules.core.imf_weo_fetcher import fetch_credit_macro

        liq_path = (
            upload_paths.get("liquidity")
            or upload_paths.get("liquidity_input")
            or upload_paths.get("liq_input")
            or ""
        )
        if not liq_path or not Path(liq_path).exists():
            return None

        # Convertir nom complet → ISO2 si besoin — même mapping que les
        # autres points d'entrée liquidité (fetch_credit_macro exige un code
        # ISO2/ISO3, pas un nom de pays type "Egypt").
        _country = params.get("country") or params.get("country_iso2") or "EG"
        _COUNTRY_TO_ISO2_SEL = {
            "egypt": "EG", "morocco": "MA", "tunisia": "TN",
            "algeria": "DZ", "nigeria": "NG", "south africa": "ZA",
            "senegal": "SN", "kenya": "KE", "ghana": "GH",
            "france": "FR", "germany": "DE", "united kingdom": "GB",
        }
        if len(_country) > 3:
            _country = _COUNTRY_TO_ISO2_SEL.get(_country.lower(), _country[:2].upper())

        macro_raw, _ = fetch_credit_macro(
            country        = _country,
            start_year     = int(params.get("history_start", 1990)),
            cache_dir      = params.get("cache_dir", "data_cache"),
            cache_ttl_days = int(params.get("cache_ttl_days", 30)),
        )
        if macro_raw is None or macro_raw.empty:
            return None

        # Renommer WEO → noms attendus par LiquidityDataLoader._validate_macro()
        # (GDP_growth, exchange_rate, ...) — sans cette étape, les colonnes
        # brutes (real_gdp_growth, exchange_rate_lcu_usd, ...) ne correspondent
        # jamais et load() lève systématiquement "colonnes macro manquantes".
        macro_df = _prepare_macro_df(macro_raw)

        inputs = LiquidityDataLoader.load(liq_path, macro_df)
        engine = LiquidityStressEngine(
            inputs, portfolio_type=params.get("portfolio_type", "mixed"),
        )
        engine.calibrate()
        LOG.info("_calibrate_liquidity_engine: calibration OK.")
        return engine

    except Exception as exc:
        LOG.info(
            "_calibrate_liquidity_engine: non disponible (%s) — ignoré", exc,
        )
        return None


def _calibrate_market_beta1(upload_paths: dict, params: dict):
    """
    Calibre la courbe des taux + IRSatellite (beta1/2/3) en réutilisant
    MarketEngineAdapter._load_and_fit() (climate_orchestrator) — évite de
    dupliquer la logique Nelson-Siegel + IRSatellite déjà écrite et validée.

    Retourne (beta1_satellite, last_beta1) — (None, None) si les données de
    courbe des taux sont indisponibles ou la calibration échoue (dégradation
    silencieuse, n'a jamais bloqué le run climat).
    """
    try:
        from .engine_adapters import MarketEngineAdapter

        # skip_fhs=True: this adapter is discarded after extracting
        # satellites/last_betas below — it's never kept for run_stressed(),
        # so the FHS Monte Carlo bootstrap it would otherwise run (GARCH ×3
        # + 10k-sim repricing) was pure wasted cost, paid on EVERY climate
        # run (this runs unconditionally, before the tournament, not just in
        # Phase 1 — see call site).
        adapter = MarketEngineAdapter(upload_paths, params, base_scenario={}, skip_fhs=True)
        pf = adapter._prefitted
        if not pf:
            return None, None

        satellites = pf.get("satellites", {})
        last_betas = pf.get("last_betas")
        beta1_sat  = satellites.get(1)
        if beta1_sat is None or last_betas is None or "beta1" not in last_betas:
            return None, None

        LOG.info("_calibrate_market_beta1: calibration OK.")
        return beta1_sat, float(last_betas["beta1"])

    except Exception as exc:
        LOG.info(
            "_calibrate_market_beta1: non disponible (%s) — ignoré", exc,
        )
        return None, None
