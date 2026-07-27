"""Dashboard blueprint — includes Phase1/Phase2 wizard endpoints."""
import threading, time, logging, sys
from pathlib import Path
from flask import Blueprint, render_template, request, jsonify, current_app, session

from ..services.run_store      import RunStore
from ..services.orchestrator   import RunOrchestrator
from ..services.scenario_manager import ScenarioManager, HISTORICAL_SCENARIOS, NGFS_SCENARIOS
from ..services.chart_builder  import ChartBuilder
from ..services.job_store      import JobStore
from ..modules.registry        import list_modules

dashboard_bp = Blueprint("dashboard", __name__)
LOG = logging.getLogger("dashboard")

# ── Country → currency mapping ────────────────────────────────────────
# (code, symbol, millions_label)
COUNTRY_CURRENCY = {
    # Full country names
    "Egypt":          ("EGP", "EGP", "M EGP"),
    "France":         ("EUR", "€",   "M€"),
    "Germany":        ("EUR", "€",   "M€"),
    "United Kingdom": ("GBP", "£",   "M£"),
    "Morocco":        ("MAD", "MAD", "M MAD"),
    "Tunisia":        ("TND", "TND", "M TND"),
    "Algeria":        ("DZD", "DZD", "M DZD"),
    "Nigeria":        ("NGN", "₦",   "M NGN"),
    "South Africa":   ("ZAR", "R",   "M ZAR"),
    "Senegal":        ("XOF", "XOF", "M XOF"),
    "Kenya":          ("KES", "KES", "M KES"),
    "Ghana":          ("GHS", "GHS", "M GHS"),
    "MENA":           ("USD", "$",   "M$"),
    "GCC":            ("SAR", "SAR", "M SAR"),
    "North Africa":   ("MAD", "MAD", "M MAD"),
    "Europe":         ("EUR", "€",   "M€"),
    # ISO2 codes (what wrapper.py stores in metadata under "country")
    "EG": ("EGP", "EGP", "M EGP"),
    "MA": ("MAD", "MAD", "M MAD"),
    "TN": ("TND", "TND", "M TND"),
    "DZ": ("DZD", "DZD", "M DZD"),
    "NG": ("NGN", "₦",   "M NGN"),
    "ZA": ("ZAR", "R",   "M ZAR"),
    "SN": ("XOF", "XOF", "M XOF"),
    "KE": ("KES", "KES", "M KES"),
    "GH": ("GHS", "GHS", "M GHS"),
    "FR": ("EUR", "€",   "M€"),
    "DE": ("EUR", "€",   "M€"),
    "GB": ("GBP", "£",   "M£"),
    "IT": ("EUR", "€",   "M€"),
    "ES": ("EUR", "€",   "M€"),
    "SA": ("SAR", "SAR", "M SAR"),
    "AE": ("AED", "AED", "M AED"),
    "QA": ("QAR", "QAR", "M QAR"),
    "KW": ("KWD", "KWD", "M KWD"),
    "BH": ("BHD", "BHD", "M BHD"),
    "OM": ("OMR", "OMR", "M OMR"),
    "JO": ("JOD", "JOD", "M JOD"),
    "TR": ("TRY", "TRY", "M TRY"),
    "US": ("USD", "$",   "M USD"),
    "BR": ("BRL", "R$",  "M BRL"),
    "IN": ("INR", "₹",   "M INR"),
    "CN": ("CNY", "¥",   "M CNY"),
    "JP": ("JPY", "¥",   "M JPY"),
}
_DEFAULT_CURRENCY = ("USD", "$", "M$")


def _currency_for_record(record: dict) -> tuple:
    """Extract (code, symbol, millions_label) from a stored run record."""
    if not record:
        return _DEFAULT_CURRENCY
    # Try credit metadata first
    country = (
        record.get("module_results", {})
              .get("credit", {})
              .get("baseline", {})
              .get("metadata", {})
              .get("country", "")
    )
    # Fallback: climate metadata
    if not country:
        country = (
            record.get("module_results", {})
                  .get("climate", {})
                  .get("baseline", {})
                  .get("metadata", {})
                  .get("country", "")
        )
    return COUNTRY_CURRENCY.get(country, _DEFAULT_CURRENCY)

# ── NGFS scenario lists per mode ──────────────────────────────────────
NGFS_LT_SCENARIOS = [
    "Net Zero 2050", "Below 2°C", "Delayed Transition",
    "Divergent Net Zero", "Nationally Determined Contributions (NDCs)",
    "Current Policies", "Fragmented World",
]
NGFS_CT_SCENARIOS = [
    "Net Zero 2050", "Below 2°C", "Delayed Transition",
    "Divergent Net Zero", "Nationally Determined Contributions (NDCs)",
    "Current Policies",
]

# ── Phase 1 progress messages (timed simulation) ─────────────────────
PHASE1_STEPS = [
    (0,   5,  "⏳ Chargement du fichier NGFS..."),
    (5,   8,  "✅ Fichier NGFS chargé"),
    (8,   12, "⏳ Chargement des données historiques..."),
    (12,  15, "✅ Données historiques chargées"),
    (15,  20, "⏳ Sélection des variables candidates (Stage 1)..."),
    (20,  30, "✅ Variables candidates identifiées"),
    (30,  35, "⏳ Calcul des corrélations..."),
    (35,  45, "✅ Corrélations calculées — filtrage VIF..."),
    (45,  50, "⏳ Estimation des modèles satellites (Stage 2)..."),
    (50,  65, "✅ Estimation Beta, Logit, Vasicek en cours..."),
    (65,  75, "✅ Modèles estimés — calcul du scoring..."),
    (75,  85, "✅ Scoring AIC/BIC/R² calculé"),
    (85,  92, "⏳ Sélection de la meilleure combinaison de scénarios..."),
    (92,  97, "✅ Combinaison optimale identifiée"),
    (97, 100, "✅ Estimation terminée — tableau de scoring prêt !"),
]

PHASE2_STEPS = [
    (0,  10, "⏳ Initialisation du modèle satellite sélectionné..."),
    (10, 25, "✅ Modèle chargé — calcul trajectoires PD Baseline..."),
    (25, 40, "✅ PD Baseline calculé — calcul trajectoires PD Adverse..."),
    (40, 55, "✅ PD Adverse calculé — calcul trajectoires PD Sévère..."),
    (55, 70, "✅ PD Sévère calculé — calcul EL / LGD / Frye-Jacobs..."),
    (70, 82, "✅ EL calculé — calcul des ratios réglementaires..."),
    (82, 92, "✅ Ratios CET1/LCR/NSFR calculés"),
    (92, 98, "✅ Consolidation des résultats..."),
    (98, 100,"✅ Analyse complète terminée !"),
]


def _get_jobs_folder():
    """Return jobs folder path."""
    import os
    base = Path(__file__).resolve().parent.parent.parent
    folder = base / "runs" / "jobs"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


def _automap_files(raw: dict) -> dict:
    from .upload import _automap
    return _automap(raw)


def _enrich_upload_paths(upload_paths: dict) -> dict:
    """Fallback: scan uploads/ folder for known files when localStorage/session is empty."""
    try:
        upload_folder = Path(current_app.config.get("UPLOAD_FOLDER", "uploads"))
        if not upload_folder.exists():
            return upload_paths
        # Always expose the absolute upload_dir so market module detector finds the right folder
        upload_paths["upload_dir"] = str(upload_folder)
        from .upload import _automap
        folder_paths = {p.name: str(p) for p in upload_folder.iterdir() if p.is_file()}
        fallback = _automap(folder_paths)
        for role, path in fallback.items():
            upload_paths.setdefault(role, path)
    except Exception:
        pass
    return upload_paths


def _available_modules(record):
    available = []
    for mid, scns in record.get("module_results", {}).items():
        for scn_data in scns.values():
            kpis = scn_data.get("kpis", {})
            if kpis and "error" not in kpis and "status" not in kpis:
                available.append(mid)
                break
    return available


def _progress_writer(job_store, job_id, steps, stop_event):
    """Background thread: writes timed progress steps until stop_event."""
    start = time.time()
    written = set()
    while not stop_event.is_set():
        elapsed = time.time() - start
        for (t_start, t_end, msg) in steps:
            if t_start <= elapsed < t_end and msg not in written:
                pct = int(t_start / steps[-1][1] * 100) if steps[-1][1] > 0 else 0
                job_store.add_step(job_id, msg, pct)
                written.add(msg)
        time.sleep(1.5)


# ════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD ROUTE
# ════════════════════════════════════════════════════════════════════
@dashboard_bp.route("/dashboard")
def dashboard():
    store       = RunStore(current_app.config["RUNS_FOLDER"])
    run_id      = request.args.get("run_id")
    active_m    = request.args.get("module", "credit")
    all_runs    = store.list_runs()
    # Only load record if run_id is explicitly in URL; otherwise wizard is shown
    record = store.load(run_id) if run_id else None
    charts, available_modules = {}, []
    if record:
        available_modules = _available_modules(record)
        if active_m not in available_modules and available_modules:
            active_m = available_modules[0]
        try:
            builder = ChartBuilder(record)
            charts  = builder.build_for_module(active_m)
        except Exception as e:
            LOG.exception("Chart build failed: %s", e)

    # Filter selector to only runs for the active module, then deduplicate
    # by scenario_label (newest per label). Fixes 'static KPI cards'
    # caused by mixing modules and 40+ identical Gulf War entries.
    module_runs = ([r for r in all_runs if active_m in (r.get('modules') or [])]
                  if record else all_runs)
    if not module_runs:
        module_runs = all_runs
    seen_labels: set = set()
    runs: list = []
    for r in module_runs:
        label = r.get('scenario_label') or r.get('run_name') or ''
        if label not in seen_labels:
            seen_labels.add(label)
            runs.append(r)
        elif r.get('run_id') == run_id:
            runs.append(r)

    curr = _currency_for_record(record)
    try:
        from ..services.idio_events import IDIO_EVENTS
        idio_list = [
            {"id": k, "label": v.get("name", k), "icon": v.get("icon", "⚡"),
             "category": v.get("category", ""), "description": v.get("description", "")}
            for k, v in IDIO_EVENTS.items()
        ]
    except Exception:
        idio_list = []
    return render_template("dashboard.html",
        run_id=run_id, record=record, runs=runs,
        charts=charts, active_module=active_m,
        available_modules=available_modules,
        modules_meta=list_modules(),
        currency_code=curr[0],
        currency_symbol=curr[1],
        currency_millions=curr[2],
        hist_list=[{"id":k,"label":v["label"],"icon":v.get("icon","📊"),
                    "description":v.get("description","")}
                   for k,v in HISTORICAL_SCENARIOS.items()],
        ngfs_list=[{"id":k,"label":v["label"],"icon":v.get("icon","🌍"),
                    "description":v.get("description","")}
                   for k,v in NGFS_SCENARIOS.items()],
        idio_list=idio_list,
        saved_list=ScenarioManager(current_app.config["SCENARIOS_FOLDER"]).list_saved(),
        ngfs_lt=NGFS_LT_SCENARIOS,
        ngfs_ct=NGFS_CT_SCENARIOS,
    )


# ════════════════════════════════════════════════════════════════════
# IDIOSYNCRATIC EVENTS API
# ════════════════════════════════════════════════════════════════════
@dashboard_bp.route("/api/idio_event/<event_id>")
def api_idio_event(event_id):
    """Return event metadata + DAG Plotly chart + base values from current run."""
    from ..services.idio_events import IDIO_EVENTS, build_dag_figure
    from ..services.idio_engine import get_base_values, BASE_LABELS

    event = IDIO_EVENTS.get(event_id)
    if not event:
        return jsonify({"error": "Event not found"}), 404

    # Load current run for base values extraction (Layer 1a)
    run_id = request.args.get("run_id") or session.get("last_run_id")
    store  = RunStore(current_app.config["RUNS_FOLDER"])
    record = store.load(run_id) if run_id else None

    # Dataset path (Layer 1b — used when no run available)
    up = _enrich_upload_paths(dict(session.get("upload_paths", {})))
    dataset_path = (up.get("historical") or up.get("capital")
                    or up.get("liquidity") or next(iter(up.values()), None))
    LOG.info("idio_event %s: run_id=%s dataset=%s keys=%s",
             event_id, run_id, dataset_path, list(up.keys()))

    base_wrapped = get_base_values(record, dataset_path=dataset_path)

    required = event.get("required_base", [])
    base_for_event = {
        k: {
            "value":  base_wrapped[k]["value"],
            "source": base_wrapped[k]["source"],
            "label":  BASE_LABELS.get(k, k),
        }
        for k in required if k in base_wrapped
    }

    dag_chart = build_dag_figure(event)

    return jsonify({
        "id":           event_id,
        "name":         event["name"],
        "category":     event["category"],
        "icon":         event["icon"],
        "description":  event["description"],
        "trigger":      event["trigger"],
        "balance_sheet":event["balance_sheet"],
        "shock_inputs": event["shock_inputs"],
        "base_values":  base_for_event,
        "dag_chart":    dag_chart,
    })


@dashboard_bp.route("/api/idio_shock", methods=["POST"])
def api_idio_shock():
    """Compute quantitative ratio impacts for an idiosyncratic shock."""
    from ..services.idio_events import IDIO_EVENTS, build_dag_figure
    from ..services.idio_engine import (
        compute_shock, flat_base, get_base_values,
        get_historical_series_from_dataset,
    )

    data     = request.get_json(force=True)
    event_id = data.get("event_id", "")
    inputs   = data.get("inputs", {})
    user_base= data.get("base", {})   # user-overridden values from the form

    if event_id not in IDIO_EVENTS:
        return jsonify({"error": "Unknown event"}), 400

    # Merge: user-provided values override auto-detected ones
    run_id = data.get("run_id") or session.get("last_run_id")
    store  = RunStore(current_app.config["RUNS_FOLDER"])
    record = store.load(run_id) if run_id else None

    up = _enrich_upload_paths(dict(session.get("upload_paths", {})))
    dataset_path = (up.get("historical") or up.get("capital")
                    or up.get("liquidity") or next(iter(up.values()), None))
    liq_path     = up.get("liquidity") or dataset_path
    LOG.info("idio_shock %s: run_id=%s dataset=%s keys=%s",
             event_id, run_id, dataset_path, list(up.keys()))

    base_wrapped = get_base_values(record, dataset_path=dataset_path)
    base_flat    = flat_base(base_wrapped)
    # Apply user overrides
    for k, v in user_base.items():
        try:
            base_flat[k] = float(v)
        except (TypeError, ValueError):
            pass

    result = compute_shock(event_id, inputs, base_flat)

    # ── Build real historical series ────────────────────────────────────────
    # LCR & NSFR: all historical rows from the uploaded liquidity dataset file
    historical_series: dict = {}
    if liq_path:
        historical_series.update(
            get_historical_series_from_dataset(liq_path, keys=("lcr", "nsfr"))
        )

    # CET1, CAR, Leverage: from credit module run record
    if record:
        cr = record.get("module_results", {}).get("credit", {})
        bm = cr.get("baseline") or cr.get("adverse") or cr.get("severe")
        if bm:
            ct = bm.get("charts_data", {}).get("capital_trajectories", {})
            for key, alias in (
                ("cet1_ratio", "CET1"),
                ("car",        "CAR"),
                ("leverage_ratio", "LEVERAGE"),
                ("tier1_ratio",    "TIER1"),
            ):
                traj_hist = ct.get(key, {}).get("historical", {})
                if traj_hist.get("x") and traj_hist.get("y"):
                    historical_series[alias] = {
                        "x": [str(v) for v in traj_hist["x"]],
                        "y": [round(float(v) * 100, 2)
                              if float(v) < 2 else round(float(v), 2)
                              for v in traj_hist["y"]],
                    }

    # Build updated DAG with highlights
    event     = IDIO_EVENTS[event_id]
    dag_chart = build_dag_figure(event, result.get("dag_highlights"))

    return jsonify({
        "ratios":            result["ratios"],
        "dag_highlights":    result["dag_highlights"],
        "dag_chart":         dag_chart,
        "historical_series": historical_series,
    })


# ════════════════════════════════════════════════════════════════════
# PHASE 1 — Estimate models (background thread + progress)
# ════════════════════════════════════════════════════════════════════
@dashboard_bp.route("/api/phase1-estimate", methods=["POST"])
def phase1_estimate():
    """Launch model estimation in background. Returns job_id immediately."""
    data         = request.get_json(force=True)
    modules      = data.get("modules", ["climate"])
    scenario_type= data.get("scenario_type", "ngfs")
    scenario_id  = data.get("scenario_id",   "net_zero")
    params       = data.get("params",        {})
    file_paths   = data.get("file_paths",    {})

    upload_paths = _automap_files(file_paths)
    sess_paths   = session.get("upload_paths", {})
    for k, v in sess_paths.items():
        upload_paths.setdefault(k, v)
    upload_paths = _enrich_upload_paths(upload_paths)

    scn_mgr  = ScenarioManager(current_app.config["SCENARIOS_FOLDER"])
    scenario = scn_mgr.get_scenario(scenario_type, scenario_id, params)

    jobs_folder = _get_jobs_folder()
    job_store   = JobStore(jobs_folder)
    job_id      = job_store.create("phase1")
    job_store.add_step(job_id, "⏳ Démarrage de l'estimation...", 0)

    runs_folder   = current_app.config["RUNS_FOLDER"]
    scenarios_folder = current_app.config["SCENARIOS_FOLDER"]

    # Phase 1 only needs the model-ranking and scenario-ranking tables to
    # populate the Step 3a/3b selection UI. The heavy engine integration
    # (NGFS PD/LGD refit, NGFS LCR/NSFR, full ClimateOrchestrator across
    # credit/liquidity/market × every scenario × year) is irrelevant here —
    # it gets recomputed from scratch in Phase 2 once the user has actually
    # chosen a model/scenario pair. Skip it so Phase 1 doesn't silently run
    # (and display as if final) results based on an auto-selected model
    # rank #1 the user hasn't picked yet.
    params = dict(params)
    params["_climate_lightweight_phase1"] = True

    def run_phase1():
        stop_evt = threading.Event()
        prog_thread = threading.Thread(
            target=_progress_writer,
            args=(job_store, job_id, PHASE1_STEPS, stop_evt),
            daemon=True)
        prog_thread.start()
        try:
            orch   = RunOrchestrator(runs_folder)
            record = orch.execute(modules, upload_paths, params, scenario)
            model_table      = []
            scenario_ranking = []
            auto_adverse     = ""
            auto_severe      = ""
            auto_score       = 0
            model_used       = "N/A"
            climate_error    = record.get("errors", {}).get("climate", "")
            credit_error     = record.get("errors", {}).get("credit",  "")
            liq_tournaments  = {}
            liq_calib_error  = ""
            mkt_satellites   = {}

            # ── Extract model_table from CLIMATE results ──────────────
            clim = record.get("module_results", {}).get("climate", {})
            if clim:
                bm = clim.get("baseline", {})
                cd = bm.get("charts_data", {})
                md = bm.get("metadata", {})
                ss = md.get("scenario_selection", {})
                model_table = cd.get("model_table", []) or md.get("model_table", [])
                scenario_ranking = ss.get("ranking", [])
                if not scenario_ranking:
                    scenario_ranking = cd.get("scenario_combos", [])
                auto_adverse = ss.get("selected_adverse", "") or cd.get("auto_adverse", "")
                auto_severe  = ss.get("selected_severe",  "") or cd.get("auto_severe",  "")
                auto_score   = ss.get("score", 0) or cd.get("auto_score", 0)
                model_used   = ss.get("model_used", "") or cd.get("model_used", "N/A")
                # 3-risk calibration integration — liquidité (5 tournois de
                # composants bilan) + marché (β₁/β₂/β₃, candidats variables)
                liq_tournaments = cd.get("liq_calib_tournaments", {})
                liq_calib_error = cd.get("liq_calib_error", "")
                mkt_satellites  = cd.get("mkt_beta_satellites", {})
                LOG.info(
                    "Phase1 liquidité/marché: %d tournoi(s) liquidité, "
                    "%d satellite(s) marché",
                    len(liq_tournaments), len(mkt_satellites),
                )
                if not model_table and not climate_error:
                    climate_error = (bm.get("kpis") or {}).get("error", "")
                    if not climate_error:
                        climate_error = ", ".join(bm.get("warnings") or [])
                LOG.info("Phase1 climate: scenario_ranking=%d combos, model_table=%d models",
                         len(scenario_ranking), len(model_table))
                if climate_error:
                    LOG.warning("Phase1 climate error: %s", climate_error)

            # ── Extract model_table from CREDIT results (when no climate) ─
            if not model_table:
                cred = record.get("module_results", {}).get("credit", {})
                if cred:
                    bm_c = (cred.get("baseline")
                            or cred.get("severe")
                            or cred.get("adverse")
                            or {})
                    cd_c = bm_c.get("charts_data", {})
                    model_table = cd_c.get("model_table", [])
                    model_used  = "satellite_credit"
                    if not model_table and not credit_error:
                        credit_error = (bm_c.get("kpis") or {}).get("error", "")
                    LOG.info("Phase1 credit: model_table=%d models", len(model_table))
                    if credit_error:
                        LOG.warning("Phase1 credit error: %s", credit_error)

            stop_evt.set()
            engine_error = climate_error or credit_error
            if model_table:
                job_store.add_step(job_id, "✅ Estimation terminée — tableaux prêts !", 100)
            else:
                step_msg = (f"⚠️ Moteur terminé sans modèles. "
                            f"{engine_error[:120] if engine_error else 'Vérifiez les fichiers et paramètres.'}")
                job_store.add_step(job_id, step_msg, 100)
            job_store.set_done(job_id, {
                "source_run_id":     record["run_id"],
                "model_table":       model_table,
                "scenario_ranking":  scenario_ranking,
                "auto_adverse":      auto_adverse,
                "auto_severe":       auto_severe,
                "auto_score":        auto_score,
                "model_used":        model_used,
                "liq_tournaments":   liq_tournaments,
                "liq_calib_error":   liq_calib_error,
                "mkt_satellites":    mkt_satellites,
                "run_name":          record["run_name"],
                "modules":           modules,
                "scenario":          scenario,
                "params":            params,
                "upload_paths":      upload_paths,
                "climate_error":     climate_error,
                "credit_error":      credit_error,
            })
        except Exception as e:
            stop_evt.set()
            LOG.exception("Phase 1 failed: %s", e)
            job_store.set_error(job_id, str(e))

    threading.Thread(target=run_phase1, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


# ════════════════════════════════════════════════════════════════════
# JOB STATUS — polling endpoint
# ════════════════════════════════════════════════════════════════════
@dashboard_bp.route("/api/job-status/<job_id>")
def job_status(job_id):
    jobs_folder = _get_jobs_folder()
    job_store   = JobStore(jobs_folder)
    job = job_store.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    result = job.get("result") or {}
    return jsonify({
        "job_id":            job["job_id"],
        "status":            job["status"],
        "steps":             job.get("steps", []),
        "progress_pct":      job.get("progress_pct", 0),
        "model_table":       result.get("model_table", []),
        "scenario_ranking":  result.get("scenario_ranking", []),
        "auto_adverse":      result.get("auto_adverse", ""),
        "auto_severe":       result.get("auto_severe", ""),
        "auto_score":        result.get("auto_score", 0),
        "model_used":        result.get("model_used", "N/A"),
        "liq_tournaments":   result.get("liq_tournaments", {}),
        "liq_calib_error":   result.get("liq_calib_error", ""),
        "mkt_satellites":    result.get("mkt_satellites", {}),
        "source_run_id":     result.get("source_run_id", ""),
        "run_id":            result.get("run_id", ""),
        "run_name":          result.get("run_name", ""),
        "error":             job.get("error"),
        "climate_error":     result.get("climate_error", ""),
        "credit_error":      result.get("credit_error",  ""),
    })


# ════════════════════════════════════════════════════════════════════
# PHASE 2 — Project with chosen model (background + progress)
# ════════════════════════════════════════════════════════════════════
@dashboard_bp.route("/api/phase2-project", methods=["POST"])
def phase2_project():
    """Re-run with forced model rank. Returns job_id immediately."""
    data          = request.get_json(force=True)
    source_run_id = data.get("source_run_id", "")
    model_rank    = int(data.get("model_rank", 0))
    modules       = data.get("modules", ["credit"])
    scenario_type = data.get("scenario_type", "historical")
    scenario_id   = data.get("scenario_id",   "gulf_war")
    params        = data.get("params",        {})
    file_paths    = data.get("file_paths",    {})

    params["forced_model_rank"] = model_rank
    # 3-risk calibration integration (Étape 3, onglets Liquidité/Marché) —
    # overrides optionnels choisis par l'utilisateur, transmis tels quels ;
    # voir ngfs_liquidity_engine.compute_ngfs_lcr_nsfr(forced_sat_ranks=...)
    # et IRSatellite.fit_all(forced_include=..., forced_exclude=...).
    if data.get("forced_liq_sat_ranks"):
        params["forced_liq_sat_ranks"] = data["forced_liq_sat_ranks"]
    if data.get("forced_mkt_include"):
        params["forced_mkt_include"] = data["forced_mkt_include"]
    if data.get("forced_mkt_exclude"):
        params["forced_mkt_exclude"] = data["forced_mkt_exclude"]
    params.setdefault("horizon_start", 2025)
    params.setdefault("horizon_end",   2029)

    upload_paths = _automap_files(file_paths)
    sess_paths   = session.get("upload_paths", {})
    for k, v in sess_paths.items():
        upload_paths.setdefault(k, v)
    upload_paths = _enrich_upload_paths(upload_paths)

    scn_mgr  = ScenarioManager(current_app.config["SCENARIOS_FOLDER"])
    scenario = scn_mgr.get_scenario(scenario_type, scenario_id, params)

    jobs_folder   = _get_jobs_folder()
    job_store     = JobStore(jobs_folder)
    job_id        = job_store.create("phase2")
    job_store.add_step(job_id, "⏳ Lancement des projections...", 0)

    runs_folder = current_app.config["RUNS_FOLDER"]

    def run_phase2():
        stop_evt = threading.Event()
        prog_thread = threading.Thread(
            target=_progress_writer,
            args=(job_store, job_id, PHASE2_STEPS, stop_evt),
            daemon=True)
        prog_thread.start()
        try:
            orch   = RunOrchestrator(runs_folder)
            record = orch.execute(modules, upload_paths, params, scenario)
            stop_evt.set()
            job_store.add_step(job_id, "✅ Dashboard prêt !", 100)
            job_store.set_done(job_id, {
                "run_id":   record["run_id"],
                "run_name": record["run_name"],
                "modules":  modules,
            })
        except Exception as e:
            stop_evt.set()
            LOG.exception("Phase 2 failed: %s", e)
            job_store.set_error(job_id, str(e))

    threading.Thread(target=run_phase2, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


# ════════════════════════════════════════════════════════════════════
# EXISTING ENDPOINTS (kept)
# ════════════════════════════════════════════════════════════════════
@dashboard_bp.route("/api/run-stress", methods=["POST"])
def run_stress():
    data       = request.get_json(force=True)
    modules    = data.get("modules", ["credit"])
    scn_type   = data.get("scenario_type", "historical")
    scn_id     = data.get("scenario_id",   "gulf_war")
    params     = data.get("params",        {})
    file_paths = data.get("file_paths",    {})
    upload_paths = _automap_files(file_paths)
    sess_paths   = session.get("upload_paths", {})
    for k, v in sess_paths.items():
        upload_paths.setdefault(k, v)
    upload_paths = _enrich_upload_paths(upload_paths)
    scn_mgr  = ScenarioManager(current_app.config["SCENARIOS_FOLDER"])
    scenario = scn_mgr.get_scenario(scn_type, scn_id, params)
    params.setdefault("ead", 5_000_000_000)
    params.setdefault("lgd_ttc", 0.45)
    params.setdefault("pd_ttc", 0.03)
    params.setdefault("horizon_start", 2025)
    params.setdefault("horizon_end",   2029)
    orch   = RunOrchestrator(current_app.config["RUNS_FOLDER"])
    record = orch.execute(modules, upload_paths, params, scenario)
    return jsonify({"run_id":record["run_id"],"run_name":record["run_name"],
                    "scenario_label":record["scenario_label"],
                    "errors":record.get("errors",{}),"status":"done"})


@dashboard_bp.route("/api/save-scenario", methods=["POST"])
def save_scenario():
    data = request.get_json(force=True)
    mgr  = ScenarioManager(current_app.config["SCENARIOS_FOLDER"])
    sid  = mgr.save_scenario(data["name"], "parametric", data.get("params", {}))
    return jsonify({"saved_id": sid})


@dashboard_bp.route("/api/rerun-with-model", methods=["POST"])
def rerun_with_model():
    data          = request.get_json(force=True)
    source_run_id = data.get("run_id")
    model_rank    = data.get("model_rank", 0)
    store  = RunStore(current_app.config["RUNS_FOLDER"])
    source = store.load(source_run_id)
    if not source:
        return jsonify({"error": "Source run not found"}), 404
    modules = source.get("selected_modules", ["climate"])
    params  = dict(source.get("params", {}))
    params["forced_model_rank"] = model_rank
    scn_mgr  = ScenarioManager(current_app.config["SCENARIOS_FOLDER"])
    scenario = scn_mgr.get_scenario(
        source.get("scenario_type","historical"),
        source.get("scenario_id","gulf_war"), params)
    upload_paths = _automap_files(data.get("file_paths", {}))
    sess_paths   = session.get("upload_paths", {})
    for k, v in sess_paths.items():
        upload_paths.setdefault(k, v)
    upload_paths = _enrich_upload_paths(upload_paths)
    orch   = RunOrchestrator(current_app.config["RUNS_FOLDER"])
    record = orch.execute(modules, upload_paths, params, scenario)
    source_results = source.get("module_results",{}).get("climate",{})
    model_table = (source_results.get("baseline",{})
                                 .get("charts_data",{})
                                 .get("model_table",[]))
    model_label = ""
    if model_table and model_rank < len(model_table):
        m = model_table[model_rank]
        model_label = f"{m.get('family','?')}({','.join((m.get('variables') or [])[:2])})"
    record["run_name"] = f"Modele_{model_rank+1}_{model_label}"[:40]
    store.save(record["run_id"], record)
    return jsonify({"run_id":record["run_id"],"run_name":record["run_name"],
                    "errors":record.get("errors",{}),"status":"done"})


@dashboard_bp.route("/api/rerun-with-sat-models", methods=["POST"])
def rerun_with_sat_models():
    """Re-run liquidity module with per-satellite forced ranks.

    Body: { run_id, sat_ranks: {satellite_var: rank_0idx}, file_paths }
    """
    data          = request.get_json(force=True)
    source_run_id = data.get("run_id")
    sat_ranks     = {k: int(v) for k, v in (data.get("sat_ranks") or {}).items()}
    store  = RunStore(current_app.config["RUNS_FOLDER"])
    source = store.load(source_run_id)
    if not source:
        return jsonify({"error": "Source run not found"}), 404

    modules = source.get("selected_modules", ["liquidity"])
    params  = dict(source.get("params", {}))
    params["forced_sat_ranks"] = sat_ranks

    scn_mgr  = ScenarioManager(current_app.config["SCENARIOS_FOLDER"])
    scenario = scn_mgr.get_scenario(
        source.get("scenario_type", "historical"),
        source.get("scenario_id",   "gulf_war"), params)

    upload_paths = _automap_files(data.get("file_paths", {}))
    sess_paths   = session.get("upload_paths", {})
    for k, v in sess_paths.items():
        upload_paths.setdefault(k, v)
    upload_paths = _enrich_upload_paths(upload_paths)

    orch   = RunOrchestrator(current_app.config["RUNS_FOLDER"])
    record = orch.execute(modules, upload_paths, params, scenario)

    label_parts = [f"{sat[:8]}R{rank+1}" for sat, rank in sat_ranks.items()]
    record["run_name"] = ("Liq_" + "_".join(label_parts))[:40] if label_parts else "Liq_rerun"
    store.save(record["run_id"], record)
    return jsonify({"run_id":   record["run_id"],
                    "run_name": record["run_name"],
                    "errors":   record.get("errors", {}),
                    "status":   "done"})


@dashboard_bp.route("/compare")
def compare():
    return render_template("compare.html",
                           runs=RunStore(current_app.config["RUNS_FOLDER"]).list_runs())

@dashboard_bp.route("/runs")
def runs_page():
    return render_template("runs.html",
                           runs=RunStore(current_app.config["RUNS_FOLDER"]).list_runs())

@dashboard_bp.route("/scenarios")
def scenarios():
    return render_template("scenarios.html", modules=list_modules())
