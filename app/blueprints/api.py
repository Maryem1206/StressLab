from flask import Blueprint, jsonify, request, current_app, session
from ..services.run_store import RunStore
from ..modules.registry import list_modules
from ..services.scenario_manager import ScenarioManager, HISTORICAL_SCENARIOS, NGFS_SCENARIOS
from ..services.idio_events import IDIO_EVENTS, build_dag_figure
from ..services.idio_engine import (
    get_base_values, compute_shock, flat_base, BASE_LABELS
)

api_bp = Blueprint("api", __name__)

@api_bp.route("/modules")
def get_modules(): return jsonify(list_modules())

@api_bp.route("/runs")
def get_runs(): return jsonify(RunStore(current_app.config["RUNS_FOLDER"]).list_runs())

@api_bp.route("/runs/<run_id>")
def get_run(run_id):
    r = RunStore(current_app.config["RUNS_FOLDER"]).load(run_id)
    return jsonify(r) if r else (jsonify({"error":"Not found"}),404)

@api_bp.route("/runs/<run_id>", methods=["DELETE"])
def del_run(run_id):
    return jsonify({"deleted": RunStore(current_app.config["RUNS_FOLDER"]).delete(run_id)})

@api_bp.route("/scenarios/historical")
def hist_scenarios():
    return jsonify([{"id":k,"label":v["label"],"icon":v.get("icon","📊")} for k,v in HISTORICAL_SCENARIOS.items()])

@api_bp.route("/scenarios/ngfs")
def ngfs_scenarios():
    return jsonify([{"id":k,"label":v["label"],"icon":v.get("icon","🌍")} for k,v in NGFS_SCENARIOS.items()])

@api_bp.route("/scenarios/saved")
def saved_scenarios():
    return jsonify(ScenarioManager(current_app.config["SCENARIOS_FOLDER"]).list_saved())


# ── Idiosyncratic events ──────────────────────────────────────────────

@api_bp.route("/idio_event/<event_id>")
def idio_event(event_id):
    """Return event metadata + base values + initial DAG figure (no highlights)."""
    event = IDIO_EVENTS.get(event_id)
    if not event:
        return jsonify({"error": f"Unknown event: {event_id}"}), 404

    # Load run record if run_id provided
    run_id = request.args.get("run_id", "").strip()
    record = None
    if run_id:
        record = RunStore(current_app.config["RUNS_FOLDER"]).load(run_id)

    # Dataset path from session
    upload_paths = session.get("upload_paths", {})
    dataset_path = upload_paths.get("historical") or upload_paths.get("liquidity")

    # Base values (run → dataset → regulatory floors)
    base_wrapped = get_base_values(record=record, dataset_path=dataset_path)

    # Only expose fields required by this event
    required = set(event.get("required_base", []))
    base_for_ui = {
        k: {
            "value":  v["value"],
            "source": v["source"],
            "label":  BASE_LABELS.get(k, k),
        }
        for k, v in base_wrapped.items()
        if k in required
    }

    # Build DAG figure without highlights
    dag_fig = build_dag_figure(event, highlights=None)

    return jsonify({
        "icon":         event.get("icon", ""),
        "name":         event.get("name", ""),
        "category":     event.get("category", ""),
        "description":  event.get("description", ""),
        "trigger":      event.get("trigger", ""),
        "balance_sheet": event.get("balance_sheet", []),
        "shock_inputs": event.get("shock_inputs", []),
        "base_values":  base_for_ui,
        "dag_chart":    dag_fig,
    })


@api_bp.route("/idio_shock", methods=["POST"])
def idio_shock():
    """Compute ratio impacts and return updated DAG + ratio cards."""
    body = request.get_json(force=True) or {}
    event_id = body.get("event_id", "")
    inputs   = body.get("inputs", {})
    overrides = body.get("base", {})

    event = IDIO_EVENTS.get(event_id)
    if not event:
        return jsonify({"error": f"Unknown event: {event_id}"}), 404

    # Load run record if provided
    run_id = body.get("run_id", "").strip()
    record = None
    if run_id:
        record = RunStore(current_app.config["RUNS_FOLDER"]).load(run_id)

    upload_paths = session.get("upload_paths", {})
    dataset_path = upload_paths.get("historical") or upload_paths.get("liquidity")

    # Build base values, then apply user overrides
    base_wrapped = get_base_values(record=record, dataset_path=dataset_path)
    base_flat    = flat_base(base_wrapped)
    for k, v in overrides.items():
        try:
            base_flat[k] = float(v)
        except (TypeError, ValueError):
            pass

    # Compute shock
    result = compute_shock(event_id, inputs, base_flat)

    # Rebuild DAG with highlights
    dag_fig = build_dag_figure(event, highlights=result.get("dag_highlights", {}))

    return jsonify({
        "ratios":    result.get("ratios", {}),
        "dag_chart": dag_fig,
    })
