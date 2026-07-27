import logging
from flask import Blueprint, render_template, request, current_app
from ..services.run_store import RunStore
from ..services.chart_builder import ChartBuilder

transmission_bp = Blueprint("transmission", __name__)
LOG = logging.getLogger("transmission")

@transmission_bp.route("/transmission")
def transmission():
    store  = RunStore(current_app.config["RUNS_FOLDER"])
    run_id = request.args.get("run_id")
    runs   = store.list_runs()
    if not run_id and runs:
        run_id = runs[0]["run_id"]
    record = store.load(run_id) if run_id else None
    charts = {}
    if record:
        try:
            builder = ChartBuilder(record)
            charts  = builder.build_transmission()
        except Exception as e:
            LOG.exception("Transmission charts failed: %s", e)
    return render_template("transmission.html", run_id=run_id, record=record,
                           runs=runs, charts=charts)
