from flask import Blueprint, send_file, jsonify, current_app
from ..services.run_store import RunStore
from ..services.exporter import ExportService

export_bp = Blueprint("export", __name__)

@export_bp.route("/excel/<run_id>")
def excel(run_id):
    r = RunStore(current_app.config["RUNS_FOLDER"]).load(run_id)
    if not r: return jsonify({"error":"Not found"}),404
    p = ExportService(current_app.config["EXPORT_FOLDER"]).to_excel(r)
    return send_file(str(p), as_attachment=True, download_name=p.name)

@export_bp.route("/pdf/<run_id>")
def pdf(run_id):
    r = RunStore(current_app.config["RUNS_FOLDER"]).load(run_id)
    if not r: return jsonify({"error":"Not found"}),404
    p = ExportService(current_app.config["EXPORT_FOLDER"]).to_pdf_html(r)
    return send_file(str(p), mimetype="text/html")
