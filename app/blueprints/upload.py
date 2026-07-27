import logging
from pathlib import Path
from flask import Blueprint, render_template, request, jsonify, current_app, redirect, url_for
from werkzeug.utils import secure_filename

upload_bp = Blueprint("upload", __name__)
LOG = logging.getLogger("upload")
ALLOWED = {"csv","xlsx","xls","yaml","yml","json","zip"}

ROLE_MAP = {
    # Market risk — yield curve Excel (specific keywords, checked first)
    # NOTE: checked before the generic fallback; all common filename variants listed
    "yieldcurve":   "excel_path",   # YieldCurve.xlsx (no separator)
    "yield_curve":  "excel_path",
    "yield curve":  "excel_path",
    "yield-curve":  "excel_path",
    "courbe_taux":  "excel_path",
    "courbe taux":  "excel_path",
    "courbe-taux":  "excel_path",
    "tbill":        "excel_path",
    "tbond":        "excel_path",
    "sovereign":    "excel_path",
    # Credit / capital
    "capital":      "capital",
    "historique":   "historical",
    "historical":   "historical",
    "history":      "historical",
    "mon_histor":   "historical",   # mon_historique.csv default name
    # Climate — NGFS
    "nigem":        "ngfs",
    "ngfs_n":       "ngfs",
    "ngfs":         "ngfs",         # plain "ngfs" prefix
    "geme3":        "ngfs_ct",
    "iiasa":        "ngfs_ct",
    "gem_e3":       "ngfs_ct",
    "gem-e3":       "ngfs_ct",
    "mapping":      "mapping",
    # Liquidity
    "liquidity":       "liquidity",
    "liquidity_input": "liquidity",
    "lcr":             "liquidity",
    "nsfr":            "liquidity",
}

def _automap(raw: dict) -> dict:
    mapped = {}
    for key, fpath in raw.items():
        fname = Path(fpath).name.lower()
        ext   = Path(fpath).suffix.lower()
        role  = next((v for k, v in ROLE_MAP.items() if k in fname), None)
        if not role:
            # xlsx/xls files without a recognised name are likely market data,
            # NOT historical PD CSVs — don't silently corrupt the climate module
            if ext in (".xlsx", ".xls"):
                role = "excel_path"
            else:
                role = "historical"
        # Never overwrite an already-mapped role with a less-specific match
        mapped.setdefault(role, fpath)
    return mapped

@upload_bp.route("/")
def index():
    return redirect(url_for("upload.upload_page"))

@upload_bp.route("/upload")
def upload_page():
    return render_template("upload.html")

@upload_bp.route("/api/upload-file", methods=["POST"])
def upload_file():
    f = request.files.get("file")
    if not f or not f.filename: return jsonify({"error":"No file"}), 400
    ext = f.filename.rsplit(".",1)[-1].lower()
    if ext not in ALLOWED: return jsonify({"error":f"Type non autorisé: .{ext}"}), 400
    fname = secure_filename(f.filename)
    dest = Path(current_app.config["UPLOAD_FOLDER"]) / fname
    f.save(str(dest))
    return jsonify({"filename":fname,"path":str(dest),"size":dest.stat().st_size})

@upload_bp.route("/api/delete-file", methods=["POST"])
def delete_file():
    """Delete an uploaded file from the server."""
    data = request.get_json(force=True)
    filename = data.get("filename", "")
    if not filename:
        return jsonify({"error": "filename required"}), 400
    safe = secure_filename(filename)
    target = Path(current_app.config["UPLOAD_FOLDER"]) / safe
    if target.exists():
        target.unlink()
        LOG.info("Deleted upload: %s", safe)
    return jsonify({"status": "ok", "filename": safe})

@upload_bp.route("/api/save-session", methods=["POST"])
def save_session():
    """Save upload session info (country, file paths) for use in dashboard."""
    data = request.get_json(force=True)
    from flask import session
    session["upload_paths"] = _automap(data.get("file_paths", {}))
    session["country"]      = data.get("country", "Egypt")
    return jsonify({"status":"ok"})
