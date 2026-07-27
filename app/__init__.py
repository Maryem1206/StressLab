import logging, os
from flask import Flask

def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config["SECRET_KEY"]        = "stresslab-v2-2025"
    app.config["MAX_CONTENT_LENGTH"]= 200 * 1024 * 1024
    app.config["UPLOAD_FOLDER"]     = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
    app.config["EXPORT_FOLDER"]     = os.path.join(os.path.dirname(os.path.dirname(__file__)), "exports")
    app.config["RUNS_FOLDER"]       = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")
    app.config["SCENARIOS_FOLDER"]  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scenarios")
    app.config["MODULES_SRC"]       = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules_src")
    app.config["JOBS_FOLDER"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", "jobs")
    for k in ("UPLOAD_FOLDER","EXPORT_FOLDER","RUNS_FOLDER","SCENARIOS_FOLDER","JOBS_FOLDER"):
        os.makedirs(app.config[k], exist_ok=True)
    # Journalise à la fois dans la console ET dans un fichier — la console
    # (fenêtre de terminal) a un historique de défilement limité, donc un
    # run long (ex. module climat) fait disparaître le début du log avant
    # la fin de l'exécution. Le fichier conserve tout, sans limite.
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "app.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )
    from .blueprints.upload   import upload_bp
    from .blueprints.dashboard import dashboard_bp
    from .blueprints.transmission import transmission_bp
    from .blueprints.api      import api_bp
    from .blueprints.export   import export_bp
    app.register_blueprint(upload_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(transmission_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(export_bp, url_prefix="/export")
    return app
