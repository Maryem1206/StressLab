import json, logging
from pathlib import Path
from datetime import datetime
LOG = logging.getLogger("run_store")

class RunStore:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def save(self, run_id, record):
        p = self.folder / f"{run_id}.json"
        p.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return p

    def load(self, run_id):
        p = self.folder / f"{run_id}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def list_runs(self):
        runs = []
        for p in sorted(self.folder.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                runs.append({"run_id":d.get("run_id",p.stem),"run_name":d.get("run_name",""),
                             "timestamp":d.get("timestamp",""),"modules":d.get("selected_modules",[]),
                             "scenario_label":d.get("scenario_label","")})
            except: pass
        return runs

    def delete(self, run_id):
        p = self.folder / f"{run_id}.json"
        if p.exists(): p.unlink(); return True
        return False
