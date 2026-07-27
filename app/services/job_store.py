"""Background job store — tracks Phase 1 / Phase 2 execution status."""
import json, threading, time, uuid, logging
from pathlib import Path

LOG = logging.getLogger("job_store")


class JobStore:
    def __init__(self, jobs_folder: str):
        self.folder = Path(jobs_folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────
    def create(self, job_type: str) -> str:
        job_id = str(uuid.uuid4())[:10]
        self._write(job_id, {
            "job_id":       job_id,
            "type":         job_type,
            "status":       "running",
            "steps":        [],
            "progress_pct": 0,
            "result":       None,
            "error":        None,
            "created_at":   time.time(),
        })
        return job_id

    def add_step(self, job_id: str, msg: str, pct: int):
        d = self._read(job_id)
        if d:
            d["steps"].append({"msg": msg, "done": True})
            d["progress_pct"] = pct
            self._write(job_id, d)

    def set_done(self, job_id: str, result: dict):
        d = self._read(job_id)
        if d:
            d["status"]       = "done"
            d["progress_pct"] = 100
            d["result"]       = result
            self._write(job_id, d)

    def set_error(self, job_id: str, error: str):
        d = self._read(job_id)
        if d:
            d["status"] = "error"
            d["error"]  = error
            self._write(job_id, d)

    def get(self, job_id: str) -> dict:
        return self._read(job_id)

    # ── Helpers ───────────────────────────────────────────────────────
    def _path(self, job_id):
        return self.folder / f"{job_id}.json"

    def _write(self, job_id, data):
        with self._lock:
            self._path(job_id).write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")

    def _read(self, job_id):
        p = self._path(job_id)
        if not p.exists():
            return None
        with self._lock:
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
