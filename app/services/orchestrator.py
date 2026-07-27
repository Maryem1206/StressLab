"""Central orchestrator — runs only selected modules, all 3 levels."""
from __future__ import annotations
import logging, uuid
from datetime import datetime
from typing import Any, Dict, List
from .run_store import RunStore
from .consolidator import ResultConsolidator
from ..modules.registry import get_module_class

LOG = logging.getLogger("orchestrator")


class RunOrchestrator:
    def __init__(self, runs_folder):
        self.store = RunStore(runs_folder)

    def execute(self, selected_modules: List[str], upload_paths: Dict,
                params: Dict, scenario: Dict) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())[:8]
        run_ts = datetime.utcnow().isoformat()
        run_name = "Run_" + datetime.utcnow().strftime("%Y%m%d_%H%M")

        LOG.info("[%s] Modules=%s | Scenario=%s", run_id, selected_modules,
                 scenario.get("label","?"))

        all_results, errors = {}, {}

        for mid in selected_modules:
            try:
                Cls = get_module_class(mid)
                wrapper = Cls(upload_paths=upload_paths, params=params, scenario=scenario)
                results = wrapper.run()
                all_results[mid] = results
                LOG.info("[%s] Module %s done", run_id, mid)
            except Exception as e:
                import traceback as _tb
                LOG.exception("[%s] Module %s failed: %s", run_id, mid, e)
                errors[mid] = f"{type(e).__name__}: {e}\n{_tb.format_exc()}"

        consolidated = ResultConsolidator().consolidate(all_results)
        record = {
            "run_id": run_id, "run_name": run_name, "timestamp": run_ts,
            "selected_modules": selected_modules,
            "scenario_label": scenario.get("label",""),
            "scenario_type": scenario.get("type",""),
            "scenario_id": scenario.get("id",""),
            "errors": errors, "consolidated": consolidated,
            "module_results": {
                mid: {lvl: r.to_dict() for lvl, r in scns.items()}
                for mid, scns in all_results.items()
            },
        }
        self.store.save(run_id, record)
        return record
