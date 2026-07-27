import logging
from pathlib import Path
import pandas as pd
LOG = logging.getLogger("exporter")

class ExportService:
    def __init__(self, folder):
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True)

    def to_excel(self, record):
        fn = f"stress_{record.get('run_id','x')}.xlsx"
        path = self.folder / fn
        with pd.ExcelWriter(str(path), engine="openpyxl") as w:
            rows = [{"Field":k,"Value":str(v)} for k,v in record.get("consolidated",{}).get("kpi_cards",{}).items()]
            pd.DataFrame(rows).to_excel(w, sheet_name="KPIs", index=False)
            for mid, scns in record.get("module_results",{}).items():
                for lvl, res in scns.items():
                    ts = res.get("time_series",{})
                    if ts:
                        pd.DataFrame.from_dict(ts, orient="index").to_excel(
                            w, sheet_name=f"{mid[:5]}_{lvl[:3]}", index=True)
        return path

    def to_pdf_html(self, record):
        fn = f"report_{record.get('run_id','x')}.html"
        path = self.folder / fn
        kpis = record.get("consolidated",{}).get("kpi_cards",{})
        rows = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k,v in kpis.items())
        html = f"""<!DOCTYPE html><html><head><meta charset=utf-8><title>Rapport</title>
<style>body{{font-family:Arial;margin:40px}}h1{{color:#FD5108}}
table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #ddd;padding:8px}}
th{{background:#FD5108;color:white}}</style></head><body>
<h1>Rapport Stress Test — {record.get("run_name","")}</h1>
<p>Run ID: {record.get("run_id","")} | Scenario: {record.get("scenario_label","")} | {record.get("timestamp","")}</p>
<table><tr><th>Indicateur</th><th>Valeur</th></tr>{rows}</table></body></html>"""
        path.write_text(html, encoding="utf-8"); return path
