"""Consolidates results from all modules into unified KPIs + chart data."""
import logging
LOG = logging.getLogger("consolidator")
COLORS = {"baseline": "#2563EB", "adverse": "#FE7C39", "severe": "#FD5108"}
LABELS = {"baseline": "Baseline", "adverse": "Adverse", "severe": "Severe"}


class ResultConsolidator:
    def consolidate(self, all_results):
        return {
            "kpi_cards":  self._kpis(all_results),
            "pd_chart":   self._line_chart(all_results, "pd_trajectories", "PD"),
            "el_chart":   self._line_chart(all_results, "el_trajectories", "EL"),
            "waterfall":  self._waterfall(all_results),
            "radar":      self._radar(all_results),
            "heatmap":    self._heatmap(all_results),
            "ratios":     self._ratios(all_results),
            "summary":    self._summary(all_results),
        }

    def _kpis(self, ar):
        c = {}
        for mod, scns in ar.items():
            for lvl, r in scns.items():
                if r.kpis.get("status") == "coming_soon": continue
                for k, v in r.kpis.items():
                    c[f"{mod}_{lvl}_{k}"] = v
        # Aggregated cross-module
        for lvl in ("baseline","adverse","severe"):
            el_total = sum(
                scns.get(lvl, type("", (), {"kpis": {}})()).kpis.get("total_el", 0) or 0
                for scns in ar.values()
            )
            c[f"el_{lvl}"] = el_total
            for mod in ar:
                r = ar[mod].get(lvl)
                if r and "avg_pd" in r.kpis and "error" not in r.kpis:
                    c.setdefault(f"pd_{lvl}", r.kpis["avg_pd"])
                if r and "avg_lgd" in r.kpis:
                    c.setdefault(f"lgd_{lvl}", r.kpis["avg_lgd"])
        if c.get("pd_baseline") and c.get("pd_severe"):
            c["pd_delta_pp"] = round((c["pd_severe"] - c["pd_baseline"]) * 100, 2)
        if c.get("el_baseline") and c.get("el_severe") and c["el_baseline"] > 0:
            c["el_delta_pct"] = round((c["el_severe"]/c["el_baseline"] - 1) * 100, 1)
        if c.get("lgd_baseline") and c.get("lgd_severe"):
            c["lgd_delta_pp"] = round((c["lgd_severe"] - c["lgd_baseline"]) * 100, 2)
        return c

    def _line_chart(self, ar, key, title):
        for mod, scns in ar.items():
            bm = scns.get("baseline")
            if bm and bm.charts_data.get(key):
                traj = bm.charts_data[key]
                return {
                    "traces": [
                        {"x": traj.get(lvl, {}).get("x", []),
                         "y": traj.get(lvl, {}).get("y", []),
                         "name": LABELS[lvl], "color": COLORS[lvl]}
                        for lvl in ("baseline","adverse","severe")
                        if traj.get(lvl, {}).get("x")
                    ],
                    "title": title,
                }
        return {"traces": [], "title": title}

    def _waterfall(self, ar):
        items = []
        for mod, scns in ar.items():
            b, s = scns.get("baseline"), scns.get("severe")
            if b and s and b.kpis.get("status") != "coming_soon":
                items.append({
                    "module": b.module_label,
                    "delta":  round(s.total_loss - b.total_loss, 4)
                })
        return {"items": items}

    def _radar(self, ar):
        cats, vb, vs = [], [], []
        for mod, scns in ar.items():
            b, s = scns.get("baseline"), scns.get("severe")
            if b and s and b.kpis.get("status") != "coming_soon":
                cats.append(b.module_label)
                vb.append(round(min(b.total_loss * 100, 100), 1))
                vs.append(round(min(s.total_loss * 100, 100), 1))
        return {"categories": cats, "baseline": vb, "severe": vs}

    def _heatmap(self, ar):
        for mod, scns in ar.items():
            bm = scns.get("baseline")
            if bm and bm.charts_data.get("pd_trajectories"):
                pt = bm.charts_data["pd_trajectories"]
                x = pt.get("baseline", {}).get("x", [])
                z = [pt.get(lvl, {}).get("y", []) for lvl in ("baseline","adverse","severe")]
                return {"z": z, "x": x, "y": ["Baseline","Adverse","Severe"]}
        return {"z": [], "x": [], "y": []}

    def _ratios(self, ar):
        def _last_valid_pct(vals):
            for v in reversed(vals or []):
                if v is not None and v == v:  # NaN != NaN
                    return round(float(v) * 100, 2)
            return None

        METRICS = [
            ("cet1_ratio",  "CET1 Ratio"),
            ("tier1_ratio", "Tier 1 Ratio"),
            ("car",         "Ratio CAR"),
        ]

        ratios = {}
        for mod, scns in ar.items():
            bm = scns.get("baseline") or scns.get("adverse") or scns.get("severe")
            if not bm:
                continue
            ct = bm.charts_data.get("capital_trajectories", {})
            if not ct:
                continue
            for metric, label in METRICS:
                hist = ct.get(metric, {}).get("historical", {})
                baseline_pct = _last_valid_pct(hist.get("y", []))
                if baseline_pct is None:
                    baseline_pct = _last_valid_pct(
                        ct.get(metric, {}).get("baseline", {}).get("y", []))
                if baseline_pct is None:
                    continue
                stressed_pct = _last_valid_pct(
                    ct.get(metric, {}).get("severe", {}).get("y", []))
                if stressed_pct is None:
                    stressed_pct = _last_valid_pct(
                        ct.get(metric, {}).get("adverse", {}).get("y", []))
                if stressed_pct is None:
                    continue
                ratios[label] = {"baseline": baseline_pct, "stressed": stressed_pct}
            if ratios:
                break
        return ratios

    def _summary(self, ar):
        return {"rows": [
            {"module": mid,
             **{lvl: round(scns.get(lvl).total_loss, 2) if scns.get(lvl) else None
                for lvl in ("baseline","adverse","severe")}}
            for mid, scns in ar.items()
        ]}
