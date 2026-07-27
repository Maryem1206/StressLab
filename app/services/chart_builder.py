"""Server-side chart generation — Plotly Python, zero CDN dependency.

NEW PwC colour palette (May 2026):
  Primary:  #FD5108 / #FE7C39 / #FFAA72
  Greys:    #A1A8B3 / #B5BCC4 / #CBD1D6
  Tints:    #FFCDA8 / #FFE8D4 / #DFE3E6 / #EEEFF1
  Status:   #059669 (success) / #E9B01F (warning) / #DC2626 (error)
  Accents:  #FF9F00 / #FA143C / #F05596
"""
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

_DEFAULT_HIST_CSV = (
    Path(__file__).resolve().parent.parent.parent
    / "modules_src" / "climate_module" / "mon_historique.csv"
)

LOG = logging.getLogger("chart_builder")

# ── Palette ──────────────────────────────────────────────────────────────────
P = {
    "orange1": "#FD5108", "orange2": "#FE7C39", "orange3": "#FFAA72",
    "dark":    "#1B1E24", "grey":    "#A1A8B3", "border":  "#B5BCC4",
    "light":   "#CBD1D6", "bg1":     "#DFE3E6", "bg2":     "#EEEFF1",
    "tint1":   "#FFCDA8", "tint2":   "#FFE8D4",
    "green":   "#059669", "yellow":  "#E9B01F", "red":     "#DC2626",
    "accent1": "#FF9F00", "accent2": "#FA143C", "accent3": "#F05596",
}
C3  = {"baseline": P["dark"], "adverse": P["orange2"], "severe": P["orange1"]}
LBL = {"baseline": "Baseline", "adverse": "Adverse", "severe": "Sévère"}

_PROJ_MAX_YEAR = 2028


def _clip_proj(d, max_year=_PROJ_MAX_YEAR):
    """Return {x, y} keeping only entries where year <= max_year."""
    rx, ry = [], []
    for x, y in zip(d.get("x", []), d.get("y", [])):
        try:
            if int(x) <= max_year:
                rx.append(x)
                ry.append(y)
        except (ValueError, TypeError):
            pass
    return {"x": rx, "y": ry}


def _load_historical_pd(pt):
    """Return historical {x, y} from stored pt dict, or fall back to default CSV."""
    h = pt.get("historical", {})
    if h.get("x"):
        return h
    try:
        if _DEFAULT_HIST_CSV.exists():
            _head = _DEFAULT_HIST_CSV.read_text(encoding="utf-8", errors="ignore")[:500]
            _sep  = ";" if _head.count(";") > _head.count(",") else ","
            _hdf  = pd.read_csv(str(_DEFAULT_HIST_CSV), sep=_sep)
            _yr   = next((c for c in _hdf.columns if "year" in c.lower()), _hdf.columns[0])
            _pdcol = next((c for c in _hdf.columns if "default" in c.lower()), _hdf.columns[1])
            _hdf  = _hdf[[_yr, _pdcol]].dropna()
            return {
                "x": [int(v) for v in _hdf[_yr].tolist()],
                "y": [float(v) for v in _hdf[_pdcol].tolist()],
            }
    except Exception as _e:
        LOG.warning("Historical PD fallback read failed: %s", _e)
    return {}

LAY = dict(
    margin=dict(t=18, r=20, b=44, l=56),
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif", size=11,
              color=P["dark"]),
    legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
    xaxis=dict(gridcolor=P["bg2"], linecolor=P["border"], zeroline=False),
    yaxis=dict(gridcolor=P["bg2"], linecolor=P["border"], zeroline=False),
)


def _rgba(hex_c, alpha):
    h = hex_c.lstrip("#")
    return f"rgba({int(h[:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


def _safe(vals):
    return [None if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
            else float(v) for v in vals]


def _div(fig, include_js=False):
    return pio.to_html(fig, full_html=False,
                       include_plotlyjs=True if include_js else False,
                       config={"responsive": True, "displayModeBar": False})


def _empty(msg):
    return (f'<div style="display:flex;align-items:center;justify-content:center;'
            f'min-height:220px;color:{P["grey"]};font-size:12px;'
            f'background:{P["bg2"]};border-radius:8px">{msg}</div>')


def _ts_to_lists(ts_raw):
    """Convert time_series from JSON format {year: {col: val}} to {col: [vals]}."""
    if isinstance(ts_raw, dict) and ts_raw:
        first_val = next(iter(ts_raw.values()), None)
        if isinstance(first_val, dict):
            # JSON format: {"2024": {"pd": 0.03, "lgd": 0.45}, ...}
            years = sorted(ts_raw.keys())
            cols = list(first_val.keys()) if first_val else []
            result = {"year": [int(y) if y.isdigit() else y for y in years]}
            for c in cols:
                result[c] = [ts_raw[y].get(c, 0) for y in years]
            return result
        elif isinstance(first_val, (int, float)):
            # Already list-like somehow
            return ts_raw
    return {}


# ═════════════════════════════════════════════════════════════════════════════
_COUNTRY_CURRENCY_MILLIONS = {
    # Full country names
    "Egypt":          "M EGP",
    "France":         "M€",
    "Germany":        "M€",
    "United Kingdom": "M£",
    "Morocco":        "M MAD",
    "Tunisia":        "M TND",
    "Algeria":        "M DZD",
    "Nigeria":        "M NGN",
    "South Africa":   "M ZAR",
    "Senegal":        "M XOF",
    "Kenya":          "M KES",
    "Ghana":          "M GHS",
    "MENA":           "M$",
    "GCC":            "M SAR",
    "North Africa":   "M MAD",
    "Europe":         "M€",
    # ISO2 codes (what wrapper.py actually stores in metadata)
    "EG": "M EGP",
    "MA": "M MAD",
    "TN": "M TND",
    "DZ": "M DZD",
    "NG": "M NGN",
    "ZA": "M ZAR",
    "SN": "M XOF",
    "KE": "M KES",
    "GH": "M GHS",
    "FR": "M€",
    "DE": "M€",
    "GB": "M£",
    "IT": "M€",
    "ES": "M€",
    "SA": "M SAR",
    "AE": "M AED",
    "QA": "M QAR",
    "KW": "M KWD",
    "BH": "M BHD",
    "OM": "M OMR",
    "JO": "M JOD",
    "TR": "M TRY",
    "US": "M USD",
    "BR": "M BRL",
    "IN": "M INR",
    "CN": "M CNY",
    "JP": "M JPY",
}


class ChartBuilder:
    def __init__(self, record):
        self.record = record
        self.cons   = record.get("consolidated", {})
        self._first = True

    def _currency_millions(self) -> str:
        """Return the currency millions label for the current run's country."""
        country = (
            self.record.get("module_results", {})
                       .get("credit", {})
                       .get("baseline", {})
                       .get("metadata", {})
                       .get("country", "")
        )
        return _COUNTRY_CURRENCY_MILLIONS.get(country, "M$")

    def build_for_module(self, module_id: str) -> dict:
        self._first = True
        if module_id == "credit":    return self._credit_charts()
        if module_id == "climate":   return self._climate_charts()
        if module_id == "liquidity": return self._liquidity_charts()
        if module_id == "market":    return self._market_charts()
        return self._all_charts()

    def build_transmission(self) -> dict:
        self._first = True
        return {k: self._safe_build(m) for k, m in [
            ("sankey",      self._sankey),
            ("network",     self._network_matrix),
            ("propagation", self._propagation_timeline),
        ]}

    # ── Chart sets per module ────────────────────────────────────────────────
    def _credit_charts(self):
        self._first = True
        return {k: self._safe_build(m) for k, m in [
            ("pd_chart",       self._credit_pd),
            ("lgd_chart",      self._credit_lgd),
            ("ecl_chart",      self._credit_ecl),
            ("macro_contrib",  self._credit_macro_contrib),
            ("capital_ratio",  self._credit_capital_ratio),
            ("leverage_chart", self._credit_leverage),
            ("cet1_chart",     self._credit_cet1_ratio),
            ("tier1_chart",    self._credit_tier1_ratio),
            ("waterfall",      self._credit_waterfall),
            ("ratios",         self.ratios_chart),
            ("model_table",    self._credit_model_table),
        ]}

    def _climate_charts(self):
        self._first = True
        # ngfs_comparison must be first: it is the first chart rendered in the template
        # and therefore must carry the Plotly.js embed (include_js=True).
        # climate_pd is built last because it is not shown in the current template.
        return {k: self._safe_build(m) for k, m in [
            ("ngfs_comparison", self._climate_ngfs_comparison),
            ("climate_wf",      self._climate_waterfall),
            ("transition_phys", self._climate_transition_physical),
            ("ratios",          self.ratios_chart),
            ("climate_pd",      self._climate_impact_pd),
        ]}

    def _liquidity_charts(self):
        self._first = True
        return {k: self._safe_build(m) for k, m in [
            ("lcr_nsfr", self._liquidity_lcr_nsfr),
            ("ratios",   self._liquidity_ratios_chart),
        ]}

    def _all_charts(self):
        self._first = True
        return {k: self._safe_build(m) for k, m in [
            ("waterfall", self._credit_waterfall),
            ("ratios",    self.ratios_chart),
        ]}

    def _safe_build(self, method):
        try:
            return method()
        except Exception as e:
            LOG.warning("Chart failed (%s): %s", method.__name__, e)
            return _empty(f"Erreur: {e}")

    def _emit(self, fig):
        html = _div(fig, include_js=self._first)
        self._first = False
        return html

    # ─────────────────────────────────────────────────────────────────────────
    #  CREDIT CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    def _credit_traj(self, traj_key, y_label, fmt=".2%", to_millions=False, tight_y=False):
        """Credit line chart using charts_data trajectories (historical + projected)."""
        cr = self.record.get("module_results", {}).get("credit", {})
        if not cr: return _empty("Pas de données crédit")
        # charts_data is the same across levels — grab from baseline
        bm = cr.get("baseline") or cr.get("adverse") or cr.get("severe")
        if not bm: return _empty("Pas de données crédit")
        pt = bm.get("charts_data", {}).get(traj_key, {})
        if not pt: return _empty("Pas de trajectoires")

        # Determine where projection starts
        meta = bm.get("metadata", {})
        horizon = meta.get("horizon", [])
        proj_start = int(min(horizon)) if horizon else None

        fig = go.Figure()
        all_vals = []
        for lvl in ("baseline", "adverse", "severe"):
            d = pt.get(lvl, {})
            if not d.get("x"): continue
            # Use integers so Plotly uses a linear axis — avoids category compression
            try:
                x = [int(v) for v in d["x"]]
            except (ValueError, TypeError):
                x = list(d["x"])
            vals = _safe(d["y"])
            if to_millions:
                vals = [v / 1e6 if v else 0 for v in vals]
            all_vals.extend([v for v in vals if v is not None])
            col = C3[lvl]
            dash = "solid" if lvl == "severe" else ("dash" if lvl == "adverse" else "dot")
            fig.add_trace(go.Scatter(
                x=x, y=vals, name=LBL[lvl],
                mode="lines+markers",
                line=dict(color=col, width=2.5, dash=dash),
                marker=dict(size=5),
                fill=None if tight_y else ("tozeroy" if lvl == "severe" else None),
                fillcolor=_rgba(col, 0.04) if (not tight_y and lvl == "severe") else None,
            ))

        # Vertical separator historical / projection
        if proj_start:
            fig.add_shape(
                type="line", x0=proj_start - 0.5, x1=proj_start - 0.5,
                y0=0, y1=1, yref="paper",
                line=dict(color=P["grey"], width=1.5, dash="dash"),
            )
            fig.add_annotation(
                x=proj_start, y=1.04, yref="paper",
                text="Projection →", showarrow=False,
                font=dict(size=9, color=P["grey"]),
            )

        fig.update_layout(**LAY)
        fig.update_xaxes(tickformat="d", dtick=2, tickangle=-45)

        y_axis_kwargs = dict(title_text=y_label, tickformat=fmt if not to_millions else None)
        if tight_y and all_vals:
            margin = (max(all_vals) - min(all_vals)) * 0.25 or max(all_vals) * 0.05
            y_axis_kwargs["range"] = [min(all_vals) - margin, max(all_vals) + margin]

        fig.update_yaxes(**y_axis_kwargs)
        return self._emit(fig)

    def _credit_pd(self):
        return self._credit_traj("pd_trajectories", "PD", fmt=".2%")

    def _credit_lgd(self):
        return self._credit_traj("lgd_trajectories", "LGD", fmt=".2%", tight_y=True)

    def _credit_ecl(self):
        # ECL values from wrapper are already in millions (ead_M is in millions)
        # to_millions=False avoids double-dividing which caused Plotly µ-prefix display
        return self._credit_traj("el_trajectories",
                                 f"ECL ({self._currency_millions()})",
                                 fmt=",.2f",
                                 to_millions=False)

    def _credit_macro_contrib(self):
        """Bar chart — real ΔPD contribution (pp) per macro variable, baseline → severe."""
        cr = self.record.get("module_results", {}).get("credit", {})
        sev = cr.get("severe")
        if not sev:
            return _empty("Pas de données")
        meta = sev.get("metadata", {})

        contributions = meta.get("pd_contributions", {})
        amplitude = meta.get("amplitude_pp", 0)
        crisis = meta.get("crisis_name", "")
        sat_amp = meta.get("satellite_amplitude_pp", None)

        if not contributions:
            if sat_amp is not None and sat_amp < 1e-4:
                return _empty(
                    "Modèle satellite insensible aux chocs — "
                    "le plancher réglementaire a généré l'amplitude ΔPD "
                    "(aucune transmission macro mesurable pour ce scénario)"
                )
            return _empty("Contributions macro non disponibles")

        # Sort by absolute contribution descending
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        labels, values, colors = [], [], []
        for var, dpd_pp in sorted_contrib:
            lbl = var.replace("_", " ").title()
            if len(lbl) > 24:
                lbl = lbl[:22] + "…"
            labels.append(lbl)
            values.append(round(dpd_pp, 2))
            # Orange = worsens PD (positive contribution), blue = improves, grey = neutral
            if abs(dpd_pp) < 0.01:
                colors.append(P.get("grey", "#aaaaaa"))
            elif dpd_pp > 0:
                colors.append(P["orange1"])
            else:
                colors.append(P["blue1"])

        if not labels:
            return _empty("Contributions macro non disponibles")

        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:+.2f} pp" for v in values],
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig.update_layout(**{**LAY, "bargap": 0.4})
        fig.update_yaxes(title_text="Contribution au ΔPD (pp)", gridcolor=P["bg2"])
        fig.update_xaxes(tickangle=-25)

        header = f"{crisis + ' | ' if crisis else ''}ΔPD total sévère : {amplitude:.1f} pp"
        fig.add_annotation(
            text=header,
            xref="paper", yref="paper", x=0.5, y=1.06,
            showarrow=False, font=dict(size=10, color=P["grey"]),
        )
        return self._emit(fig)

    def _credit_concentration(self):
        """Histogram — PD distribution across the 3 scenario levels."""
        cr = self.record.get("module_results", {}).get("credit", {})
        if not cr: return _empty("Pas de données")
        fig = go.Figure()
        for lvl, col in C3.items():
            r = cr.get(lvl)
            if not r: continue
            ts = _ts_to_lists(r.get("time_series", {}))
            if not ts.get("pd"): continue
            fig.add_trace(go.Histogram(
                x=_safe(ts["pd"]),
                name=LBL[lvl], marker_color=_rgba(col, 0.65),
                nbinsx=10, opacity=0.75,
            ))
        fig.update_layout(**{**LAY, "barmode": "overlay", "bargap": 0.08})
        fig.update_xaxes(title_text="PD", tickformat=".1%")
        fig.update_yaxes(title_text="Fréquence")
        return self._emit(fig)

    @staticmethod
    def _parse_cap_traj(traj: dict, proj_start):
        """Return (hist_x, hist_y, {lvl: (x, y)}) handling both old and new formats.

        New format: traj has a 'historical' key + scenario keys with stress-only data.
        Old format: each scenario key contains hist+stress concatenated — split at proj_start.
        """
        def _clean(xs, ys):
            try:
                xi = [int(v) for v in xs]
            except (ValueError, TypeError):
                xi = list(xs)
            yi = [v if v is not None and not (isinstance(v, float) and np.isnan(v))
                  else None for v in ys]
            return xi, yi

        hist_x, hist_y = [], []
        scenarios: dict = {}

        if "historical" in traj:
            # New format — historical already separated
            h = traj.get("historical", {})
            if h.get("x"):
                hist_x, hist_y = _clean(h["x"], h["y"])
            for lvl in ("baseline", "adverse", "severe"):
                d = traj.get(lvl, {})
                if d.get("x"):
                    scenarios[lvl] = _clean(d["x"], d["y"])
        else:
            # Old format — split each scenario at proj_start to extract historical
            hist_done = False
            for lvl in ("baseline", "adverse", "severe"):
                d = traj.get(lvl, {})
                if not d.get("x"):
                    continue
                xi, yi = _clean(d["x"], d["y"])
                if proj_start:
                    split = next((i for i, v in enumerate(xi) if v >= proj_start), len(xi))
                    if not hist_done and split > 0:
                        hist_x, hist_y = xi[:split], yi[:split]
                        hist_done = True
                    # Scenario starts one point before proj_start for visual connection
                    sc_start = max(0, split - 1) if split > 0 else 0
                    scenarios[lvl] = (xi[sc_start:], yi[sc_start:])
                else:
                    scenarios[lvl] = (xi, yi)

        return hist_x, hist_y, scenarios

    def _credit_capital_ratio(self):
        """Line chart — CAR / CET1 ratio evolution (historical + stress)."""
        cr = self.record.get("module_results", {}).get("credit", {})
        if not cr: return _empty("Pas de données")
        bm = cr.get("baseline") or cr.get("adverse") or cr.get("severe")
        if not bm: return _empty("Pas de données")

        ct = bm.get("charts_data", {}).get("capital_trajectories", {})
        car_traj = ct.get("car", {})
        if not car_traj: return _empty("Données capital non disponibles")

        has_data = False
        for lvl_data in car_traj.values():
            if any(v is not None and not (isinstance(v, float) and np.isnan(v))
                   for v in lvl_data.get("y", [])):
                has_data = True
                break
        if not has_data:
            return _empty("Fichier capital non uploadé — CAR non disponible")

        meta = bm.get("metadata", {})
        min_car = meta.get("min_car", 0.1275)
        horizon = meta.get("horizon", [])
        proj_start = int(min(horizon)) if horizon else None

        hist_x, hist_y, scenarios = self._parse_cap_traj(car_traj, proj_start)

        fig = go.Figure()
        if hist_x:
            fig.add_trace(go.Scatter(
                x=hist_x, y=hist_y, name="Historique",
                mode="lines+markers",
                line=dict(color=P["grey"], width=2.5, dash="solid"),
                marker=dict(size=5),
            ))
        for lvl in ("baseline", "adverse", "severe"):
            if lvl not in scenarios: continue
            x, vals = scenarios[lvl]
            if all(v is None for v in vals): continue
            col = C3[lvl]
            dash = "solid" if lvl == "severe" else ("dash" if lvl == "adverse" else "dot")
            fig.add_trace(go.Scatter(
                x=x, y=vals, name=LBL[lvl],
                mode="lines+markers",
                line=dict(color=col, width=2.5, dash=dash),
                marker=dict(size=5),
            ))

        fig.add_hline(
            y=min_car, line_width=2, line_dash="dash", line_color="red",
            annotation_text=f"Min CAR {min_car*100:.1f}%",
            annotation_position="bottom right",
            annotation_font_size=9, annotation_font_color="red",
        )
        if proj_start:
            fig.add_shape(type="line", x0=proj_start - 0.5, x1=proj_start - 0.5,
                          y0=0, y1=1, yref="paper",
                          line=dict(color=P["grey"], width=1.5, dash="dash"))

        fig.update_layout(**LAY)
        fig.update_xaxes(tickformat="d", dtick=1, tickangle=-45)
        fig.update_yaxes(title_text="CAR (%)", tickformat=".1%")
        return self._emit(fig)

    def _credit_leverage(self):
        """Line chart — Leverage ratio evolution (historical + stress)."""
        cr = self.record.get("module_results", {}).get("credit", {})
        if not cr: return _empty("Pas de données")
        bm = cr.get("baseline") or cr.get("adverse") or cr.get("severe")
        if not bm: return _empty("Pas de données")

        ct = bm.get("charts_data", {}).get("capital_trajectories", {})
        lev_traj = ct.get("leverage_ratio", {})
        if not lev_traj: return _empty("Données levier non disponibles")

        has_data = False
        for lvl_data in lev_traj.values():
            if any(v is not None and not (isinstance(v, float) and np.isnan(v))
                   for v in lvl_data.get("y", [])):
                has_data = True
                break
        if not has_data:
            return _empty("Fichier capital non uploadé — Levier non disponible")

        meta = bm.get("metadata", {})
        min_lev = meta.get("min_leverage", 0.03)
        horizon = meta.get("horizon", [])
        proj_start = int(min(horizon)) if horizon else None

        hist_x, hist_y, scenarios = self._parse_cap_traj(lev_traj, proj_start)

        fig = go.Figure()
        if hist_x:
            fig.add_trace(go.Scatter(
                x=hist_x, y=hist_y, name="Historique",
                mode="lines+markers",
                line=dict(color=P["grey"], width=2.5, dash="solid"),
                marker=dict(size=5),
            ))
        for lvl in ("baseline", "adverse", "severe"):
            if lvl not in scenarios: continue
            x, vals = scenarios[lvl]
            if all(v is None for v in vals): continue
            col = C3[lvl]
            dash = "solid" if lvl == "severe" else ("dash" if lvl == "adverse" else "dot")
            fig.add_trace(go.Scatter(
                x=x, y=vals, name=LBL[lvl],
                mode="lines+markers",
                line=dict(color=col, width=2.5, dash=dash),
                marker=dict(size=5),
            ))

        fig.add_hline(
            y=min_lev, line_width=2, line_dash="dash", line_color="red",
            annotation_text=f"Min Levier {min_lev*100:.0f}%",
            annotation_position="bottom right",
            annotation_font_size=9, annotation_font_color="red",
        )

        if proj_start:
            fig.add_shape(type="line", x0=proj_start - 0.5, x1=proj_start - 0.5,
                          y0=0, y1=1, yref="paper",
                          line=dict(color=P["grey"], width=1.5, dash="dash"))

        fig.update_layout(**LAY)
        fig.update_xaxes(tickformat="d", dtick=1, tickangle=-45)
        fig.update_yaxes(title_text="Ratio Levier (%)", tickformat=".1%")
        return self._emit(fig)

    def _credit_cet1_ratio(self):
        """Line chart — CET1 ratio trajectory (historical + stress scenarios)."""
        cr = self.record.get("module_results", {}).get("credit", {})
        if not cr: return _empty("Pas de données")
        bm = cr.get("baseline") or cr.get("adverse") or cr.get("severe")
        if not bm: return _empty("Pas de données")

        ct = bm.get("charts_data", {}).get("capital_trajectories", {})
        traj = ct.get("cet1_ratio", {})

        has_data = bool(traj) and any(
            v is not None and not (isinstance(v, float) and np.isnan(v))
            for lvl_data in traj.values()
            for v in lvl_data.get("y", [])
        )
        if not has_data:
            return _empty(
                "CET1 non calculable — ajoutez une colonne 'cet1' "
                "ou 'at1' (CET1 = Tier1 − AT1) dans votre fichier capital"
            )

        meta = bm.get("metadata", {})
        min_cet1 = meta.get("min_cet1", 0.045)
        horizon = meta.get("horizon", [])
        proj_start = int(min(horizon)) if horizon else None

        hist_x, hist_y, scenarios = self._parse_cap_traj(traj, proj_start)

        fig = go.Figure()
        if hist_x:
            fig.add_trace(go.Scatter(
                x=hist_x, y=hist_y, name="Historique",
                mode="lines+markers",
                line=dict(color=P["grey"], width=2.5, dash="solid"),
                marker=dict(size=5),
            ))
        for lvl in ("baseline", "adverse", "severe"):
            if lvl not in scenarios: continue
            x, vals = scenarios[lvl]
            if all(v is None for v in vals): continue
            col = C3[lvl]
            dash = "solid" if lvl == "severe" else ("dash" if lvl == "adverse" else "dot")
            fig.add_trace(go.Scatter(
                x=x, y=vals, name=LBL[lvl],
                mode="lines+markers",
                line=dict(color=col, width=2.5, dash=dash),
                marker=dict(size=5),
            ))

        fig.add_hline(
            y=min_cet1, line_width=2, line_dash="dash", line_color="red",
            annotation_text=f"Min CET1 {min_cet1*100:.1f}%",
            annotation_position="bottom right",
            annotation_font_size=9, annotation_font_color="red",
        )
        if proj_start:
            fig.add_shape(type="line", x0=proj_start - 0.5, x1=proj_start - 0.5,
                          y0=0, y1=1, yref="paper",
                          line=dict(color=P["grey"], width=1.5, dash="dash"))

        fig.update_layout(**LAY)
        fig.update_xaxes(tickformat="d", dtick=1, tickangle=-45)
        fig.update_yaxes(title_text="CET1 Ratio (%)", tickformat=".1%")
        return self._emit(fig)

    def _credit_tier1_ratio(self):
        """Line chart — Tier 1 ratio trajectory (historical + stress scenarios)."""
        cr = self.record.get("module_results", {}).get("credit", {})
        if not cr: return _empty("Pas de données")
        bm = cr.get("baseline") or cr.get("adverse") or cr.get("severe")
        if not bm: return _empty("Pas de données")

        ct = bm.get("charts_data", {}).get("capital_trajectories", {})
        traj = ct.get("tier1_ratio", {})
        if not traj: return _empty("Données Tier 1 non disponibles")

        has_data = False
        for lvl_data in traj.values():
            if any(v is not None and not (isinstance(v, float) and np.isnan(v))
                   for v in lvl_data.get("y", [])):
                has_data = True
                break
        if not has_data:
            return _empty("Fichier capital non uploadé — Tier 1 non disponible")

        meta = bm.get("metadata", {})
        min_tier1 = meta.get("min_tier1", 0.060)
        horizon = meta.get("horizon", [])
        proj_start = int(min(horizon)) if horizon else None

        hist_x, hist_y, scenarios = self._parse_cap_traj(traj, proj_start)

        fig = go.Figure()
        if hist_x:
            fig.add_trace(go.Scatter(
                x=hist_x, y=hist_y, name="Historique",
                mode="lines+markers",
                line=dict(color=P["grey"], width=2.5, dash="solid"),
                marker=dict(size=5),
            ))
        for lvl in ("baseline", "adverse", "severe"):
            if lvl not in scenarios: continue
            x, vals = scenarios[lvl]
            if all(v is None for v in vals): continue
            col = C3[lvl]
            dash = "solid" if lvl == "severe" else ("dash" if lvl == "adverse" else "dot")
            fig.add_trace(go.Scatter(
                x=x, y=vals, name=LBL[lvl],
                mode="lines+markers",
                line=dict(color=col, width=2.5, dash=dash),
                marker=dict(size=5),
            ))

        fig.add_hline(
            y=min_tier1, line_width=2, line_dash="dash", line_color="red",
            annotation_text=f"Min Tier 1 {min_tier1*100:.1f}%",
            annotation_position="bottom right",
            annotation_font_size=9, annotation_font_color="red",
        )
        if proj_start:
            fig.add_shape(type="line", x0=proj_start - 0.5, x1=proj_start - 0.5,
                          y0=0, y1=1, yref="paper",
                          line=dict(color=P["grey"], width=1.5, dash="dash"))

        fig.update_layout(**LAY)
        fig.update_xaxes(tickformat="d", dtick=1, tickangle=-45)
        fig.update_yaxes(title_text="Tier 1 Ratio (%)", tickformat=".1%")
        return self._emit(fig)

    def _credit_waterfall(self):
        """Waterfall — CET1 decomposition: EL credit losses + RWA increase (ASRF)."""
        cr = self.record.get("module_results", {}).get("credit", {})
        b_r, s_r = cr.get("baseline"), cr.get("severe")
        if not b_r or not s_r: return _empty("Pas de données")

        meta = b_r.get("metadata", {})
        min_cet1 = meta.get("min_cet1", 0.045)
        ct = b_r.get("charts_data", {}).get("capital_trajectories", {})

        def _last_valid(vals):
            for v in reversed(vals or []):
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    return float(v)
            return None

        # CET1 initial = last historical observed value
        cet1_start = _last_valid(
            ct.get("cet1_ratio", {}).get("historical", {}).get("y", []))
        if cet1_start is None:
            return _empty(
                "CET1 non calculable — ajoutez une colonne 'cet1' "
                "ou 'at1' (CET1 = Tier1 − AT1) dans votre fichier capital"
            )
        cet1_start *= 100  # decimal → percentage

        # CET1 final = last projected value in severe scenario
        ct_sev = s_r.get("charts_data", {}).get("capital_trajectories", {})
        cet1_end = _last_valid(
            ct_sev.get("cet1_ratio", {}).get("severe", {}).get("y", []))
        if cet1_end is None:
            return _empty("Projection CET1 sévère non disponible")
        cet1_end *= 100  # decimal → percentage

        total_impact = cet1_start - cet1_end  # pp, positive = worsening

        # ── EL vs RWA decomposition (data-driven) ───────────────────────────
        # EL impact : ΔEL brut (sévère − baseline) / RWA_baseline_end, sans PPNR ni NI
        # RWA impact : résidu
        delta_el_sev = ct_sev.get("delta_el", {}).get("severe", {}).get("y", [])
        rwa_bl_last = _last_valid(
            ct.get("rwa_stressed", {}).get("baseline", {}).get("y", []))

        if delta_el_sev and rwa_bl_last and rwa_bl_last > 0:
            total_delta_el = sum(
                v for v in delta_el_sev
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            )
            el_impact = -(total_delta_el / rwa_bl_last * 100)
        else:
            el_impact = 0.0

        rwa_impact = -(total_impact + el_impact)  # rounding-safe residual

        el_text = f"{el_impact:+.2f} pp"

        fig = go.Figure(go.Waterfall(
            x=["CET1 Initial",
               "Impact pertes de crédit (ΔEL)",
               "Impact augmentation RWA (ASRF)",
               "CET1 Final"],
            y=[cet1_start, el_impact, rwa_impact, 0],
            measure=["absolute", "relative", "relative", "total"],
            text=[f"{cet1_start:.2f}%", el_text,
                  f"{rwa_impact:+.2f} pp", f"{cet1_end:.2f}%"],
            textposition="outside", textfont=dict(size=10),
            connector=dict(line=dict(color=P["border"], width=1)),
            increasing=dict(marker_color=P["green"]),
            decreasing=dict(marker_color=P["orange1"]),
            totals=dict(marker_color=P["dark"]),
        ))
        fig.add_hline(
            y=min_cet1 * 100, line_width=2, line_dash="dash", line_color="red",
            annotation_text=f"Seuil CET1 {min_cet1*100:.1f}%",
            annotation_position="bottom right",
            annotation_font_size=9, annotation_font_color="red",
        )
        fig.update_layout(**{**LAY, "bargap": 0.35})
        fig.update_yaxes(title_text="CET1 Ratio (%)")
        return self._emit(fig)

    def _credit_model_table(self):
        """HTML ranking table of satellite models — coefficients + p-values."""
        cr  = self.record.get("module_results", {}).get("credit", {})
        bm  = cr.get("baseline") or cr.get("severe") or cr.get("adverse")
        if not bm:
            return "<p style='color:#6B7280;font-size:12px'>Pas de données crédit</p>"
        rows = bm.get("charts_data", {}).get("model_table", [])
        if not rows:
            return "<p style='color:#6B7280;font-size:12px'>Tableau de modèles non disponible</p>"

        def _fmt_coefs(coefs: dict, pv: dict) -> str:
            parts = []
            for var, val in coefs.items():
                pval = pv.get(var)
                stars = ""
                if pval is not None:
                    if pval < 0.01:   stars = "***"
                    elif pval < 0.05: stars = "**"
                    elif pval < 0.10: stars = "*"
                pv_str = f" <span style='color:#9CA3AF;font-size:9px'>(p={pval:.3f})</span>" if pval is not None else ""
                parts.append(
                    f"<span style='font-weight:600'>{var}</span>: "
                    f"{val:+.4f}{stars}{pv_str}"
                )
            return "<br>".join(parts)

        def _fmt_intercept(val, pval) -> str:
            if val is None:
                return "—"
            pv_str = ""
            if pval is not None:
                stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
                pv_str = f"<br><span style='color:#9CA3AF;font-size:9px'>(p={pval:.3f}){stars}</span>"
            return f"{val:+.4f}{pv_str}"

        FAMILY_COLOR = {"logit": "#3B82F6", "vasicek": "#8B5CF6", "beta": "#10B981"}
        FAMILY_LABEL = {"logit": "Logit OLS", "vasicek": "Vasicek OLS", "beta": "Beta MLE"}

        html = """
<div style="overflow-x:auto;margin-top:4px">
<table style="width:100%;border-collapse:collapse;font-size:11.5px;font-family:inherit">
<thead>
<tr style="background:#F9FAFB;border-bottom:2px solid #E5E7EB">
  <th style="padding:8px 10px;text-align:center;white-space:nowrap">Rang</th>
  <th style="padding:8px 10px;text-align:left;white-space:nowrap">Famille</th>
  <th style="padding:8px 10px;text-align:left">Variables</th>
  <th style="padding:8px 10px;text-align:right;white-space:nowrap">Intercept</th>
  <th style="padding:8px 10px;text-align:left">Coefficients & p-values</th>
  <th style="padding:8px 10px;text-align:right;white-space:nowrap">R²</th>
  <th style="padding:8px 10px;text-align:right;white-space:nowrap">AIC</th>
  <th style="padding:8px 10px;text-align:center;white-space:nowrap">Signe OK</th>
</tr>
</thead>
<tbody>
"""
        for row in rows:
            rang      = row.get("rang", "")
            is_best   = bool(row.get("is_selected", rang == 1))
            family    = row.get("modele", "")
            variables = row.get("variables", [])
            intercept = row.get("intercept")
            pv_int    = row.get("pv_intercept")
            coefs     = row.get("coefs", {})
            pv_coefs  = row.get("pv_coefs", {})
            r2        = row.get("r2", 0)
            aic       = row.get("aic", 0)
            sign_ok   = row.get("sign_ok", True)
            viol      = row.get("violating_vars", [])

            bg       = "#FFF7ED" if is_best else ("#fff" if rang % 2 else "#F9FAFB")
            star     = "★ " if is_best else ""
            fcolor   = FAMILY_COLOR.get(family, "#374151")
            flabel   = FAMILY_LABEL.get(family, family)
            vars_str = " + ".join(variables) if isinstance(variables, list) else str(variables)
            sign_cell = (
                "<span style='color:#10B981;font-weight:700'>✓</span>"
                if sign_ok else
                f"<span style='color:#EF4444;font-weight:700' title='{', '.join(viol)}'>✗ ({len(viol)})</span>"
            )

            fitted_vals  = row.get("fitted_values", [])
            actual_vals  = row.get("actual_values", [])
            obs_years    = row.get("obs_years", [])
            has_insample = is_best and bool(obs_years) and bool(actual_vals)

            sat_id = f"credit-rank-{rang}"
            rank_cell = (
                f"<span style='cursor:pointer' onclick=\"toggleSatDetail('{sat_id}')\" title='Voir Réel vs Estimé'>"
                f"{star}{rang} <span id='arrow-{sat_id}' style='color:#9CA3AF;font-size:10px'>▶</span></span>"
                if has_insample else f"{star}{rang}"
            )

            html += f"""
<tr style="background:{bg};border-bottom:1px solid #E5E7EB">
  <td style="padding:8px 10px;text-align:center;font-weight:700;color:{'#FD5108' if is_best else '#6B7280'}">{rank_cell}</td>
  <td style="padding:8px 10px">
    <span style="background:{fcolor}22;color:{fcolor};border-radius:4px;
                 padding:2px 7px;font-weight:700;font-size:10.5px">{flabel}</span>
  </td>
  <td style="padding:8px 10px;color:#374151;font-size:10.5px">{vars_str}</td>
  <td style="padding:8px 10px;text-align:right;font-family:monospace;font-size:10.5px">{_fmt_intercept(intercept, pv_int)}</td>
  <td style="padding:8px 10px;font-family:monospace;font-size:10.5px;line-height:1.6">{_fmt_coefs(coefs, pv_coefs)}</td>
  <td style="padding:8px 10px;text-align:right;font-weight:700;color:{'#10B981' if r2 >= 0.7 else ('#D97706' if r2 >= 0.4 else '#EF4444')}">{r2:.4f}</td>
  <td style="padding:8px 10px;text-align:right;color:#6B7280">{aic:.1f}</td>
  <td style="padding:8px 10px;text-align:center">{sign_cell}</td>
</tr>
"""
            if has_insample:
                detail_rows = ""
                for i, yr in enumerate(obs_years):
                    a   = actual_vals[i] if i < len(actual_vals) else 0.0
                    f   = fitted_vals[i] if i < len(fitted_vals) else 0.0
                    res = a - f
                    rp  = (res / a * 100) if a != 0 else 0.0
                    zebra  = "white" if i % 2 == 0 else "#F9FAFB"
                    resclr = "#DC2626" if res < 0 else "#059669"
                    rpclr  = "#DC2626" if abs(rp) > 10 else ("#D97706" if abs(rp) > 5 else "#059669")
                    sign   = "+" if res >= 0 else ""
                    signp  = "+" if rp >= 0 else ""
                    detail_rows += (
                        f"<tr style='background:{zebra}'>"
                        f"<td style='padding:4px 10px;text-align:center;border:1px solid #E5E7EB;font-weight:600;color:#6B7280'>{yr}</td>"
                        f"<td style='padding:4px 10px;text-align:right;border:1px solid #E5E7EB;font-family:monospace'>{a:.4f}</td>"
                        f"<td style='padding:4px 10px;text-align:right;border:1px solid #E5E7EB;font-family:monospace;color:#1D4ED8'>{f:.4f}</td>"
                        f"<td style='padding:4px 10px;text-align:right;border:1px solid #E5E7EB;font-family:monospace;color:{resclr}'>{sign}{res:.4f}</td>"
                        f"<td style='padding:4px 10px;text-align:right;border:1px solid #E5E7EB;font-family:monospace;color:{rpclr}'>{signp}{rp:.1f}%</td>"
                        f"</tr>"
                    )
                html += f"""
<tr id="detail-{sat_id}" style="display:none;background:#F8FAFC">
  <td colspan="8" style="padding:12px 16px;border-top:1px dashed #E5E7EB">
    <div style="font-size:11px;font-weight:700;color:#374151;margin-bottom:8px">
      Valeurs In-Sample — Modèle Rang {rang} ({flabel} · {vars_str})
      <span style="font-weight:400;color:#9CA3AF;margin-left:8px">
        ({len(obs_years)} observations · cliquer à nouveau pour fermer)
      </span>
    </div>
    <div style="overflow-x:auto">
      <table style="border-collapse:collapse;font-size:11px;min-width:500px">
        <thead>
          <tr style="background:#EFF6FF;color:#1D4ED8;text-transform:uppercase;font-size:9.5px;letter-spacing:.04em">
            <th style="padding:5px 10px;text-align:center;border:1px solid #DBEAFE">Année</th>
            <th style="padding:5px 10px;text-align:right;border:1px solid #DBEAFE">PD Réelle</th>
            <th style="padding:5px 10px;text-align:right;border:1px solid #DBEAFE">PD Estimée</th>
            <th style="padding:5px 10px;text-align:right;border:1px solid #DBEAFE">Résidu</th>
            <th style="padding:5px 10px;text-align:right;border:1px solid #DBEAFE">Résidu %</th>
          </tr>
        </thead>
        <tbody>{detail_rows}</tbody>
      </table>
    </div>
  </td>
</tr>
"""
        html += """
</tbody>
</table>
<div style="margin-top:6px;font-size:10px;color:#9CA3AF">
  *** p&lt;0.01 &nbsp;** p&lt;0.05 &nbsp;* p&lt;0.10 &nbsp;·&nbsp;
  ★ Modèle sélectionné — cliquer sur le rang ★1 pour voir Réel vs Estimé
</div>
</div>
"""
        return html

    # ─────────────────────────────────────────────────────────────────────────
    #  CLIMATE CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    def _climate_impact_pd(self):
        """Dual axis — historical PD + scenario projections / Stress Index."""
        clim = self.record.get("module_results", {}).get("climate", {})
        bm = clim.get("baseline")
        if not bm: return _empty("Pas de données climatiques")
        pt = bm.get("charts_data", {}).get("pd_trajectories", {})
        if not pt: return _empty("Pas de trajectoires PD")
        is_ct = bm.get("kpis", {}).get("ngfs_mode", "LT") == "CT"
        _proj = _clip_proj if is_ct else (lambda d: d)
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Historical PD series (grey, behind projections)
        hist_d = _load_historical_pd(pt)
        if hist_d.get("x"):
            fig.add_trace(go.Scatter(
                x=[int(v) for v in hist_d["x"]],
                y=_safe(hist_d["y"]),
                name="PD Historique",
                mode="lines+markers",
                line=dict(color=P["grey"], width=2, dash="dot"),
                marker=dict(size=4),
            ), secondary_y=False)

        # Scenario projections (clipped to 2028 for CT only)
        for lvl in ("baseline", "adverse", "severe"):
            d = _proj(pt.get(lvl, {}))
            if not d.get("x"): continue
            fig.add_trace(go.Scatter(
                x=[int(v) for v in d["x"]],
                y=_safe(d["y"]),
                name=f"PD {LBL[lvl]}",
                mode="lines+markers",
                line=dict(color=C3[lvl], width=2.5),
                marker=dict(size=5),
            ), secondary_y=False)

        # Stress Index bars on secondary y (clipped for CT only)
        base_d = _proj(pt.get("baseline", {}))
        sev_d  = _proj(pt.get("severe", {}))
        if base_d.get("y") and sev_d.get("y"):
            n = min(len(base_d["y"]), len(sev_d["y"]))
            stress_idx = [round((s - b) / max(b, 0.001) * 100, 1)
                          for b, s in zip(base_d["y"][:n], sev_d["y"][:n])]
            fig.add_trace(go.Bar(
                x=[int(v) for v in sev_d["x"][:n]],
                y=stress_idx,
                name="Stress Index (%)",
                marker_color=_rgba(P["orange1"], 0.18),
                marker_line=dict(color=P["orange1"], width=1),
            ), secondary_y=True)

        fig.update_yaxes(title_text="PD", tickformat=".2%",
                         gridcolor=P["bg2"], secondary_y=False)
        fig.update_yaxes(title_text="Stress Index (%)",
                         gridcolor=P["bg2"], secondary_y=True,
                         showgrid=False)
        fig.update_layout(**{
            **LAY,
            "margin": dict(t=18, r=20, b=72, l=56),
            "xaxis": dict(gridcolor=P["bg2"], linecolor=P["border"],
                          zeroline=False, tickangle=45, tickformat="d", dtick=5),
            "legend": dict(orientation="h", y=-0.38, font=dict(size=9)),
        })
        return self._emit(fig)

    def _climate_ngfs_comparison(self):
        """Full-width — Historical PD + Baseline / Adverse / Severe projections."""
        clim = self.record.get("module_results", {}).get("climate", {})
        bm = clim.get("baseline")
        if not bm: return _empty("Pas de données")
        pt = bm.get("charts_data", {}).get("pd_trajectories", {})
        is_ct = bm.get("kpis", {}).get("ngfs_mode", "LT") == "CT"
        _proj = _clip_proj if is_ct else (lambda d: d)
        fig = go.Figure()

        # Historical PD series (grey dashed, before projections)
        hist_d = _load_historical_pd(pt)
        proj_start = None
        if hist_d.get("x"):
            fig.add_trace(go.Scatter(
                x=[int(v) for v in hist_d["x"]],
                y=_safe(hist_d["y"]),
                name="PD Historique",
                mode="lines+markers",
                line=dict(color=P["grey"], width=2.5, dash="dot"),
                marker=dict(size=5),
            ))
            proj_start = int(hist_d["x"][-1])

        # Scenario projections with fill between adverse and severe
        fills = {"baseline": None, "adverse": None, "severe": "tonexty"}
        for lvl in ("baseline", "adverse", "severe"):
            d = _proj(pt.get(lvl, {}))
            if not d.get("x"): continue
            fig.add_trace(go.Scatter(
                x=[int(v) for v in d["x"]],
                y=_safe(d["y"]),
                name=f"PD {LBL[lvl]}",
                mode="lines+markers",
                line=dict(color=C3[lvl], width=3 if lvl != "baseline" else 2),
                marker=dict(size=6 if lvl != "baseline" else 4),
                fill=fills[lvl],
                fillcolor=_rgba(C3["severe"], 0.07) if lvl == "severe" else None,
            ))

        # Vertical separator line at last historical year
        if proj_start is not None:
            fig.add_shape(
                type="line",
                x0=proj_start, x1=proj_start,
                y0=0, y1=1, yref="paper",
                line=dict(color=P["border"], width=1.5, dash="dash"),
            )
            fig.add_annotation(
                x=proj_start, y=1, yref="paper",
                text="◀ Hist.  Proj. ▶", showarrow=False,
                font=dict(size=10, color=P["grey"]),
                xanchor="center", yanchor="bottom",
            )

        fig.update_layout(
            margin=dict(t=32, r=160, b=80, l=64),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif",
                      size=12, color=P["dark"]),
            xaxis=dict(gridcolor=P["bg2"], linecolor=P["border"],
                       zeroline=False, tickangle=45, tickfont=dict(size=11),
                       tickformat="d", dtick=5),
            yaxis=dict(gridcolor=P["bg2"], linecolor=P["border"],
                       zeroline=False, tickformat=".2%",
                       title=dict(text="Probabilité de Défaut (PD)", font=dict(size=12))),
            legend=dict(orientation="v", x=1.01, y=0.98,
                        xanchor="left", yanchor="top",
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor=P["border"], borderwidth=1,
                        font=dict(size=11)),
            hovermode="x unified",
        )
        return self._emit(fig)

    def _climate_waterfall(self):
        """Waterfall — decompose PD increase from climate stress."""
        clim = self.record.get("module_results", {}).get("climate", {})
        b_r, s_r = clim.get("baseline"), clim.get("severe")
        if not b_r or not s_r: return _empty("Pas de données")
        pd_b = b_r.get("kpis", {}).get("avg_pd", 0) or 0
        pd_s = s_r.get("kpis", {}).get("avg_pd", 0) or 0
        delta = pd_s - pd_b
        # Decompose into transition + physical (60/40 split as default)
        trans_share = 0.6
        fig = go.Figure(go.Waterfall(
            x=["PD Baseline", "Risque Transition", "Risque Physique", "PD Sévère"],
            y=[pd_b * 100, delta * trans_share * 100,
               delta * (1 - trans_share) * 100, 0],
            measure=["absolute", "relative", "relative", "total"],
            text=[f"{pd_b*100:.2f}%",
                  f"+{delta*trans_share*100:.2f} pp",
                  f"+{delta*(1-trans_share)*100:.2f} pp",
                  f"{pd_s*100:.2f}%"],
            textposition="outside", textfont=dict(size=10),
            connector=dict(line=dict(color=P["border"], width=1)),
            increasing=dict(marker_color=P["orange1"]),
            decreasing=dict(marker_color=P["green"]),
            totals=dict(marker_color=P["red"]),
        ))
        fig.update_layout(**{**LAY, "bargap": 0.35})
        fig.update_yaxes(title_text="PD (%)")
        return self._emit(fig)

    def _climate_transition_physical(self):
        """Stacked area — Transition vs Physical risk over time."""
        clim = self.record.get("module_results", {}).get("climate", {})
        bm = clim.get("baseline")
        if not bm: return _empty("Pas de données")
        pt = bm.get("charts_data", {}).get("pd_trajectories", {})
        base_d = pt.get("baseline", {})
        sev_d  = pt.get("severe", {})
        if not base_d.get("y") or not sev_d.get("y"):
            return _empty("Trajectoires insuffisantes")
        x = [int(v) for v in sev_d["x"]]
        deltas = [max(s - b, 0) for b, s in zip(base_d["y"], sev_d["y"])]
        trans  = [d * 0.6 for d in deltas]
        phys   = [d * 0.4 for d in deltas]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=[v * 100 for v in trans], name="Risque de Transition",
            mode="lines", fill="tozeroy",
            line=dict(color=P["orange2"], width=0.5),
            fillcolor=_rgba(P["orange2"], 0.45),
        ))
        fig.add_trace(go.Scatter(
            x=x, y=[v * 100 for v in [t + p for t, p in zip(trans, phys)]],
            name="Risque Physique",
            mode="lines", fill="tonexty",
            line=dict(color=P["accent2"], width=0.5),
            fillcolor=_rgba(P["accent2"], 0.35),
        ))
        fig.update_layout(**{**LAY,
            "xaxis": dict(gridcolor=P["bg2"], linecolor=P["border"],
                          zeroline=False, tickformat="d", dtick=5)})
        fig.update_yaxes(title_text="ΔPD vs Baseline (pp)")
        return self._emit(fig)

    # ─────────────────────────────────────────────────────────────────────────
    #  MARKET CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    def _market_charts(self):
        self._first = True
        return {k: self._safe_build(m) for k, m in [
            ("yield_curve_comparison", self._market_yield_curve_comparison),
            ("yield_curve_shift",      self._market_yield_curve_shift),
            ("beta_history",           self._market_beta_history),
            ("loss_distribution",      self._market_loss_distribution),
            ("instrument_waterfall",   self._market_instrument_waterfall),
            ("stress_window_chart",    self._market_stress_window),
            ("ns_fit_quality",         self._market_ns_fit_quality),
            ("svar_chart",             self._market_svar_chart),
            ("ses_chart",              self._market_ses_chart),
        ]}

    def _market_beta_history(self):
        """Line chart — NS β1/β2/β3 historical time series + T+12 stress projections."""
        mr = self.record.get("module_results", {}).get("market", {})
        bm = mr.get("baseline") or mr.get("adverse") or mr.get("severe")
        if not bm:
            return _empty("Pas de données marché")
        ts_raw = bm.get("time_series", {})
        if not ts_raw or not isinstance(ts_raw, dict):
            return _empty("Pas de série temporelle betas")

        # Trim history to the last ~10 years (from 2016) so the T -> T+12
        # stress projection occupies a visible fraction of the chart instead
        # of a thin sliver squeezed against 15+ years of history.
        _HIST_MIN_DATE = "2016-01-01"
        dates = sorted(d for d in ts_raw.keys() if d >= _HIST_MIN_DATE)
        if not dates:
            dates = sorted(ts_raw.keys())
        b1 = _safe([ts_raw[d].get("beta1") for d in dates])
        b2 = _safe([ts_raw[d].get("beta2") for d in dates])
        b3 = _safe([ts_raw[d].get("beta3") for d in dates])

        try:
            proj_date = str(
                (pd.Timestamp(dates[-1]) + pd.DateOffset(months=12)).date()
            )
        except Exception:
            proj_date = None

        fig = go.Figure()

        # ── Historical series ──────────────────────────────────────────────
        for vals, name, col, dash in [
            (b1, "β₁ Niveau",   P["dark"],    "solid"),
            (b2, "β₂ Pente",    P["orange1"], "dash"),
            (b3, "β₃ Courbure", P["grey"],    "dot"),
        ]:
            fig.add_trace(go.Scatter(
                x=dates, y=vals, name=name,
                mode="lines",
                line=dict(color=col, width=2, dash=dash),
            ))

        # ── Separator + shading — use add_shape (works with string-date axes) ──
        if proj_date:
            fig.add_shape(
                type="line",
                x0=dates[-1], x1=dates[-1], y0=0, y1=1, yref="paper",
                line=dict(dash="dot", color=P["border"], width=1.5),
            )
            fig.add_shape(
                type="rect",
                x0=dates[-1], x1=proj_date, y0=0, y1=1, yref="paper",
                fillcolor=_rgba(P["bg1"], 0.55), line_width=0,
                layer="below",
            )
            fig.add_annotation(
                x=dates[-1], y=0.97, xref="x", yref="paper",
                text="T", showarrow=False,
                font=dict(size=9, color=P["grey"]),
                xanchor="right",
            )
            fig.add_annotation(
                x=proj_date, y=0.97, xref="x", yref="paper",
                text="T+12", showarrow=False,
                font=dict(size=9, color=P["grey"]),
                xanchor="left",
            )

            # ── T+12 extensions — full monthly trajectory per scenario ──────
            # Visual encoding:
            #   colour = scenario  (grey=baseline, orange=adverse, red=severe)
            #   dash   = beta      (solid=β₁, dash=β₂, dot=β₃)
            # This matches the historical lines so the reader can track each
            # beta factor across the historical→projection boundary.
            last_b1 = b1[-1] if b1 else None
            last_b2 = b2[-1] if b2 else None
            last_b3 = b3[-1] if b3 else None

            # (beta_key, last_observed_value, fallback_stressed_kpi, dash_style)
            BETA_META = [
                ("beta1", last_b1, "solid"),
                ("beta2", last_b2, "dash"),
                ("beta3", last_b3, "dot"),
            ]
            SCEN_COLS = {
                "baseline": (P["grey"],    "Baseline T+12"),
                "adverse":  (P["orange2"], "Adverse T+12"),
                "severe":   (P["orange1"], "Sévère T+12"),
            }
            first_scen = {lvl: True for lvl in SCEN_COLS}

            for bk, last_val, beta_dash in BETA_META:
                if last_val is None:
                    continue
                for lvl, (scen_col, scen_lbl) in SCEN_COLS.items():
                    r = mr.get(lvl)
                    if not isinstance(r, dict):
                        continue
                    cd_scen      = r.get("charts_data", {})
                    monthly      = cd_scen.get("beta_monthly_path", {})
                    stressed_val = r.get("kpis", {}).get(f"{bk}_stressed")
                    if stressed_val is None:
                        continue

                    if monthly and monthly.get("dates") and bk in monthly:
                        x_proj = [dates[-1]] + monthly["dates"]
                        y_proj = [last_val] + monthly[bk]
                    else:
                        x_proj = [dates[-1], proj_date]
                        y_proj = [last_val, float(stressed_val)]

                    fig.add_trace(go.Scatter(
                        x=x_proj,
                        y=y_proj,
                        mode="lines+markers",
                        name=scen_lbl,
                        legendgroup=lvl,
                        showlegend=first_scen[lvl],
                        line=dict(color=scen_col, width=2.5, dash=beta_dash),
                        marker=dict(size=6, color=scen_col),
                    ))
                    first_scen[lvl] = False

        fig.update_layout(**{**LAY,
            "legend": dict(orientation="h", y=-0.28, font=dict(size=10)),
            "margin": dict(t=18, r=20, b=64, l=56),
        })
        fig.update_xaxes(tickangle=-45, nticks=12)
        # Give the T -> T+12 projection room to breathe instead of being
        # flush against the plot's right edge.
        if proj_date:
            try:
                _x_end = str((pd.Timestamp(proj_date) + pd.DateOffset(months=1)).date())
                fig.update_xaxes(range=[dates[0], _x_end])
            except Exception:
                pass
        fig.update_yaxes(title_text="Facteurs Nelson-Siegel (%)", tickformat=".2f")
        return self._emit(fig)

    def _market_yield_curve_shift(self):
        """Grouped bar chart — Δy(τ) per scenario per maturity."""
        mr  = self.record.get("module_results", {}).get("market", {})
        if not mr:
            return _empty("Pas de données marché")

        maturities = ["T91j", "T182j", "T273j", "T364j", "T3Y", "T5Y"]
        mat_labels  = ["91j", "182j", "273j", "364j", "3Y", "5Y"]

        fig = go.Figure()
        for lvl, col, lbl in [
            ("baseline", P["dark"],    "Baseline"),
            ("adverse",  P["orange2"], "Adverse"),
            ("severe",   P["orange1"], "Sévère"),
        ]:
            r = mr.get(lvl, {})
            dy = r.get("charts_data", {}).get("delta_y", {}) if isinstance(r, dict) else {}
            vals = [float(dy.get(m, 0) or 0) * 100 for m in maturities]  # pp → bp
            fig.add_trace(go.Bar(
                name=lbl, x=mat_labels, y=vals,
                marker_color=col,
                text=[f"{v:+.0f}" for v in vals],
                textposition="outside",
                textfont=dict(size=9),
            ))

        fig.add_hline(y=0, line_width=1, line_color=P["border"])
        fig.update_layout(**{**LAY, "barmode": "group", "bargap": 0.2})
        fig.update_yaxes(title_text="Variation du taux souverain (points de base)")
        fig.update_xaxes(title_text="Maturité")
        return self._emit(fig)

    def _market_svar_chart(self):
        """Bar chart — sVaR (FHS) vs scenario losses.

        Calcul de chaque barre :
          sVaR 99% FHS  : percentile 99% de 10 000 simulations bootstrap
                          depuis la fenêtre BCBS (pire 12 mois Σ Var(Δβ))
          Perte scénario: ΔP = Σ BPV_k × Δy(τ_k) × Notionnel_k
                          où Δy vient de la projection AR(1)+macro à T+12
        """
        mr = self.record.get("module_results", {}).get("market", {})
        bm = mr.get("baseline") or mr.get("adverse") or mr.get("severe")
        if not bm:
            return _empty("Pas de données marché")

        kpis     = bm.get("kpis", {}) if isinstance(bm, dict) else {}
        svar     = float(kpis.get("svar_99", 0) or 0)
        notional = float(kpis.get("total_notional_bn", 0) or 0) * 1000  # Md→M EGP

        # Collect |ΔP| per scenario
        losses = {}
        for lvl in ("baseline", "adverse", "severe"):
            r = mr.get(lvl)
            if isinstance(r, dict):
                dp = float(r.get("kpis", {}).get("delta_p_m_egp", 0) or 0)
                losses[lvl] = abs(dp)

        labels = ["sVaR 99% (FHS)", "Perte Baseline", "Perte Adverse", "Perte Sévère"]
        values = [
            svar,
            losses.get("baseline", 0),
            losses.get("adverse",  0),
            losses.get("severe",   0),
        ]
        colors = [
            _rgba(P["red"],     0.85),
            _rgba(P["dark"],    0.80),
            _rgba(P["orange2"], 0.85),
            _rgba(P["orange1"], 0.85),
        ]

        # Y-axis capped at displayed values (notional shown as annotation only)
        y_max = max((v for v in values if v), default=1)
        y_ceil = y_max * 1.35

        fig = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=[f"{v:,.0f}" for v in values],
            textposition="outside",
            textfont=dict(size=11, color=P["dark"]),
            width=0.55,
        ))

        # Notionnel as annotation (not hline — would blow up the y-axis)
        if notional > 0:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.99, y=0.99,
                text=f"Notionnel total : {notional:,.0f} M EGP",
                showarrow=False,
                font=dict(size=9, color=P["grey"]),
                xanchor="right", yanchor="top",
                bgcolor=P["bg2"], bordercolor=P["border"], borderpad=3,
            )

        fig.update_layout(**{**LAY, "bargap": 0.40,
                             "margin": dict(t=18, r=20, b=44, l=70)})
        fig.update_yaxes(title_text="Perte (M EGP)", range=[0, y_ceil])
        return self._emit(fig)

    def _market_ses_chart(self):
        """Bar chart — tail distribution with sVaR/sES markers + ES/VaR ratio."""
        mr = self.record.get("module_results", {}).get("market", {})
        bm = mr.get("baseline") or mr.get("adverse") or mr.get("severe")
        if not bm:
            return _empty("Pas de données marché")

        kpis = bm.get("kpis", {}) if isinstance(bm, dict) else {}
        pct  = bm.get("charts_data", {}).get("loss_distribution_pct", {}) \
               if isinstance(bm, dict) else {}

        svar = float(kpis.get("svar_99",  0) or 0)
        ses  = float(kpis.get("ses_975",  0) or 0)

        # Tail percentiles only (p75 → p99)
        tail_ps   = [75, 90, 95, 97, 99]
        tail_vals = []
        tail_lbls = []
        for p in tail_ps:
            key = f"p{p}"
            if key in pct:
                tail_vals.append(float(pct[key] or 0))
                tail_lbls.append(f"p{p}%")

        if not tail_vals:
            return _empty("Distribution des pertes non disponible")

        # ES/VaR ratio annotation
        ratio_lbl = f"ES/VaR = {ses/svar:.2f}" if svar > 0 else ""

        fig = go.Figure(go.Bar(
            x=tail_lbls, y=tail_vals,
            marker_color=[
                P["orange1"] if v >= svar else
                (_rgba(P["orange2"], 0.7) if v >= ses else _rgba(P["dark"], 0.5))
                for v in tail_vals
            ],
            text=[f"{v:,.0f}" for v in tail_vals],
            textposition="outside",
            textfont=dict(size=9),
        ))

        for val, lbl, col in [
            (ses,  f"sES 97.5%",  P["orange2"]),
            (svar, f"sVaR 99%",   P["red"]),
        ]:
            if val > 0:
                fig.add_hline(
                    y=val, line_dash="dash", line_color=col, line_width=2,
                    annotation_text=lbl,
                    annotation_position="top right",
                    annotation_font_size=9,
                    annotation_font_color=col,
                )

        if ratio_lbl:
            fig.add_annotation(
                text=ratio_lbl, xref="paper", yref="paper",
                x=0.02, y=0.97, showarrow=False,
                font=dict(size=11, color=P["dark"]),
                bgcolor="white", bordercolor=P["border"], borderpad=4,
            )

        fig.update_layout(**{**LAY, "bargap": 0.3})
        fig.update_yaxes(title_text="Perte (M EGP)")
        fig.update_xaxes(title_text="Percentile")
        return self._emit(fig)

    def _market_yield_curve_comparison(self):
        """Time-series — toutes les maturités disponibles + légende Scénario/Maturité."""
        mr = self.record.get("module_results", {}).get("market", {})
        bm = mr.get("baseline") or mr.get("adverse") or mr.get("severe")
        if not bm:
            return _empty("Pas de données marché")

        cd  = bm.get("charts_data", {}) if isinstance(bm, dict) else {}
        obs = cd.get("yield_curve_observed", {})
        if not obs:
            return _empty("Courbe observée non disponible")

        last_date_str = cd.get("yield_curve_last_date", "")
        try:
            proj_date = str(
                (pd.Timestamp(last_date_str) + pd.DateOffset(months=12)).date()
            )
        except Exception:
            proj_date = None

        hist       = cd.get("yield_curve_history", {})
        hist_mats  = hist.get("maturities", []) if hist else []
        hist_dates = hist.get("dates", [])       if hist else []
        hist_vals  = hist.get("values", [])      if hist else []

        # Trim history to the last ~10 years (from 2016) so the T -> T+12
        # stress projection occupies a visible fraction of the chart instead
        # of a thin sliver squeezed against 15+ years of history.
        _HIST_MIN_DATE = "2016-01-01"
        if hist_dates and hist_vals:
            _keep = [i for i, d in enumerate(hist_dates) if d >= _HIST_MIN_DATE]
            if _keep:
                hist_dates = [hist_dates[i] for i in _keep]
                hist_vals  = [hist_vals[i]  for i in _keep]

        # Toutes les maturités connues, dans l'ordre croissant
        ALL_MATS_META = [
            ("T91j",  "91j",  "#1f77b4"),
            ("T182j", "182j", "#ff7f0e"),
            ("T273j", "273j", "#2ca02c"),
            ("T364j", "364j", "#d62728"),
            ("T3Y",   "3Y",   "#9467bd"),
            ("T5Y",   "5Y",   "#8c564b"),
            ("T10Y",  "10Y",  "#e377c2"),
        ]
        # Garder les maturités présentes dans l'historique ET dans obs.
        # Fallback : si pas d'historique, utiliser obs directement (projections seules).
        if hist_mats:
            SHOW_MATS = [
                (mat, lbl, col) for mat, lbl, col in ALL_MATS_META
                if mat in hist_mats and mat in obs
            ]
        else:
            SHOW_MATS = [
                (mat, lbl, col) for mat, lbl, col in ALL_MATS_META
                if mat in obs
            ]

        SCEN_PROJ = [
            ("baseline", "dot",      "Baseline"),
            ("adverse",  "dash",     "Adverse"),
            ("severe",   "longdash", "Sévère"),
        ]
        SCEN_COLORS = {
            "baseline": "#4a90d9",
            "adverse":  "#e07b2a",
            "severe":   "#c0392b",
        }

        fig = go.Figure()

        # ── 1. Courbes historiques (une par maturité) ─────────────────────
        for mat, lbl, col in SHOW_MATS:
            if mat not in hist_mats or not hist_dates:
                # Pas d'historique pour cette maturité — on saute (projections seules)
                continue
            idx = hist_mats.index(mat)
            y_h = _safe([row[idx] for row in hist_vals])
            if not any(v is not None for v in y_h):
                continue
            fig.add_trace(go.Scatter(
                x=hist_dates, y=y_h,
                name=lbl,
                legendgroup=mat,
                mode="lines",
                line=dict(color=col, width=2.2),
                showlegend=True,
                hovertemplate=f"<b>{lbl}</b><br>%{{x}}<br>%{{y:.2f}}%<extra></extra>",
            ))

        # ── 2. Séparateur T / zone projection ────────────────────────────
        if last_date_str and proj_date:
            fig.add_shape(
                type="line",
                x0=last_date_str, x1=last_date_str, y0=0, y1=1, yref="paper",
                line=dict(dash="dot", color=P["border"], width=1.5),
            )
            fig.add_shape(
                type="rect",
                x0=last_date_str, x1=proj_date, y0=0, y1=1, yref="paper",
                fillcolor=_rgba(P["bg1"], 0.45), line_width=0, layer="below",
            )
            fig.add_annotation(
                x=last_date_str, y=0.97, xref="x", yref="paper",
                text="T", showarrow=False,
                font=dict(size=9, color=P["grey"]), xanchor="right",
            )
            fig.add_annotation(
                x=proj_date, y=0.97, xref="x", yref="paper",
                text="T+12", showarrow=False,
                font=dict(size=9, color=P["grey"]), xanchor="left",
            )

        # ── 3. Projections T → T+12 (couleur = maturité, tiret = scénario) ─
        SCEN_DASH = {"baseline": "dot", "adverse": "dash", "severe": "longdash"}
        SCEN_LBL  = {"baseline": "Baseline", "adverse": "Adverse", "severe": "Sévère"}

        # Maturités sans historique → ajouter une trace fantôme pour la légende
        mats_with_hist = {mat for mat, _, _ in SHOW_MATS if mat in hist_mats and hist_dates}
        for mat, lbl, col in SHOW_MATS:
            if mat not in mats_with_hist:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="lines",
                    name=lbl, legendgroup=mat,
                    line=dict(color=col, width=2.2),
                    showlegend=True,
                ))

        for mat, lbl, mat_col in SHOW_MATS:
            if not last_date_str or mat not in obs:
                continue
            t_val = float(obs[mat])

            for lvl in ("baseline", "adverse", "severe"):
                r = mr.get(lvl)
                if not isinstance(r, dict):
                    continue
                cd_scen  = r.get("charts_data", {})
                scen_lbl = SCEN_LBL[lvl]
                scen_dash = SCEN_DASH[lvl]

                monthly = cd_scen.get("yield_curve_monthly_path", {})
                if monthly and mat in monthly and monthly.get("dates"):
                    _mdates = monthly["dates"]
                    _mvals  = [v for v in monthly[mat] if v is not None]
                    x_proj  = [last_date_str] + _mdates[:len(_mvals)]
                    y_proj  = [t_val] + _mvals
                else:
                    stressed = cd_scen.get("yield_curve_stressed", {})
                    if not stressed or mat not in stressed:
                        continue
                    x_proj = [last_date_str, proj_date]
                    y_proj = [t_val, float(stressed[mat])]

                # Première trace de projection : visible dans la légende si pas d'historique
                _show_in_legend = (mat not in mats_with_hist and lvl == "baseline")
                fig.add_trace(go.Scatter(
                    x=x_proj, y=y_proj,
                    mode="lines+markers",
                    name=f"{lbl} {scen_lbl}",
                    legendgroup=mat,
                    showlegend=_show_in_legend,
                    line=dict(color=mat_col, width=2.2, dash=scen_dash),
                    marker=dict(size=5, color=mat_col),
                    hovertemplate=(
                        f"<b>{lbl}</b> · {scen_lbl}<br>"
                        "%{x}<br><b>%{y:.2f}%</b><extra></extra>"
                    ),
                ))

        # ── 4. Entrées fantômes pour la légende scénarios (styles de tiret) ─
        for lvl in ("baseline", "adverse", "severe"):
            if mr.get(lvl):
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    name=SCEN_LBL[lvl],
                    legendgroup="__scen__",
                    line=dict(color=P["grey"], width=2, dash=SCEN_DASH[lvl]),
                    showlegend=True,
                ))

        # ── 5. Annotations légende manuelle en bas ───────────────────────
        fig.add_annotation(
            text=(
                "<b>Maturité :</b> couleur de ligne &nbsp;&nbsp;"
                "<b>Scénario :</b> "
                "<span style='color:#888'>···</span> Baseline &nbsp;"
                "<span style='color:#888'>- -</span> Adverse &nbsp;"
                "<span style='color:#888'>──</span> Sévère"
            ),
            xref="paper", yref="paper",
            x=0.0, y=-0.12,
            showarrow=False,
            font=dict(size=9, color=P["grey"]),
            align="left",
            xanchor="left",
        )

        fig.update_layout(**{**LAY,
            "height": 480,       # synchronisé avec chart-body height:480px dans le template
            "legend": dict(
                orientation="v",
                x=1.01, y=1.0,
                xanchor="left", yanchor="top",
                font=dict(size=10),
                itemsizing="constant",
                tracegroupgap=4,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
            ),
            "margin": dict(t=24, r=155, b=55, l=55),
        })
        fig.update_xaxes(
            title_text="Date",
            tickangle=-45,
            nticks=16,
            showgrid=True,
            gridcolor=P.get("bg2", "#e8e8e8"),
            linecolor=P.get("border", "#cccccc"),
            showline=True,
            ticks="outside",
            tickcolor=P.get("border", "#cccccc"),
        )
        # Give the T -> T+12 projection room to breathe instead of being
        # flush against the plot's right edge.
        if proj_date:
            try:
                _x_start = hist_dates[0] if hist_dates else last_date_str
                _x_end = str((pd.Timestamp(proj_date) + pd.DateOffset(months=1)).date())
                fig.update_xaxes(range=[_x_start, _x_end])
            except Exception:
                pass
        fig.update_yaxes(
            title_text="Taux (%)",
            showgrid=True,
            gridcolor=P.get("bg2", "#e8e8e8"),
            linecolor=P.get("border", "#cccccc"),
            showline=True,
            ticks="outside",
            tickcolor=P.get("border", "#cccccc"),
            ticksuffix="%",
        )
        return self._emit(fig)

    def _market_instrument_waterfall(self):
        """Horizontal bar chart — ΔP per instrument (severe scenario)."""
        mr = self.record.get("module_results", {}).get("market", {})
        sv = mr.get("severe") or mr.get("adverse") or mr.get("baseline")
        if not isinstance(sv, dict):
            return _empty("Pas de données sévère")

        breakdown = sv.get("charts_data", {}).get("instrument_breakdown", {})
        if not breakdown:
            return _empty("Décomposition par instrument non disponible")

        items = sorted(breakdown.items(), key=lambda x: x[1])
        labels = [k for k, _ in items]
        values = [round(float(v), 2) for _, v in items]
        total  = round(sum(values), 2)

        labels.append("TOTAL")
        values.append(total)

        colors = []
        for v in values[:-1]:
            colors.append(_rgba(P["red"], 0.75) if v < 0 else _rgba(P["green"], 0.75))
        colors.append(P["red"] if total < 0 else P["green"])

        fig = go.Figure(go.Bar(
            y=labels, x=values, orientation="h",
            marker_color=colors,
            text=[f"{v:+,.1f}" for v in values],
            textposition="outside",
            textfont=dict(size=10),
        ))
        fig.add_vline(x=0, line_width=1.5, line_color=P["border"])
        fig.update_layout(**{**LAY, "margin": dict(t=18, r=80, b=44, l=110), "bargap": 0.25})
        fig.update_xaxes(title_text="ΔP (M EGP)")
        return self._emit(fig)

    def _market_stress_window(self):
        """Line chart — rolling 12-month beta variance.

        Two annotated zones:
        ① BCBS calibration window (rouge) — pire 12 mois historiques → calibre sVaR/sES FHS
        ② Projection scénario (bleu) — T → T+12 → horizon du choc macro AR(1)
        """
        mr = self.record.get("module_results", {}).get("market", {})
        bm = mr.get("baseline") or mr.get("adverse") or mr.get("severe")
        if not isinstance(bm, dict):
            return _empty("Pas de données marché")

        cd  = bm.get("charts_data", {})
        vol = cd.get("stress_window_volatility", {})
        if not vol:
            return _empty("Volatilité fenêtre de stress non disponible")

        dates  = sorted(vol.keys())
        values = [float(vol[d]) for d in dates]
        y_max  = max(values) if values else 1

        sw_start = cd.get("stress_window_start", "")
        sw_end   = cd.get("stress_window_end",   "")

        # T and T+12 — scenario projection horizon
        t_date = cd.get("yield_curve_last_date", "")
        try:
            t12_date = str(
                (pd.Timestamp(t_date) + pd.DateOffset(months=12)).date()
            )
        except Exception:
            t12_date = ""

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            name="Σ Var(Δβ) — fenêtre 12 mois glissante",
            mode="lines",
            line=dict(color="#0C447C", width=2),
            fill="tozeroy",
            fillcolor=_rgba("#0C447C", 0.07),
        ))

        # ── Zone ① : BCBS calibration window (rouge) ─────────────────────
        if sw_start and sw_end and sw_start in dates and sw_end in dates:
            fig.add_shape(
                type="rect",
                x0=sw_start, x1=sw_end, y0=0, y1=1, yref="paper",
                fillcolor=_rgba(P["red"], 0.12), line_width=0, layer="below",
            )
            fig.add_shape(
                type="line",
                x0=sw_start, x1=sw_start, y0=0, y1=1, yref="paper",
                line=dict(color=P["red"], width=1, dash="dot"),
            )
            fig.add_shape(
                type="line",
                x0=sw_end, x1=sw_end, y0=0, y1=1, yref="paper",
                line=dict(color=P["red"], width=1, dash="dot"),
            )
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.99, y=0.99,
                text=(
                    f"<b>① Calibration FHS (BCBS)</b><br>"
                    f"{sw_start} → {sw_end}<br>"
                    f"<i>Pire fenêtre historique → bootstrap sVaR/sES</i>"
                ),
                showarrow=False,
                font=dict(size=9, color=P["red"]),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=P["red"],
                borderpad=4,
                xanchor="right", yanchor="top",
                align="right",
            )

        # ── Zone ② : Scenario projection horizon T → T+12 (bleu) ─────────
        if t_date and t12_date:
            fig.add_shape(
                type="rect",
                x0=t_date, x1=t12_date, y0=0, y1=1, yref="paper",
                fillcolor=_rgba("#0C447C", 0.10), line_width=0, layer="below",
            )
            fig.add_shape(
                type="line",
                x0=t_date, x1=t_date, y0=0, y1=1, yref="paper",
                line=dict(color="#0C447C", width=1.5, dash="dash"),
            )
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.99, y=0.60,
                text=(
                    f"<b>② Stress scénario</b>  "
                    f"T={t_date}  T+12={t12_date}<br>"
                    f"<i>Projection AR(1)+macro → ΔP portefeuille</i>"
                ),
                showarrow=False,
                font=dict(size=9, color="#0C447C"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#0C447C",
                borderpad=4,
                xanchor="right", yanchor="top",
                align="right",
            )

        fig.update_layout(**{**LAY,
            "legend": dict(orientation="h", y=-0.22, font=dict(size=10)),
        })
        fig.update_xaxes(nticks=12, tickangle=-45)
        fig.update_yaxes(title_text="Σ Var(Δβₖ)")
        return self._emit(fig)

    def _market_ns_fit_quality(self):
        """Scatter (observed) + smooth line (NS fitted) at last date."""
        mr = self.record.get("module_results", {}).get("market", {})
        bm = mr.get("baseline") or mr.get("adverse") or mr.get("severe")
        if not isinstance(bm, dict):
            return _empty("Pas de données marché")

        cd       = bm.get("charts_data", {})
        kpi      = bm.get("kpis", {}) if isinstance(bm, dict) else {}
        observed = cd.get("ns_fit_observed", {})
        curve    = cd.get("ns_fit_curve",    {})

        if not observed and not curve:
            return _empty("Données NS fitting non disponibles")

        MAT_YEARS = {
            "T91j": 91/365, "T182j": 182/365, "T273j": 273/365, "T364j": 364/365,
            "T3Y": 3.0, "T5Y": 5.0, "T10Y": 10.0,
        }
        fig = go.Figure()

        if curve and curve.get("x") and curve.get("y"):
            fig.add_trace(go.Scatter(
                x=curve["x"], y=curve["y"],
                name="Courbe NS fittée",
                mode="lines",
                line=dict(color="#0C447C", width=2.5),
            ))

        if observed:
            obs_sorted = sorted(observed, key=lambda m: MAT_YEARS.get(m, 0))
            obs_x = [MAT_YEARS[m] for m in obs_sorted]
            obs_y = [float(observed[m]) for m in obs_sorted]

            # Compute in-sample MSE between NS curve and observed points
            if curve and curve.get("x") and curve.get("y"):
                import bisect
                cx, cy = curve["x"], curve["y"]
                residuals = []
                for ox, oy in zip(obs_x, obs_y):
                    idx = min(bisect.bisect_left(cx, ox), len(cx) - 1)
                    residuals.append((oy - cy[idx]) ** 2)
                rmse = float(np.sqrt(np.mean(residuals))) if residuals else float("nan")
            else:
                rmse = float("nan")

            fig.add_trace(go.Scatter(
                x=obs_x, y=obs_y,
                name="Taux observés",
                mode="markers",
                marker=dict(color="#FD5108", size=10, symbol="circle-open",
                            line=dict(width=2)),
            ))

            # Annotation: λ optimal + RMSE
            ns_lam = kpi.get("ns_lambda")
            ns_mse = kpi.get("ns_lambda_mse")
            ann_parts = []
            if ns_lam is not None and not (isinstance(ns_lam, float) and np.isnan(ns_lam)):
                ann_parts.append(f"λ optimal = {ns_lam:.4f} an⁻¹  (τ* ≈ {1/ns_lam:.2f} an)")
            if not np.isnan(rmse):
                ann_parts.append(f"RMSE = {rmse:.4f} %  (dernière date)")
            if ns_mse is not None and not (isinstance(ns_mse, float) and np.isnan(ns_mse)):
                ann_parts.append(f"MSE moyen (hist.) = {ns_mse:.4f} %²")
            if ann_parts:
                fig.add_annotation(
                    xref="paper", yref="paper", x=0.99, y=0.99,
                    text="<br>".join(ann_parts),
                    showarrow=False,
                    align="right",
                    font=dict(size=10, color=P["dark"]),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor=P["border"],
                    borderpad=4,
                    xanchor="right", yanchor="top",
                )

        fig.update_layout(**{**LAY, "legend": dict(orientation="h", y=-0.22, font=dict(size=10))})
        fig.update_xaxes(title_text="Maturité (années)")
        fig.update_yaxes(title_text="Taux (%)")
        return self._emit(fig)

    def _market_loss_distribution(self):
        """Histogram of FHS loss distribution with sVaR/sES markers."""
        mr = self.record.get("module_results", {}).get("market", {})
        bm = mr.get("baseline") or mr.get("adverse") or mr.get("severe")
        if not bm:
            return _empty("Pas de données marché")

        kpis = bm.get("kpis", {}) if isinstance(bm, dict) else {}
        pct  = bm.get("charts_data", {}).get("loss_distribution_pct", {}) \
               if isinstance(bm, dict) else {}

        svar = float(kpis.get("svar_99",  0) or 0)
        ses  = float(kpis.get("ses_975",  0) or 0)

        if not pct:
            return _empty("Distribution des pertes non disponible")

        # Rebuild approximate histogram from percentiles p1 p5 p25 p50 p75 p95 p99
        ps   = [1, 5, 25, 50, 75, 95, 99]
        vals = [float(pct.get(f"p{p}", 0) or 0) for p in ps]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"p{p}" for p in ps], y=vals,
            name="Distribution FHS",
            marker_color=_rgba(P["orange1"], 0.6),
            marker_line=dict(color=P["orange1"], width=1),
        ))

        for val, lbl, col in [
            (svar, f"sVaR 99%  {svar:.1f}", P["red"]),
            (ses,  f"sES 97.5% {ses:.1f}",  P["orange1"]),
        ]:
            if val > 0:
                fig.add_hline(
                    y=val, line_dash="dash", line_color=col, line_width=2,
                )
                fig.add_annotation(
                    xref="paper", x=0.01,
                    y=val,
                    text=f"<b>{lbl}</b>",
                    showarrow=False,
                    font=dict(size=9, color="white"),
                    bgcolor=col,
                    borderpad=3,
                    xanchor="left",
                    yanchor="bottom",
                )

        fig.update_layout(**LAY)
        fig.update_yaxes(title_text="Perte (M EGP)")
        fig.update_xaxes(title_text="Percentile")
        return self._emit(fig)

    # ─────────────────────────────────────────────────────────────────────────
    #  LIQUIDITY CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    def _liquidity_lcr_nsfr(self):
        """LCR and NSFR trajectories — historical series + stressed projections."""
        liq = self.record.get("module_results", {}).get("liquidity", {})
        first_level = next(iter(liq.values()), {}) if liq else {}
        charts_data = first_level.get("charts_data", {})
        lcr_traj  = charts_data.get("lcr_trajectories",  {})
        nsfr_traj = charts_data.get("nsfr_trajectories", {})
        if not lcr_traj and not nsfr_traj:
            return _empty("Pas de trajectoires LCR/NSFR")

        # Regulatory minima (BCBS 238 / BCBS 295)
        LCR_MIN  = 100.0
        NSFR_MIN = 100.0

        fig = make_subplots(rows=1, cols=2,
            subplot_titles=("LCR (%)", "NSFR (%)"),
            shared_yaxes=False)

        # Dernier point historique → sert de pont vers les séries projetées
        hist_lcr_last_x  = lcr_traj.get("historical",  {}).get("x",  [None])[-1]
        hist_lcr_last_y  = lcr_traj.get("historical",  {}).get("y",  [None])[-1]
        hist_nsfr_last_x = nsfr_traj.get("historical", {}).get("x",  [None])[-1]
        hist_nsfr_last_y = nsfr_traj.get("historical", {}).get("y",  [None])[-1]
        bridge = {
            "lcr":  (hist_lcr_last_x,  hist_lcr_last_y),
            "nsfr": (hist_nsfr_last_x, hist_nsfr_last_y),
        }

        # Draw order: historical first, then scenario projections
        DRAW_ORDER = ["historical", "baseline", "adverse", "severe"]

        for col_i, (traj, reg_min, tkey) in enumerate(
            [(lcr_traj, LCR_MIN, "lcr"), (nsfr_traj, NSFR_MIN, "nsfr")], start=1
        ):
            bx, by = bridge[tkey]

            for level in DRAW_ORDER:
                d = traj.get(level)
                if not d:
                    continue
                is_hist = (level == "historical")
                if is_hist:
                    color  = P["dark"]
                    dash   = "dot"
                    width  = 2
                    symbol = "square"
                    label  = "Historique (bilan réel)"
                    x_vals = d.get("x", [])
                    y_vals = d.get("y", [])
                else:
                    color  = C3.get(level, P["grey"])
                    dash   = "solid"
                    width  = 2
                    symbol = "circle"
                    label  = LBL.get(level, level)
                    # Préfixer avec le dernier point historique pour relier les courbes
                    x_vals = ([bx] + d.get("x", [])) if bx is not None else d.get("x", [])
                    y_vals = ([by] + d.get("y", [])) if by is not None else d.get("y", [])

                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    name=label if col_i == 1 else None,
                    showlegend=(col_i == 1),
                    mode="lines+markers",
                    line=dict(color=color, width=width, dash=dash),
                    marker=dict(size=5, symbol=symbol),
                ), row=1, col=col_i)

            # Regulatory floor line
            fig.add_hline(
                y=reg_min,
                line_dash="dash",
                line_color=P["red"],
                line_width=1.5,
                annotation_text=f"Min. réglementaire {int(reg_min)}%",
                annotation_position="bottom right",
                annotation_font_size=9,
                annotation_font_color=P["red"],
                row=1, col=col_i,
            )

        # Compute separate y-axis ranges per subplot (LCR and NSFR have very different scales)
        def _axis_range(traj_dict, floor=85.0, headroom=1.10):
            vals = [v for d in traj_dict.values()
                    for v in d.get("y", [])
                    if v is not None and isinstance(v, (int, float))
                    and v == v and abs(v) != float("inf")]
            if not vals:
                return [floor, 200.0]
            lo = min(min(vals) * 0.90, floor)
            hi = max(vals) * headroom
            return [max(0, lo), hi]

        lcr_range  = _axis_range(lcr_traj,  floor=85.0)
        # NSFR is defined in [80%, ~200%] — tighter range than LCR
        nsfr_range = _axis_range(nsfr_traj, floor=85.0, headroom=1.08)

        fig.update_layout(**{**LAY, "legend": dict(orientation="h", y=-0.18)})
        fig.update_yaxes(title_text="Ratio (%)", ticksuffix="%",
                         range=lcr_range,  row=1, col=1)
        fig.update_yaxes(title_text="Ratio (%)", ticksuffix="%",
                         range=nsfr_range, row=1, col=2)
        fig.update_xaxes(tickformat="d", tickangle=-45)
        return self._emit(fig)

    def _liquidity_ratios_chart(self):
        """Horizontal bar chart — LCR & NSFR baseline vs sévère."""
        liq = self.record.get("module_results", {}).get("liquidity", {})
        bm = liq.get("baseline") or liq.get("adverse") or liq.get("severe")
        sv = liq.get("severe") or liq.get("adverse") or liq.get("baseline")
        if not bm:
            return _empty("Pas de données")

        bm_kpis = bm.get("kpis", {}) if isinstance(bm, dict) else {}
        sv_kpis = sv.get("kpis", {}) if isinstance(sv, dict) else {}

        def _finite(v, default=0.0):
            try:
                f = float(v or 0)
                return f if (f == f and abs(f) != float("inf")) else default
            except Exception:
                return default
        lcr_bl   = _finite(bm_kpis.get("lcr_baseline"))
        nsfr_bl  = _finite(bm_kpis.get("nsfr_baseline"))
        lcr_st   = _finite(sv_kpis.get("lcr_stressed"))
        nsfr_st  = _finite(sv_kpis.get("nsfr_stressed"))

        if not any([lcr_bl, nsfr_bl, lcr_st, nsfr_st]):
            return _empty("KPIs LCR/NSFR non disponibles")

        labels = ["LCR", "NSFR"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Baseline", y=labels,
            x=[lcr_bl, nsfr_bl], orientation="h",
            marker_color=P["dark"],
            texttemplate="%{x:.1f}%", textposition="inside",
            textfont=dict(color="white", size=10),
        ))
        fig.add_trace(go.Bar(
            name="Sévère", y=labels,
            x=[lcr_st, nsfr_st], orientation="h",
            marker_color=P["orange1"],
            texttemplate="%{x:.1f}%", textposition="inside",
            textfont=dict(color="white", size=10),
        ))
        # Regulatory floor at 100%
        fig.add_vline(x=100, line_dash="dash", line_color="red",
                      annotation_text="Seuil 100%",
                      annotation_position="top right",
                      annotation_font_size=9, annotation_font_color="red")
        fig.update_layout(**{**LAY, "barmode": "group", "bargap": 0.25,
                             "margin": dict(t=18, r=20, b=44, l=80)})
        fig.update_xaxes(title_text="Ratio (%)", ticksuffix="%")
        return self._emit(fig)

    # ─────────────────────────────────────────────────────────────────────────
    #  SHARED CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    def ratios_chart(self):
        ratios = self.cons.get("ratios", {})
        if not ratios: return _empty("Pas de données")
        labels = list(ratios.keys())
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Baseline", y=labels,
            x=[ratios[k]["baseline"] for k in labels], orientation="h",
            marker_color=P["dark"],
            texttemplate="%{x:.1f}%", textposition="inside",
            textfont=dict(color="white", size=10),
        ))
        fig.add_trace(go.Bar(
            name="Stressé", y=labels,
            x=[ratios[k]["stressed"] for k in labels], orientation="h",
            marker_color=P["orange1"],
            texttemplate="%{x:.1f}%", textposition="inside",
            textfont=dict(color="white", size=10),
        ))
        fig.update_layout(**{**LAY, "barmode": "group", "bargap": 0.25,
                             "margin": dict(t=18, r=20, b=44, l=110)})
        return self._emit(fig)

    # ─────────────────────────────────────────────────────────────────────────
    #  TRANSMISSION CHARTS (unchanged logic)
    # ─────────────────────────────────────────────────────────────────────────
    def _sankey(self):
        cons = self.cons.get("kpi_cards", {})
        el_sev = abs(cons.get("el_severe", 500e6) or 500e6) / 1e6
        el_adv = abs(cons.get("el_adverse", 300e6) or 300e6) / 1e6
        nodes = ["Choc Macro", "Crédit", "Liquidité", "Marché", "Capital", "Pertes Totales"]
        src = [0, 0, 1, 2, 3, 1, 2, 3]
        tgt = [1, 2, 3, 4, 4, 5, 5, 5]
        val = [el_sev*0.4, el_sev*0.3, el_sev*0.15, el_sev*0.1, el_sev*0.05,
               el_adv*0.4, el_adv*0.3, el_adv*0.1]
        fig = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=20, label=nodes,
                      color=[P["dark"], P["orange1"], P["accent3"],
                             P["accent1"], P["green"], P["grey"]]),
            link=dict(source=src, target=tgt,
                      value=[max(v, 1) for v in val],
                      color=[_rgba(P["orange1"], .3)]*3
                            + [_rgba(P["accent3"], .3)]*2
                            + [_rgba(P["orange1"], .2)]*3),
        ))
        fig.update_layout(**{**LAY, "margin": dict(t=20, r=20, b=20, l=20)})
        return self._emit(fig)

    def _network_matrix(self):
        import random; random.seed(12)
        risks = ["Crédit", "Climatique", "Marché", "Liquidité"]
        z = [[round(random.random()*0.8, 2) if i != j else 1.0
              for j in range(4)] for i in range(4)]
        for i in range(4):
            for j in range(i): z[i][j] = z[j][i]
        fig = go.Figure(go.Heatmap(
            z=z, x=risks, y=risks,
            colorscale=[[0, P["tint2"]], [0.5, P["orange2"]], [1, P["orange1"]]],
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}", textfont=dict(size=12), showscale=True,
        ))
        fig.update_layout(**{**LAY, "margin": dict(t=10, r=80, b=60, l=80)})
        return self._emit(fig)

    def _propagation_timeline(self):
        years = list(range(2024, 2031))
        phases = {
            "Choc initial":    [1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.0],
            "Impact Crédit":   [0.2, 0.7, 1.0, 0.9, 0.7, 0.5, 0.3],
            "Impact Liquidité": [0.0, 0.3, 0.7, 1.0, 0.8, 0.6, 0.4],
            "Impact Capital":  [0.0, 0.1, 0.4, 0.8, 1.0, 0.9, 0.7],
        }
        colors_map = [P["dark"], P["orange1"], P["accent3"], P["green"]]
        fig = go.Figure()
        for (name, vals), col in zip(phases.items(), colors_map):
            fig.add_trace(go.Scatter(
                x=years, y=vals, name=name, mode="lines+markers",
                line=dict(color=col, width=2.5), marker=dict(size=6),
            ))
        fig.update_layout(**LAY)
        fig.update_yaxes(title_text="Intensité")
        return self._emit(fig)
