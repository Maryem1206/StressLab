"""Scenario Manager.
Every scenario type automatically produces 3 severity levels:
Baseline, Adverse, Severe. The user never selects the level manually.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("scenario_manager")

# ── Pre-defined historical shocks ─────────────────────────────────────────────
# Each entry defines multipliers / additive shocks for Baseline / Adverse / Severe
HISTORICAL_SCENARIOS: Dict[str, Dict] = {
    "gulf_war": {
        "label": "Guerre du Golfe (1990–1991)",
        "icon": "⚔️",
        "description": "Choc pétrolier, récession régionale, hausse des taux",
        "crisis_name": "gulf_war_1990",
        "shocks": {
            "baseline": {"gdp_delta": 0.0,   "rate_bps": 0,   "oil_pct": 0.0,
                         "inflation_delta": 0.0, "unemployment_delta": 0.0, "fx_pct": 0.0},
            "adverse":  {"gdp_delta": -0.03,  "rate_bps": 150, "oil_pct": 0.40,
                         "inflation_delta": 0.02, "unemployment_delta": 0.02, "fx_pct": -0.10},
            "severe":   {"gdp_delta": -0.06,  "rate_bps": 300, "oil_pct": 0.80,
                         "inflation_delta": 0.05, "unemployment_delta": 0.04, "fx_pct": -0.20},
        },
    },
    "gfc_2008": {
        "label": "Crise Financière Mondiale (2008–2009)",
        "icon": "🏦",
        "description": "Effondrement bancaire, credit crunch, récession globale",
        "crisis_name": "gfc_2008",
        "shocks": {
            "baseline": {"gdp_delta": 0.0,   "rate_bps": 0,   "oil_pct": 0.0,
                         "inflation_delta": 0.0, "unemployment_delta": 0.0, "fx_pct": 0.0},
            "adverse":  {"gdp_delta": -0.04,  "rate_bps": -100,"oil_pct": -0.30,
                         "inflation_delta": -0.01,"unemployment_delta": 0.03, "fx_pct": -0.12},
            "severe":   {"gdp_delta": -0.09,  "rate_bps": -200,"oil_pct": -0.55,
                         "inflation_delta": -0.02,"unemployment_delta": 0.06, "fx_pct": -0.25},
        },
    },
    "covid": {
        "label": "Crise COVID-19 (2020)",
        "icon": "🦠",
        "description": "Arrêt économique brutal, choc d'offre et de demande simultané",
        "crisis_name": "covid_2020",
        "shocks": {
            "baseline": {"gdp_delta": 0.0,   "rate_bps": 0,   "oil_pct": 0.0,
                         "inflation_delta": 0.0, "unemployment_delta": 0.0, "fx_pct": 0.0},
            "adverse":  {"gdp_delta": -0.05,  "rate_bps": -75, "oil_pct": -0.35,
                         "inflation_delta": -0.01,"unemployment_delta": 0.04, "fx_pct": -0.08},
            "severe":   {"gdp_delta": -0.12,  "rate_bps": -150,"oil_pct": -0.60,
                         "inflation_delta": -0.03,"unemployment_delta": 0.09, "fx_pct": -0.18},
        },
    },
    "ukraine_energy": {
        "label": "Guerre Ukraine / Crise Énergie (2022)",
        "icon": "⚡",
        "description": "Choc énergétique, inflation extrême, sanctions",
        "crisis_name": "russia_ukraine_2022",
        "shocks": {
            "baseline": {"gdp_delta": 0.0,   "rate_bps": 0,   "oil_pct": 0.0,
                         "inflation_delta": 0.0, "unemployment_delta": 0.0, "fx_pct": 0.0},
            "adverse":  {"gdp_delta": -0.02,  "rate_bps": 250, "oil_pct": 0.50,
                         "inflation_delta": 0.04, "unemployment_delta": 0.01, "fx_pct": -0.08},
            "severe":   {"gdp_delta": -0.05,  "rate_bps": 450, "oil_pct": 1.00,
                         "inflation_delta": 0.08, "unemployment_delta": 0.03, "fx_pct": -0.18},
        },
    },
    "oil_shock": {
        "label": "Choc Pétrolier",
        "icon": "🛢️",
        "description": "Effondrement ou explosion des prix du pétrole",
        "crisis_name": None,
        "shocks": {
            "baseline": {"gdp_delta": 0.0,   "rate_bps": 0,   "oil_pct": 0.0,
                         "inflation_delta": 0.0, "unemployment_delta": 0.0, "fx_pct": 0.0},
            "adverse":  {"gdp_delta": -0.02,  "rate_bps": 100, "oil_pct": 0.60,
                         "inflation_delta": 0.03, "unemployment_delta": 0.01, "fx_pct": -0.06},
            "severe":   {"gdp_delta": -0.04,  "rate_bps": 200, "oil_pct": 1.20,
                         "inflation_delta": 0.06, "unemployment_delta": 0.02, "fx_pct": -0.15},
        },
    },
    "arab_spring": {
        "label": "Printemps Arabe — Révolution Égyptienne (2011)",
        "icon": "🌍",
        "description": "Instabilité politique, fuite de capitaux, choc touristique",
        "crisis_name": "arab_spring_2011",
        "shocks": {
            "baseline": {"gdp_delta": 0.0,   "rate_bps": 0,   "oil_pct": 0.0,
                         "inflation_delta": 0.0, "unemployment_delta": 0.0, "fx_pct": 0.0},
            "adverse":  {"gdp_delta": -0.02,  "rate_bps": 100, "oil_pct": 0.10,
                         "inflation_delta": 0.02, "unemployment_delta": 0.02, "fx_pct": -0.05},
            "severe":   {"gdp_delta": -0.04,  "rate_bps": 200, "oil_pct": 0.20,
                         "inflation_delta": 0.04, "unemployment_delta": 0.04, "fx_pct": -0.10},
        },
    },
    "taper_tantrum": {
        "label": "Taper Tantrum — Fuite de Capitaux EM (2013)",
        "icon": "💸",
        "description": "Remontée des taux US, fuite des capitaux marchés émergents",
        "crisis_name": "taper_tantrum_2013",
        "shocks": {
            "baseline": {"gdp_delta": 0.0,   "rate_bps": 0,   "oil_pct": 0.0,
                         "inflation_delta": 0.0, "unemployment_delta": 0.0, "fx_pct": 0.0},
            "adverse":  {"gdp_delta": -0.01,  "rate_bps": 200, "oil_pct": 0.05,
                         "inflation_delta": 0.01, "unemployment_delta": 0.01, "fx_pct": -0.10},
            "severe":   {"gdp_delta": -0.03,  "rate_bps": 350, "oil_pct": 0.10,
                         "inflation_delta": 0.02, "unemployment_delta": 0.02, "fx_pct": -0.20},
        },
    },
    "egypt_2016": {
        "label": "Dévaluation EGP & Réformes FMI (2016–2017)",
        "icon": "🇪🇬",
        "description": "Dévaluation brutale de la livre égyptienne, inflation record, taux CBE à 19%",
        "crisis_name": "egypt_2016",
        "shocks": {
            "baseline": {"gdp_delta": 0.0,   "rate_bps": 0,    "oil_pct": 0.0,
                         "inflation_delta": 0.0,  "unemployment_delta": 0.0, "fx_pct": 0.0},
            "adverse":  {"gdp_delta": -0.01,  "rate_bps": 300,  "oil_pct": 0.0,
                         "inflation_delta": 0.10,  "unemployment_delta": 0.01, "fx_pct": -0.25},
            "severe":   {"gdp_delta": -0.02,  "rate_bps": 600,  "oil_pct": 0.0,
                         "inflation_delta": 0.20,  "unemployment_delta": 0.02, "fx_pct": -0.50},
        },
    },
}

# ── NGFS Climate scenario families ────────────────────────────────────────────
NGFS_SCENARIOS: Dict[str, Dict] = {
    "net_zero": {
        "label": "Net Zero 2050",
        "icon": "🌿",
        "description": "Transition ordonnée vers la neutralité carbone",
        "ngfs_baseline": "Baseline",
        "ngfs_adverse":  "Net Zero 2050",
        "ngfs_severe":   "Below 2°C",
    },
    "delayed_transition": {
        "label": "Transition Retardée",
        "icon": "⏳",
        "description": "Politiques climatiques tardives, choc de transition",
        "ngfs_baseline": "Baseline",
        "ngfs_adverse":  "Delayed Transition",
        "ngfs_severe":   "Divergent Net Zero",
    },
    "hot_house": {
        "label": "Hot House World",
        "icon": "🔥",
        "description": "Aucune politique climatique — risque physique maximal",
        "ngfs_baseline": "Baseline",
        "ngfs_adverse":  "NDCs",
        "ngfs_severe":   "Current Policies",
    },
}

# ── Parametric scenario defaults ──────────────────────────────────────────────
PARAMETRIC_DEFAULTS = {
    "baseline": {"gdp_delta": 0.0, "rate_bps": 0,   "oil_pct": 0.0,
                 "inflation_delta": 0.0, "unemployment_delta": 0.0, "fx_pct": 0.0,
                 "carbon_price_delta": 0.0, "volatility_delta": 0.0},
    "adverse":  {"gdp_delta": -0.03,"rate_bps": 150, "oil_pct": 0.30,
                 "inflation_delta": 0.02,"unemployment_delta": 0.02, "fx_pct": -0.10,
                 "carbon_price_delta": 0.20,"volatility_delta": 0.15},
    "severe":   {"gdp_delta": -0.07,"rate_bps": 300, "oil_pct": 0.70,
                 "inflation_delta": 0.05,"unemployment_delta": 0.05, "fx_pct": -0.22,
                 "carbon_price_delta": 0.50,"volatility_delta": 0.35},
}


class ScenarioManager:
    """Manages all scenario types and auto-generates 3 severity levels."""

    def __init__(self, scenarios_folder: str):
        self.folder = Path(scenarios_folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────
    def get_scenario(self, scenario_type: str, scenario_id: str,
                     custom_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Return a scenario dict with all 3 levels: baseline / adverse / severe."""
        if scenario_type == "historical":
            return self._build_historical(scenario_id)
        elif scenario_type == "ngfs":
            return self._build_ngfs(scenario_id)
        elif scenario_type == "parametric":
            return self._build_parametric(custom_params or {})
        elif scenario_type == "idiosyncratic":
            cp = custom_params or {}
            # shock_inputs may be nested under "shock_inputs" key or flat at top level
            shock_inputs = cp.get("shock_inputs") or {
                k: cp.get(k, 0.0)
                for k in ("w", "npl_add_pp", "hqla_coef", "outflow_coef",
                          "asf_coef", "rsf_coef", "capital_coef", "rwa_coef")
            }
            return self._build_idiosyncratic(scenario_id, shock_inputs)
        elif scenario_type == "saved":
            return self._load_saved(scenario_id)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

    def save_scenario(self, name: str, scenario_type: str,
                      params: Dict) -> str:
        """Save a user-defined scenario to disk."""
        sid = name.lower().replace(" ", "_")
        data = {"name": name, "type": scenario_type, "params": params}
        (self.folder / f"{sid}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        LOG.info("Scenario saved: %s", sid)
        return sid

    def list_saved(self) -> List[Dict]:
        return [
            json.loads(p.read_text(encoding="utf-8"))
            for p in self.folder.glob("*.json")
        ]

    def list_historical(self) -> List[Dict]:
        return [{"id": k, **{k2: v2 for k2, v2 in v.items() if k2 != "shocks"}}
                for k, v in HISTORICAL_SCENARIOS.items()]

    def list_ngfs(self) -> List[Dict]:
        return [{"id": k, **{k2: v2 for k2, v2 in v.items() if k2 not in ("ngfs_baseline","ngfs_adverse","ngfs_severe")}}
                for k, v in NGFS_SCENARIOS.items()]

    # ── Internal builders ──────────────────────────────────────────────────────
    def _build_historical(self, sid: str) -> Dict:
        cfg = HISTORICAL_SCENARIOS.get(sid)
        if not cfg:
            raise ValueError(f"Unknown historical scenario: {sid}")
        return {
            "type":        "historical",
            "id":          sid,
            "name":        sid,
            "crisis_name": cfg.get("crisis_name"),   # clé CRISIS_CATALOG (None si n/a)
            "label":       cfg["label"],
            "icon":        cfg.get("icon", "📊"),
            "description": cfg.get("description", ""),
            "levels":      cfg["shocks"],             # baseline / adverse / severe
        }

    def _build_ngfs(self, sid: str) -> Dict:
        cfg = NGFS_SCENARIOS.get(sid)
        if not cfg:
            raise ValueError(f"Unknown NGFS scenario: {sid}")
        return {
            "type": "ngfs", "id": sid,
            "label": cfg["label"], "icon": cfg.get("icon", "🌍"),
            "description": cfg.get("description", ""),
            "levels": {
                "baseline": {"ngfs_scenario": cfg["ngfs_baseline"]},
                "adverse":  {"ngfs_scenario": cfg["ngfs_adverse"]},
                "severe":   {"ngfs_scenario": cfg["ngfs_severe"]},
            },
        }

    def _build_parametric(self, custom: Dict) -> Dict:
        levels = {}
        for lvl in ("baseline", "adverse", "severe"):
            base = dict(PARAMETRIC_DEFAULTS[lvl])
            base.update(custom.get(lvl, {}))
            levels[lvl] = base
        return {
            "type": "parametric", "id": "custom",
            "label": "Scénario Paramétrique",
            "icon": "🎛️",
            "description": "Scénario défini par l'utilisateur",
            "levels": levels,
        }

    def _build_idiosyncratic(self, sid: str, shock_inputs: Dict) -> Dict:
        """
        Builds 3 severity levels for an idiosyncratic event.

        severity scaling:
            baseline : all shocks × 0
            adverse  : all shocks × w × 0.5
            severe   : all shocks × w × 1.0

        shock_inputs keys:
            w             : intensity weight [0–1]
            npl_add_pp    : credit — NPL additive shock (percentage points)
            hqla_coef     : liquidity — HQLA reduction factor
            outflow_coef  : liquidity — outflow increase factor
            asf_coef      : liquidity — ASF reduction factor
            rsf_coef      : liquidity — RSF increase factor
            capital_coef  : capital — direct capital reduction factor
            rwa_coef      : capital — RWA inflation factor
        """
        try:
            from ..services.idio_events import IDIO_EVENTS
            event = IDIO_EVENTS.get(sid, {})
        except Exception:
            event = {}

        label = event.get("name", sid)
        icon  = event.get("icon", "⚡")
        desc  = event.get("description", "")

        w = float(shock_inputs.get("w", 1.0))

        _IDIO_KEYS = (
            "npl_add_pp", "hqla_coef", "outflow_coef",
            "asf_coef", "rsf_coef", "capital_coef", "rwa_coef",
        )

        def _level(scale: float) -> Dict:
            effective_w = w * scale
            return {k: float(shock_inputs.get(k, 0.0)) * effective_w for k in _IDIO_KEYS}

        LOG.info(
            "Idiosyncratic scenario '%s': w=%.2f, shocks=%s",
            sid, w,
            {k: shock_inputs.get(k, 0.0) for k in _IDIO_KEYS},
        )

        return {
            "type":        "idiosyncratic",
            "id":          sid,
            "label":       label,
            "icon":        icon,
            "description": desc,
            "levels": {
                "baseline": _level(0.0),
                "adverse":  _level(0.5),
                "severe":   _level(1.0),
            },
        }

    def _load_saved(self, sid: str) -> Dict:
        p = self.folder / f"{sid}.json"
        if not p.exists():
            raise ValueError(f"Saved scenario not found: {sid}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return self._build_parametric(data.get("params", {}))
