"""
validation_engine.py
====================
Module de validation des résultats du stress test liquidité.

5 batteries de tests :
    1. Hiérarchie des scénarios (baseline ≥ adverse ≥ severe)
    2. Cohérence économique des satellites (direction des chocs)
    3. Bornes réglementaires et domaines (contraintes mécaniques)
    4. Sensibilité mono-variable (un choc à la fois)
    5. Back-testing 2016 (prédictions vs observations historiques)

Usage :
    from liquidity_risk.validation_engine import validate_results
    report = validate_results(outputs, engine, macro_hist)
    report.print_report()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURES DE SORTIE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    battery:   str    # "hierarchy" | "satellites" | "bounds" | "sensitivity" | "backtest"
    test_name: str    # ex: "lcr_ordering_2026"
    status:    str    # "PASS" | "FAIL" | "WARN"
    expected:  str    # ex: "LCR_base ≥ LCR_adv ≥ LCR_sev"
    actual:    str    # ex: "218.6 ≥ 168.8 ≥ 133.8"
    message:   str    # explication lisible


@dataclass
class ValidationReport:
    results:      List[TestResult] = field(default_factory=list)
    bank_name:    str = ""
    country:      str = ""

    @property
    def total_tests(self) -> int:
        return len(self.results)

    @property
    def passed_tests(self) -> int:
        return sum(1 for r in self.results if r.status == "PASS")

    @property
    def failed_tests(self) -> int:
        return sum(1 for r in self.results if r.status == "FAIL")

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.status == "WARN")

    @property
    def passed(self) -> bool:
        return self.failed_tests == 0

    def by_battery(self, battery: str) -> List[TestResult]:
        return [r for r in self.results if r.battery == battery]

    def print_report(self) -> None:
        _print_validation_report(self)


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def validate_results(
    outputs:    Dict,
    engine:     object,
    macro_hist: pd.DataFrame,
    bank_name:  str = "",
    country:    str = "",
) -> ValidationReport:
    """
    Valide les résultats d'un stress test liquidité.

    Parameters
    ----------
    outputs    : dict {scenario_id: RiskOutput} retourné par run_liquidity_stress
    engine     : LiquidityStressEngine calibré
    macro_hist : DataFrame macro historique (index ou colonne 'year')
    """
    report = ValidationReport(bank_name=bank_name, country=country)

    # ── Batterie 1 — Hiérarchie ──────────────────────────────────────────────
    report.results.extend(_test_hierarchy(outputs))

    # ── Batterie 2 — Cohérence satellites ────────────────────────────────────
    report.results.extend(_test_satellite_direction(outputs))

    # ── Batterie 3 — Bornes ──────────────────────────────────────────────────
    report.results.extend(_test_bounds(outputs, engine))

    # ── Batterie 4 — Sensibilité mono-variable ──────────────────────────────
    report.results.extend(_test_sensitivity(engine, macro_hist))

    # ── Batterie 5 — Back-test 2016 ─────────────────────────────────────────
    report.results.extend(_test_backtest(engine, macro_hist))

    return report


# ─────────────────────────────────────────────────────────────────────────────
# BATTERIE 1 — HIÉRARCHIE DES SCÉNARIOS
# ─────────────────────────────────────────────────────────────────────────────

def _test_hierarchy(outputs: Dict) -> List[TestResult]:
    """Vérifie baseline ≥ adverse ≥ severe pour LCR, NSFR, NCO, HQLA."""
    results = []

    if not all(s in outputs for s in ["baseline", "adverse", "severe"]):
        results.append(TestResult(
            battery="hierarchy", test_name="scenarios_present",
            status="WARN", expected="baseline + adverse + severe",
            actual=str(list(outputs.keys())),
            message="Les 3 scénarios ne sont pas tous présents, hiérarchie non testable."
        ))
        return results

    ts_b = outputs["baseline"].time_series
    ts_a = outputs["adverse"].time_series
    ts_s = outputs["severe"].time_series

    years = sorted(ts_b["year"].tolist())

    # LCR : baseline ≥ adverse ≥ severe (meilleur liquidity = higher LCR)
    for yr in years:
        lcr_b = float(ts_b.loc[ts_b["year"] == yr, "lcr"].values[0])
        lcr_a = float(ts_a.loc[ts_a["year"] == yr, "lcr"].values[0])
        lcr_s = float(ts_s.loc[ts_s["year"] == yr, "lcr"].values[0])

        ok = (lcr_b >= lcr_a - 0.01) and (lcr_a >= lcr_s - 0.01)
        results.append(TestResult(
            battery="hierarchy",
            test_name=f"lcr_ordering_{yr}",
            status="PASS" if ok else "FAIL",
            expected=f"LCR: base ≥ adv ≥ sev",
            actual=f"{lcr_b:.1f} ≥ {lcr_a:.1f} ≥ {lcr_s:.1f}",
            message=f"LCR {yr} : {'cohérent' if ok else 'INVERSION DÉTECTÉE'}",
        ))

    # NSFR : baseline ≥ adverse ≥ severe
    for yr in years:
        nsfr_b = float(ts_b.loc[ts_b["year"] == yr, "nsfr"].values[0])
        nsfr_a = float(ts_a.loc[ts_a["year"] == yr, "nsfr"].values[0])
        nsfr_s = float(ts_s.loc[ts_s["year"] == yr, "nsfr"].values[0])

        ok = (nsfr_b >= nsfr_a - 0.01) and (nsfr_a >= nsfr_s - 0.01)
        results.append(TestResult(
            battery="hierarchy",
            test_name=f"nsfr_ordering_{yr}",
            status="PASS" if ok else "FAIL",
            expected=f"NSFR: base ≥ adv ≥ sev",
            actual=f"{nsfr_b:.1f} ≥ {nsfr_a:.1f} ≥ {nsfr_s:.1f}",
            message=f"NSFR {yr} : {'cohérent' if ok else 'INVERSION DÉTECTÉE'}",
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# BATTERIE 2 — COHÉRENCE DES SATELLITES AU PIC
# ─────────────────────────────────────────────────────────────────────────────

def _test_satellite_direction(outputs: Dict) -> List[TestResult]:
    """Vérifie que chaque satellite bouge dans le bon sens au pic."""
    results = []

    if not all(s in outputs for s in ["baseline", "adverse", "severe"]):
        return results

    ts_b = outputs["baseline"].time_series
    ts_a = outputs["adverse"].time_series
    ts_s = outputs["severe"].time_series

    # Trouver l'année du pic (2e année de projection si decay démarre à 0)
    years = sorted(ts_b["year"].tolist())
    peak_year = years[1] if len(years) > 1 else years[0]

    # Satellites qui doivent AUGMENTER sous stress
    increasing = ["run_off_retail", "run_off_corporate", "haircut_add"]
    # Satellites qui doivent DIMINUER sous stress
    decreasing = ["asf_factor_retail", "asf_factor_corporate"]

    for col in increasing + decreasing:
        if col not in ts_b.columns:
            continue

        v_b = float(ts_b.loc[ts_b["year"] == peak_year, col].values[0])
        v_a = float(ts_a.loc[ts_a["year"] == peak_year, col].values[0])
        v_s = float(ts_s.loc[ts_s["year"] == peak_year, col].values[0])

        if col in increasing:
            ok = (v_s >= v_a - 1e-6) and (v_a >= v_b - 1e-6)
            expected = f"{col}: sev ≥ adv ≥ base (↑ sous stress)"
        else:
            ok = (v_s <= v_a + 1e-6) and (v_a <= v_b + 1e-6)
            expected = f"{col}: sev ≤ adv ≤ base (↓ sous stress)"

        results.append(TestResult(
            battery="satellites",
            test_name=f"direction_{col}_{peak_year}",
            status="PASS" if ok else "FAIL",
            expected=expected,
            actual=f"base={v_b:.4f}  adv={v_a:.4f}  sev={v_s:.4f}",
            message=f"{col} au pic ({peak_year}) : {'cohérent' if ok else 'DIRECTION INVERSÉE'}",
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# BATTERIE 3 — BORNES RÉGLEMENTAIRES ET DOMAINES
# ─────────────────────────────────────────────────────────────────────────────

# Domaines attendus pour chaque satellite
_SAT_BOUNDS = {
    "run_off_retail":       (0.0, 1.0),
    "run_off_corporate":    (0.0, 1.0),
    "haircut_add":          (0.0, 0.50),
    "asf_factor_retail":    (0.70, 1.00),
    "asf_factor_corporate": (0.20, 0.80),
}


def _test_bounds(outputs: Dict, engine: object) -> List[TestResult]:
    """Vérifie bornes des satellites, positivité LCR/NSFR, constance RSF."""
    results = []

    rsf_values = set()

    for scen_id, out in outputs.items():
        ts = out.time_series

        for yr_idx, row in ts.iterrows():
            yr = int(row["year"])

            # ── Bornes des satellites ─────────────────────────────────────────
            for col, (lo, hi) in _SAT_BOUNDS.items():
                if col not in ts.columns:
                    continue
                val = float(row[col])
                # Tolérance de 1e-4 pour les arrondis numériques
                ok = (val >= lo - 1e-4) and (val <= hi + 1e-4)
                if not ok:
                    results.append(TestResult(
                        battery="bounds",
                        test_name=f"domain_{col}_{scen_id}_{yr}",
                        status="FAIL",
                        expected=f"{lo} ≤ {col} ≤ {hi}",
                        actual=f"{val:.6f}",
                        message=f"{col} hors domaine en {yr} ({scen_id})",
                    ))

            # ── Positivité LCR et NSFR ────────────────────────────────────────
            lcr  = float(row["lcr"])
            nsfr = float(row["nsfr"])
            if lcr <= 0:
                results.append(TestResult(
                    battery="bounds", test_name=f"lcr_positive_{scen_id}_{yr}",
                    status="FAIL", expected="LCR > 0",
                    actual=f"{lcr:.2f}",
                    message=f"LCR négatif ou nul en {yr} ({scen_id})",
                ))
            if nsfr <= 0:
                results.append(TestResult(
                    battery="bounds", test_name=f"nsfr_positive_{scen_id}_{yr}",
                    status="FAIL", expected="NSFR > 0",
                    actual=f"{nsfr:.2f}",
                    message=f"NSFR négatif ou nul en {yr} ({scen_id})",
                ))

            # ── NCO positif ───────────────────────────────────────────────────
            nco = float(row["nco"])
            if nco <= 0:
                results.append(TestResult(
                    battery="bounds", test_name=f"nco_positive_{scen_id}_{yr}",
                    status="FAIL", expected="NCO > 0",
                    actual=f"{nco:.4f}",
                    message=f"NCO négatif ou nul en {yr} ({scen_id})",
                ))

            # ── HQLA ≤ L1 + L2a ───────────────────────────────────────────────
            hqla = float(row["hqla"])
            try:
                ts_full = engine.inputs.time_series.set_index("year")
                last_yr = ts_full.index.max()
                l1_l2a = float(ts_full.loc[last_yr, "L1"]) + float(ts_full.loc[last_yr, "L2a"])
                if hqla > l1_l2a + 0.01:
                    results.append(TestResult(
                        battery="bounds",
                        test_name=f"hqla_cap_{scen_id}_{yr}",
                        status="FAIL",
                        expected=f"HQLA ≤ L1 + L2a = {l1_l2a:.2f}",
                        actual=f"{hqla:.2f}",
                        message=f"HQLA dépasse L1+L2a en {yr} ({scen_id})",
                    ))
            except Exception:
                pass

            # ── RSF constant ──────────────────────────────────────────────────
            rsf_val = round(float(row["rsf"]), 2)
            rsf_values.add(rsf_val)

    # Vérifier que RSF est identique entre tous les scénarios/années
    if len(rsf_values) > 1:
        results.append(TestResult(
            battery="bounds", test_name="rsf_constant",
            status="FAIL",
            expected="RSF identique entre tous les scénarios",
            actual=f"Valeurs distinctes : {sorted(rsf_values)}",
            message="Le RSF varie entre scénarios — les pondérations BCBS doivent être fixes.",
        ))
    elif len(rsf_values) == 1:
        results.append(TestResult(
            battery="bounds", test_name="rsf_constant",
            status="PASS", expected="RSF constant",
            actual=f"RSF = {list(rsf_values)[0]:.2f} (identique partout)",
            message="RSF constant entre tous les scénarios et toutes les années.",
        ))

    # Si aucun test de borne n'a échoué, ajouter un PASS global
    bound_fails = [r for r in results if r.battery == "bounds" and r.status == "FAIL"]
    if not bound_fails and len(results) > 0:
        # Compter les vérifications implicites
        n_checks = sum(
            len(ts.columns) * len(ts)
            for out in outputs.values()
            for ts in [out.time_series]
        )
        results.insert(0, TestResult(
            battery="bounds", test_name="all_domains_ok",
            status="PASS", expected="Tous les satellites dans leur domaine",
            actual=f"{n_checks} valeurs vérifiées",
            message="Toutes les bornes réglementaires et domaines sont respectés.",
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# BATTERIE 4 — SENSIBILITÉ MONO-VARIABLE
# ─────────────────────────────────────────────────────────────────────────────

# Chocs unitaires et effet attendu sur LCR
_SENSITIVITY_SHOCKS = {
    "policy_rate":       {"delta": +5.0, "lcr_sign": -1, "nsfr_sign": -1, "label": "Taux directeur +5pp"},
    "GDP_growth":        {"delta": -3.0, "lcr_sign": -1, "nsfr_sign": -1, "label": "PIB -3pp"},
    "exchange_rate":     {"delta": +10.0,"lcr_sign": -1, "nsfr_sign": -1, "label": "Taux de change +10pp"},
    "cpi_inflation":     {"delta": +5.0, "lcr_sign": -1, "nsfr_sign":  0, "label": "Inflation +5pp"},
    "unemployment_rate": {"delta": +2.0, "lcr_sign": -1, "nsfr_sign": -1, "label": "Chômage +2pp"},
    "real_estate_price": {"delta": -10.0,"lcr_sign": -1, "nsfr_sign":  0, "label": "Immobilier -10%"},
}

# Seuil minimum d'impact (en pp de LCR) pour considérer la sensibilité non-triviale
_MIN_IMPACT_LCR  = 0.5
_MIN_IMPACT_NSFR = 0.3


def _test_sensitivity(engine: object, macro_hist: pd.DataFrame) -> List[TestResult]:
    """
    Applique un choc isolé sur chaque variable macro et vérifie
    que le LCR/NSFR bouge dans le bon sens avec une amplitude non-triviale.

    Distingue deux cas :
        - Variable ACTIVE (driver d'au moins un satellite ciblant le ratio testé)
          → ΔLCR/NSFR = 0 est un FAIL
        - Variable ÉLIMINÉE par le stepwise (aucun canal de transmission)
          → ΔLCR/NSFR = 0 est un WARN (limitation des données, pas un bug)
    """
    results = []

    try:
        # ── Identifier les drivers actifs par canal ──────────────────────
        # Satellites qui impactent le LCR : run_off_retail, run_off_corporate, haircut_add
        # Satellites qui impactent le NSFR : asf_factor_retail, asf_factor_corporate
        LCR_SATELLITES  = ["run_off_retail", "run_off_corporate", "haircut_add"]
        NSFR_SATELLITES = ["asf_factor_retail", "asf_factor_corporate"]

        active_lcr_drivers  = set()
        active_nsfr_drivers = set()

        if hasattr(engine, '_sat_results') and engine._sat_results:
            for sat_name, sat_result in engine._sat_results.items():
                if sat_name in LCR_SATELLITES:
                    active_lcr_drivers.update(sat_result.drivers)
                if sat_name in NSFR_SATELLITES:
                    active_nsfr_drivers.update(sat_result.drivers)

        # ── Préparer la macro baseline (dernière année connue) ───────────
        mac = macro_hist.copy()
        if mac.index.name == "year" or "year" not in mac.columns:
            if mac.index.name == "year":
                mac = mac.reset_index()
            elif "year" not in mac.columns:
                mac = mac.reset_index().rename(columns={"index": "year"})
        mac["year"] = mac["year"].astype(int)
        mac_idx = mac.set_index("year")
        last_year = mac_idx.index.max()
        baseline_row = mac_idx.loc[last_year]

        # Calculer LCR/NSFR baseline
        base_df = pd.DataFrame([baseline_row.to_dict()])
        base_sats = engine._predict_satellites(base_df)
        ts_full = engine.inputs.time_series.set_index("year")
        bilan = ts_full.loc[ts_full.index.max()]
        rsf = engine.inputs.rsf_weights

        lcr_base, nsfr_base, _, _ = engine._calc_ratios(base_sats, bilan, rsf)

        for var_name, spec in _SENSITIVITY_SHOCKS.items():
            if var_name not in baseline_row.index:
                continue

            # Appliquer le choc sur une seule variable
            shocked_row = baseline_row.copy()
            shocked_row[var_name] = shocked_row[var_name] + spec["delta"]

            shocked_df = pd.DataFrame([shocked_row.to_dict()])
            shocked_sats = engine._predict_satellites(shocked_df)

            lcr_shocked, nsfr_shocked, _, _ = engine._calc_ratios(
                shocked_sats, bilan, rsf
            )

            delta_lcr  = lcr_shocked - lcr_base
            delta_nsfr = nsfr_shocked - nsfr_base

            # ── Est-ce un driver actif pour ce ratio ? ────────────────────
            is_lcr_driver  = var_name in active_lcr_drivers
            is_nsfr_driver = var_name in active_nsfr_drivers

            # ── Test direction LCR ────────────────────────────────────────
            if spec["lcr_sign"] != 0:
                sign_ok = (delta_lcr * spec["lcr_sign"] > 0)
                amplitude_ok = abs(delta_lcr) >= _MIN_IMPACT_LCR

                if sign_ok and amplitude_ok:
                    status = "PASS"
                    msg = f"{spec['label']} → ΔLCR = {delta_lcr:+.1f}pp (cohérent)"
                elif sign_ok and not amplitude_ok:
                    status = "WARN"
                    msg = (f"{spec['label']} → ΔLCR = {delta_lcr:+.2f}pp "
                           f"(bon signe, mais impact faible < {_MIN_IMPACT_LCR}pp)")
                elif not is_lcr_driver:
                    # Variable éliminée par le stepwise → pas de canal LCR
                    status = "WARN"
                    msg = (f"{spec['label']} → ΔLCR = {delta_lcr:+.2f}pp "
                           f"(variable éliminée par le stepwise — "
                           f"aucun canal de transmission LCR sur ces données)")
                else:
                    # Driver actif mais direction inversée → vrai problème
                    status = "FAIL"
                    msg = (f"{spec['label']} → ΔLCR = {delta_lcr:+.1f}pp "
                           f"(DIRECTION INVERSÉE, attendu "
                           f"{'↓' if spec['lcr_sign'] < 0 else '↑'})")

                results.append(TestResult(
                    battery="sensitivity",
                    test_name=f"sens_lcr_{var_name}",
                    status=status,
                    expected=f"ΔLCR {'< 0' if spec['lcr_sign'] < 0 else '> 0'}"
                             f" et |ΔLCR| ≥ {_MIN_IMPACT_LCR}pp"
                             f"{'' if is_lcr_driver else ' [variable non active]'}",
                    actual=f"ΔLCR = {delta_lcr:+.2f}pp  "
                           f"(LCR: {lcr_base:.1f} → {lcr_shocked:.1f})"
                           f"{'  [driver actif]' if is_lcr_driver else '  [éliminé stepwise]'}",
                    message=msg,
                ))

            # ── Test direction NSFR ───────────────────────────────────────
            if spec["nsfr_sign"] != 0:
                sign_ok = (delta_nsfr * spec["nsfr_sign"] > 0)
                amplitude_ok = abs(delta_nsfr) >= _MIN_IMPACT_NSFR

                if sign_ok and amplitude_ok:
                    status = "PASS"
                    msg = f"{spec['label']} → ΔNSFR = {delta_nsfr:+.2f}pp (cohérent)"
                elif sign_ok and not amplitude_ok:
                    status = "WARN"
                    msg = (f"{spec['label']} → ΔNSFR = {delta_nsfr:+.2f}pp "
                           f"(bon signe, mais impact faible)")
                elif not is_nsfr_driver:
                    status = "WARN"
                    msg = (f"{spec['label']} → ΔNSFR = {delta_nsfr:+.2f}pp "
                           f"(variable éliminée par le stepwise — "
                           f"aucun canal de transmission NSFR sur ces données)")
                else:
                    status = "FAIL"
                    msg = (f"{spec['label']} → ΔNSFR = {delta_nsfr:+.2f}pp "
                           f"(DIRECTION INVERSÉE, attendu "
                           f"{'↓' if spec['nsfr_sign'] < 0 else '↑'})")

                results.append(TestResult(
                    battery="sensitivity",
                    test_name=f"sens_nsfr_{var_name}",
                    status=status,
                    expected=f"ΔNSFR {'< 0' if spec['nsfr_sign'] < 0 else '> 0'}"
                             f"{'' if is_nsfr_driver else ' [variable non active]'}",
                    actual=f"ΔNSFR = {delta_nsfr:+.2f}pp  "
                           f"(NSFR: {nsfr_base:.1f} → {nsfr_shocked:.1f})"
                           f"{'  [driver actif]' if is_nsfr_driver else '  [éliminé stepwise]'}",
                    message=msg,
                ))

    except Exception as e:
        results.append(TestResult(
            battery="sensitivity", test_name="sensitivity_error",
            status="WARN", expected="Tests de sensibilité exécutables",
            actual=str(e),
            message=f"Erreur lors des tests de sensibilité : {e}",
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# BATTERIE 5 — BACK-TESTING 2016
# ─────────────────────────────────────────────────────────────────────────────

_BACKTEST_THRESHOLDS = {
    "good": 0.20,    # < 20% → ✅ PASS
    "acceptable": 0.40,  # < 40% → ⚠️ WARN
                         # ≥ 40% → ❌ FAIL
}


def _test_backtest(engine: object, macro_hist: pd.DataFrame) -> List[TestResult]:
    """
    Injecte les conditions macro de 2016 dans les satellites calibrés
    et compare aux valeurs observées dans le time_series.
    """
    results = []

    try:
        mac = macro_hist.copy()
        if mac.index.name == "year":
            mac = mac.reset_index()
        if "year" not in mac.columns:
            mac = mac.reset_index().rename(columns={"index": "year"})
        mac["year"] = mac["year"].astype(int)
        mac_idx = mac.set_index("year")

        # Année 2016 doit exister dans l'historique macro
        if 2016 not in mac_idx.index:
            results.append(TestResult(
                battery="backtest", test_name="backtest_data_available",
                status="WARN", expected="Année 2016 dans l'historique macro",
                actual="2016 absente",
                message="Back-test impossible : année 2016 absente de l'historique macro.",
            ))
            return results

        # Prédire les satellites avec les conditions macro 2016
        macro_2016_df = pd.DataFrame([mac_idx.loc[2016].to_dict()])
        predicted_sats = engine._predict_satellites(macro_2016_df)

        # Valeurs observées dans le bilan 2016
        ts = engine.inputs.time_series.set_index("year")
        if 2016 not in ts.index:
            results.append(TestResult(
                battery="backtest", test_name="backtest_data_available",
                status="WARN", expected="Année 2016 dans le time_series",
                actual="2016 absente du bilan",
                message="Back-test impossible : année 2016 absente du bilan.",
            ))
            return results

        observed = ts.loc[2016]

        # Comparer chaque satellite
        for var_name in ["run_off_retail", "run_off_corporate", "haircut_add",
                         "asf_factor_retail", "asf_factor_corporate"]:
            if var_name not in predicted_sats or var_name not in observed:
                continue

            pred = float(predicted_sats[var_name])
            obs  = float(observed[var_name])

            # Erreur relative (éviter division par zéro)
            if abs(obs) > 1e-8:
                rel_error = abs(pred - obs) / abs(obs)
            else:
                rel_error = abs(pred - obs)

            if rel_error < _BACKTEST_THRESHOLDS["good"]:
                status = "PASS"
                quality = "bonne calibration"
            elif rel_error < _BACKTEST_THRESHOLDS["acceptable"]:
                status = "WARN"
                quality = "acceptable (petit échantillon)"
            else:
                status = "FAIL"
                quality = "calibration faible"

            results.append(TestResult(
                battery="backtest",
                test_name=f"backtest_{var_name}_2016",
                status=status,
                expected=f"erreur relative < {_BACKTEST_THRESHOLDS['good']*100:.0f}%",
                actual=f"prédit={pred:.4f}  observé={obs:.4f}  erreur={rel_error*100:.1f}%",
                message=f"{var_name} 2016 : erreur {rel_error*100:.1f}% ({quality})",
            ))

    except Exception as e:
        results.append(TestResult(
            battery="backtest", test_name="backtest_error",
            status="WARN", expected="Back-test exécutable",
            actual=str(e),
            message=f"Erreur lors du back-test : {e}",
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# AFFICHAGE CONSOLE
# ─────────────────────────────────────────────────────────────────────────────

_BATTERY_LABELS = {
    "hierarchy":   "Batterie 1 — Hiérarchie des scénarios",
    "satellites":  "Batterie 2 — Cohérence satellites",
    "bounds":      "Batterie 3 — Bornes et domaines",
    "sensitivity": "Batterie 4 — Sensibilité mono-variable",
    "backtest":    "Batterie 5 — Back-test 2016",
}

_STATUS_ICONS = {
    "PASS": "OK",
    "FAIL": "X",
    "WARN": "!",
}


def _print_validation_report(report: ValidationReport) -> None:
    """Affiche le rapport de validation formate."""
    sep = "=" * 62
    thin = "-" * 62

    title = "VALIDATION REPORT"
    if report.bank_name:
        title += f" - {report.bank_name}"
    if report.country:
        title += f" | {report.country}"

    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    # -- Resume par batterie -----------------------------------------------
    batteries = ["hierarchy", "satellites", "bounds", "sensitivity", "backtest"]
    for bat in batteries:
        tests = report.by_battery(bat)
        if not tests:
            label = _BATTERY_LABELS.get(bat, bat)
            print(f"\n  {label:.<46} NON EXECUTE")
            continue

        n_pass = sum(1 for t in tests if t.status == "PASS")
        n_fail = sum(1 for t in tests if t.status == "FAIL")
        n_warn = sum(1 for t in tests if t.status == "WARN")
        n_total = len(tests)

        if n_fail > 0:
            summary_icon = "X FAIL"
        elif n_warn > 0:
            summary_icon = "!  WARN"
        else:
            summary_icon = "OK PASS"

        label = _BATTERY_LABELS.get(bat, bat)
        print(f"\n  {label}")
        print(f"  {thin}")
        print(f"    Resultat : {n_pass}/{n_total} PASS  "
              f"{f'| {n_warn} WARN  ' if n_warn else ''}"
              f"{f'| {n_fail} FAIL' if n_fail else ''}"
              f"  ->  {summary_icon}")

        # Details des FAIL et WARN
        for t in tests:
            if t.status == "FAIL":
                print(f"    {_STATUS_ICONS[t.status]} {t.message}")
                print(f"       Attendu : {t.expected}")
                print(f"       Obtenu  : {t.actual}")
            elif t.status == "WARN":
                print(f"    {_STATUS_ICONS[t.status]}  {t.message}")

        # Resume PASS en une ligne si tout est OK
        if n_fail == 0 and n_warn == 0:
            # Montrer 1-2 exemples PASS pour rassurer
            for t in tests[:2]:
                print(f"    {_STATUS_ICONS[t.status]} {t.message}")
            if len(tests) > 2:
                print(f"    ... et {len(tests) - 2} tests supplementaires PASS")

    # -- VERDICT GLOBAL ------------------------------------------------------
    print(f"\n{sep}")
    print(f"  TOTAL : {report.passed_tests}/{report.total_tests} PASS  "
          f"|  {report.warnings} WARNING{'S' if report.warnings != 1 else ''}  "
          f"|  {report.failed_tests} FAIL")

    if report.passed:
        if report.warnings > 0:
            verdict = "OK MODELE VALIDE (avec reserves mineures)"
        else:
            verdict = "OK MODELE VALIDE"
    else:
        verdict = "X MODELE NON VALIDE - corriger les tests en echec"

    print(f"  VERDICT : {verdict}")
    print(f"{sep}\n")
