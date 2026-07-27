"""
scenario_selector.py
====================
Automatic selection of the optimal (adverse, severe) NGFS scenario pair
from **all scenarios available in the dataset**, using a multi-criteria
framework of hard constraints + weighted soft scores.

Design principles
-----------------
* Generic   — country-agnostic, risk-type-agnostic, model-family-agnostic.
* Decoupled — receives a pre-built ``pd_matrix``; no projection logic inside.
              Caller (multi_scenario.py) projects PD for every available
              scenario and passes the result here.
* Configurable — all thresholds and weights live in ``ScenarioSelectorConfig``.

Typical integration in multi_scenario.py
-----------------------------------------
    # 1. Project PD for every NGFS scenario with the chosen model
    pd_matrix: Dict[str, pd.Series] = {}
    for sc_name in ngfs_country["scenario"].unique():
        proj = _project_ct_levels(...) if is_ct else _project_model(...)
        if proj:
            pd_matrix[sc_name] = pd.Series(
                {p["year"]: p["value"] for p in proj}
            ).dropna()

    # 2. Find optimal pair
    from .scenario_selector import find_optimal_scenario_pair, ScenarioSelectorConfig
    result = find_optimal_scenario_pair(
        pd_matrix=pd_matrix,
        baseline_name=scn_map["baseline"],
        cfg=ScenarioSelectorConfig(),
    )
    if result.valid:
        scn_map["adverse"] = result.adverse
        scn_map["severe"]  = result.severe
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioSelectorConfig:
    """
    All knobs for the scenario pair selector.

    Hard constraints  (violation → pair rejected)
    -----------------------------------------------
    min_stress_spread       : float   minimum absolute spread
                                      mean(PD_adverse) - mean(PD_baseline)
                                      below which a scenario is considered
                                      "not stressed" and eliminated.  [0.001]

    min_adv_sev_spread      : float   minimum absolute spread
                                      mean(PD_severe) - mean(PD_adverse)
                                      below which the two stressed scenarios
                                      are considered indistinguishable.  [0.0005]

    min_monotonicity_ratio  : float   DÉSACTIVÉ comme hard filter (défaut=0.0).
                                      La monotonicity année par année est conservée
                                      comme soft score S7 — elle influence le
                                      classement mais ne rejette plus de paires.
                                      Raison : avec peu d'années de projection,
                                      les trajectoires se croisent naturellement
                                      même pour des scénarios économiquement corrects,
                                      ce qui rejetait les paires les plus sévères.

    max_early_stress_lag    : int     adverse must exceed baseline within
                                      the first N years of the horizon.  [3]

    Soft scores weights  (sum need not equal 1 — auto-normalised)
    --------------------------------------------------------------
    w_spread_abs      : absolute spread  severe vs baseline
    w_spread_rel      : relative spread  severe / baseline
    w_differentiation : spread  severe vs adverse
    w_persistence_adv : % years  PD_adverse > PD_baseline
    w_persistence_sev : % years  PD_severe  > PD_baseline
    w_convexity       : stress concentrated in later years (climate-realistic)
    w_monotonicity    : smooth year-by-year ordering quality
    w_regularity      : adverse in ideal band of B→S spread
    w_stability       : low variance of adverse position over time
    w_severity_coherence : a priori NGFS severity hierarchy respected
    w_early_consistency  : % early years where adverse > baseline
    w_yearly_joint_order : % ALL years where adverse > baseline AND severe > adverse
    """
    # ── Hard constraints ──────────────────────────────────────────────
    min_stress_spread:      float = 0.001
    min_adv_sev_spread:     float = 0.0001
    min_monotonicity_ratio: float = 0.0
    max_early_stress_lag:   int   = 3

    # ── Soft score weights (auto-normalised) ──────────────────────────
    w_spread_abs:          float = 1.0
    w_spread_rel:          float = 1.0
    w_differentiation:     float = 1.5
    w_persistence_adv:     float = 0.8
    w_persistence_sev:     float = 0.8
    w_convexity:           float = 0.6
    w_monotonicity:        float = 1.2
    w_regularity:          float = 0.8
    w_stability:           float = 0.6
    # Nouveaux scores économiques — poids élevés car plus discriminants
    w_severity_coherence:  float = 2.0   # S10 : hiérarchie NGFS a priori
    w_early_consistency:   float = 1.8   # S11 : cohérence années initiales
    w_yearly_joint_order:  float = 2.0   # S12 : % années B<A<S simultanément

    # ── Regularity band ─────────────────────────────────────────────────
    reg_band_low:  float = 0.30
    reg_band_high: float = 0.70

    # ── NGFS severity ranks (configurable, chargé depuis mapping.yaml) ──
    # Plus le rang est élevé, plus le scénario est sévère économiquement.
    # Un scénario absent reçoit le rang médian = 3.
    scenario_severity_ranks: Dict[str, int] = field(default_factory=lambda: {
        "Net Zero 2050":                                1,
        "Below 2°C":                                   2,
        "Nationally Determined Contributions (NDCs)":  3,
        "Delayed transition":                          4,
        "Current Policies":                            5,
        "Fragmented World":                            6,
    })

    @property
    def weight_vector(self) -> Dict[str, float]:
        raw = {
            "spread_abs":          self.w_spread_abs,
            "spread_rel":          self.w_spread_rel,
            "differentiation":     self.w_differentiation,
            "persistence_adv":     self.w_persistence_adv,
            "persistence_sev":     self.w_persistence_sev,
            "convexity":           self.w_convexity,
            "monotonicity":        self.w_monotonicity,
            "regularity":          self.w_regularity,
            "stability":           self.w_stability,
            "severity_coherence":  self.w_severity_coherence,
            "early_consistency":   self.w_early_consistency,
            "yearly_joint_order":  self.w_yearly_joint_order,
        }
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioPairResult:
    """
    Output of ``find_optimal_scenario_pair``.

    Attributes
    ----------
    valid       : bool       True if at least one valid pair was found.
    baseline    : str        Name of the baseline scenario (fixed).
    adverse     : str        Name of the selected adverse scenario.
    severe      : str        Name of the selected severe scenario.
    score       : float      Composite score of the best pair (0–1).
    constraints : dict       {C_name: bool} for the winning pair.
    soft_scores : dict       {S_name: normalised_value} for the winning pair.
    ranking_df  : DataFrame  Full ranking of all valid pairs (descending score).
    rejected_df : DataFrame  Pairs eliminated by hard constraints + reason.
    n_candidates: int        Total number of (adverse, severe) pairs evaluated.
    """
    valid:        bool
    baseline:     str
    adverse:      str         = ""
    severe:       str         = ""
    score:        float       = np.nan
    constraints:  Dict[str, bool]       = field(default_factory=dict)
    soft_scores:  Dict[str, float]      = field(default_factory=dict)
    ranking_df:   pd.DataFrame          = field(default_factory=pd.DataFrame)
    rejected_df:  pd.DataFrame          = field(default_factory=pd.DataFrame)
    n_candidates: int         = 0


# ─────────────────────────────────────────────────────────────────────────────
# Hard constraints
# ─────────────────────────────────────────────────────────────────────────────

def _check_constraints(
    pd_base: pd.Series,
    pd_adv:  pd.Series,
    pd_sev:  pd.Series,
    cfg:     ScenarioSelectorConfig,
) -> Tuple[bool, Dict[str, bool], str]:
    """
    Evaluate all hard constraints for a (adverse, severe) pair.

    Returns
    -------
    passed   : bool            True if all constraints pass.
    detail   : dict            {constraint_name: bool}
    reason   : str             Human-readable rejection reason (empty if passed).
    """
    common = pd_base.index.intersection(pd_adv.index).intersection(pd_sev.index)
    if len(common) == 0:
        return False, {}, "No overlapping years between scenarios"

    b = pd_base[common].astype(float)
    a = pd_adv[common].astype(float)
    s = pd_sev[common].astype(float)

    # C1 — global ordering: mean(base) < mean(adv) < mean(sev)
    c1 = (b.mean() < a.mean()) and (a.mean() < s.mean())

    # C2 — year-by-year monotonicity: severe >= adverse in ≥ ratio of years
    c2_ratio = float((s >= a).mean())
    c2 = c2_ratio >= cfg.min_monotonicity_ratio

    # C3 — early stress: adverse > baseline within first N years
    first_n = sorted(common)[: cfg.max_early_stress_lag]
    c3 = bool((a[first_n] > b[first_n]).any()) if first_n else False

    # C4 — minimum spread: adverse meaningfully above baseline
    spread_adv = float(a.mean() - b.mean())
    c4 = spread_adv >= cfg.min_stress_spread

    # C5 — differentiation: severe meaningfully above adverse
    spread_sev_adv = float(s.mean() - a.mean())
    c5 = spread_sev_adv >= cfg.min_adv_sev_spread

    detail = {"C1_ordering": c1, "C2_monotonicity": c2,
              "C3_early_stress": c3, "C4_adv_spread": c4,
              "C5_differentiation": c5}
    passed = all(detail.values())

    reasons = []
    if not c1:
        reasons.append(
            f"C1: ordering violated "
            f"(base={b.mean():.4f} adv={a.mean():.4f} sev={s.mean():.4f})"
        )
    if not c2:
        reasons.append(f"C2: monotonicity={c2_ratio:.0%} < {cfg.min_monotonicity_ratio:.0%}")
    if not c3:
        reasons.append(f"C3: adverse not above baseline in first {cfg.max_early_stress_lag} years")
    if not c4:
        reasons.append(f"C4: adv_spread={spread_adv:.4f} < {cfg.min_stress_spread}")
    if not c5:
        reasons.append(f"C5: adv-sev_spread={spread_sev_adv:.4f} < {cfg.min_adv_sev_spread}")

    return passed, detail, " | ".join(reasons)


# ─────────────────────────────────────────────────────────────────────────────
# Soft scores
# ─────────────────────────────────────────────────────────────────────────────

def _compute_soft_scores(
    pd_base: pd.Series,
    pd_adv:  pd.Series,
    pd_sev:  pd.Series,
    severity_ranks: Optional[Dict[str, int]] = None,
    adv_name: str = "",
    sev_name: str = "",
) -> Dict[str, float]:
    """
    Compute raw (un-normalised) soft scores for a valid pair.

    All scores are designed so that **higher = better**.

    S1  spread_abs         : mean(PD_severe) - mean(PD_baseline)
    S2  spread_rel         : mean(PD_severe) / mean(PD_baseline)   (capped at 10)
    S3  differentiation    : mean(PD_severe) - mean(PD_adverse)
    S4  persistence_adv    : % years  PD_adverse > PD_baseline
    S5  persistence_sev    : % years  PD_severe  > PD_baseline
    S6  convexity          : correlation of PD_severe deviation with a late-heavy
                             linear weight — rewards scenarios where stress
                             materialises progressively (climate-realistic)
    S7  monotonicity       : mean( sigmoid(PD_severe(t) - PD_adverse(t)) ) in [0.5, 1]
                             — smooth measure of year-by-year ordering quality
    S8  regularity         : % years where adverse sits in the [30%, 70%] band
                             of the B→S spread — rewards well-centered adverse
    S9  stability          : exp(-k * std(position)) — rewards low variance
                             of adverse's relative position over time
    S10 severity_coherence : cohérence avec la hiérarchie NGFS a priori
                             rank(adverse) < rank(severe) → score=1.0
                             sinon pénalisé proportionnellement à l'écart
    S11 early_consistency  : % des années de la 1ère moitié de l'horizon
                             où adverse > baseline — pénalise les inversions initiales
    S12 yearly_joint_order : % des TOUTES les années où simultanément
                             adverse > baseline ET severe > adverse
                             — mesure la cohérence globale année par année
    """
    common = pd_base.index.intersection(pd_adv.index).intersection(pd_sev.index)
    b = pd_base[common].astype(float)
    a = pd_adv[common].astype(float)
    s = pd_sev[common].astype(float)
    n = len(common)

    mean_b = float(b.mean())
    mean_a = float(a.mean())
    mean_s = float(s.mean())

    # S1
    spread_abs = mean_s - mean_b

    # S2
    spread_rel = (mean_s / mean_b) if mean_b > 1e-12 else 0.0
    spread_rel = min(spread_rel, 10.0)

    # S3
    differentiation = mean_s - mean_a

    # S4
    persistence_adv = float((a > b).mean())

    # S5
    persistence_sev = float((s > b).mean())

    # S6 — convexity: late-heavy linear weight [1/n, 2/n, ..., 1]
    if n > 1:
        weights = np.linspace(1.0 / n, 1.0, n)
        deviation = (s - b).values
        if deviation.std() > 1e-12:
            convexity = float(np.corrcoef(deviation, weights)[0, 1])
        else:
            convexity = 0.0
        convexity = (convexity + 1.0) / 2.0
    else:
        convexity = 0.5

    # S7 — monotonicity quality via soft sigmoid
    diff_sev_adv = (s - a).values
    mono_scores = 1.0 / (1.0 + np.exp(-10.0 * diff_sev_adv))
    monotonicity = float(mono_scores.mean())

    # S8 — regularity: % years where adverse sits in [30%, 70%] of B→S spread
    denom = (s - b).values
    valid_denom = np.abs(denom) > 1e-12
    if valid_denom.sum() > 0:
        pos = np.where(
            valid_denom,
            (a - b).values / np.where(valid_denom, denom, 1.0),
            np.nan,
        )
        pos_valid = pos[valid_denom]
        in_band = (pos_valid >= 0.30) & (pos_valid <= 0.70)
        regularity = float(in_band.sum() / len(pos_valid))
    else:
        regularity = 0.0

    # S9 — stability: low variance of adverse's relative position over time
    if valid_denom.sum() >= 2:
        pos_for_std = (a.values[valid_denom] - b.values[valid_denom]) / denom[valid_denom]
        stability = float(np.exp(-4.0 * np.std(pos_for_std, ddof=1)))
    else:
        stability = 0.5

    # ── Nouveaux scores économiques ───────────────────────────────────

    # S10 — Cohérence avec la hiérarchie NGFS a priori
    # Principe : rank(adverse) < rank(severe) → ordre économique correct
    # Score = 1.0 si cohérent, décroît proportionnellement si inversé
    if severity_ranks and adv_name and sev_name:
        default_rank = 3  # rang neutre si scénario absent de la table
        rank_adv = severity_ranks.get(adv_name, default_rank)
        rank_sev = severity_ranks.get(sev_name, default_rank)
        max_rank = max(severity_ranks.values()) if severity_ranks else 6
        if rank_adv < rank_sev:
            # Ordre correct : adverse moins sévère que severe
            # Plus l'écart est grand, mieux c'est
            severity_coherence = min(1.0, (rank_sev - rank_adv) / max_rank)
        elif rank_adv == rank_sev:
            # Même rang : neutre
            severity_coherence = 0.3
        else:
            # Ordre inversé : adverse plus sévère que severe → pénalité forte
            severity_coherence = 0.0
    else:
        severity_coherence = 0.5  # neutre si pas de table disponible

    # S11 — Cohérence des années initiales
    # % des années de la 1ère moitié de l'horizon où adverse > baseline
    # Pénalise les paires où le stress s'inverse en début de période
    half = max(1, n // 2)
    early_b = b.iloc[:half]
    early_a = a.iloc[:half]
    if len(early_b) > 0:
        early_consistency = float((early_a > early_b).mean())
    else:
        early_consistency = 0.5

    # S12 — Cohérence globale année par année (joint ordering)
    # % des années où SIMULTANÉMENT adverse > baseline ET severe > adverse
    # C'est le vrai critère de cohérence — S4 et S5 le mesurent séparément,
    # S12 exige les deux conditions en même temps
    if n > 0:
        joint_ok = (a > b) & (s > a)
        yearly_joint_order = float(joint_ok.mean())
    else:
        yearly_joint_order = 0.0

    return {
        "spread_abs":         spread_abs,
        "spread_rel":         spread_rel,
        "differentiation":    differentiation,
        "persistence_adv":    persistence_adv,
        "persistence_sev":    persistence_sev,
        "convexity":          convexity,
        "monotonicity":       monotonicity,
        "regularity":         regularity,
        "stability":          stability,
        "severity_coherence": severity_coherence,
        "early_consistency":  early_consistency,
        "yearly_joint_order": yearly_joint_order,
    }


def _normalise_scores(
    records: List[Dict],
    score_keys: List[str],
) -> List[Dict]:
    """
    Min-max normalise each soft score across all valid pairs so that
    every dimension contributes on the same [0, 1] scale.
    """
    for key in score_keys:
        values = [r[key] for r in records]
        v_min, v_max = min(values), max(values)
        spread = v_max - v_min
        for r in records:
            r[f"{key}_norm"] = (
                (r[key] - v_min) / spread if spread > 1e-12 else 0.5
            )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_scenario_pair(
    pd_matrix:     Dict[str, pd.Series],
    baseline_name: str,
    cfg:           Optional[ScenarioSelectorConfig] = None,
    enforce_cross_family: bool = False,
) -> ScenarioPairResult:
    """
    Find the optimal (adverse, severe) scenario pair from all NGFS scenarios.

    Parameters
    ----------
    pd_matrix     : mapping {scenario_name → pd.Series(index=year, values=PD)}
                    Must include the baseline scenario.
    baseline_name : exact key of the baseline scenario in pd_matrix.
    enforce_cross_family : si True, adverse et severe doivent appartenir à des
                           familles différentes (CT uniquement). La famille est
                           déterminée par le préfixe avant le premier "_".
    cfg           : ScenarioSelectorConfig (defaults used if None).

    Returns
    -------
    ScenarioPairResult
    """
    if cfg is None:
        cfg = ScenarioSelectorConfig()

    # ── Validate inputs ───────────────────────────────────────────────
    if baseline_name not in pd_matrix:
        log.error("Baseline '%s' not found in pd_matrix. Keys: %s",
                  baseline_name, list(pd_matrix.keys()))
        return ScenarioPairResult(valid=False, baseline=baseline_name)

    pd_base = pd_matrix[baseline_name].dropna().sort_index()

    stressed_names = [k for k in pd_matrix if k != baseline_name]
    if len(stressed_names) < 2:
        log.error("Need at least 2 non-baseline scenarios; got %d.", len(stressed_names))
        return ScenarioPairResult(valid=False, baseline=baseline_name)

    # ── Diagnostic : PD moyennes par scénario ─────────────────────────
    log.info("── PD diagnostics ──────────────────────────────────────")
    log.info("  %-45s mean=%.4f  min=%.4f  max=%.4f",
             f"Baseline '{baseline_name}':",
             pd_base.mean(), pd_base.min(), pd_base.max())
    for sc in stressed_names:
        s = pd_matrix[sc].dropna()
        diff = s.mean() - pd_base.mean()
        log.info("  %-45s mean=%.4f  min=%.4f  max=%.4f  Δ=%+.4f",
                 f"'{sc}':", s.mean(), s.min(), s.max(), diff)
    log.info("────────────────────────────────────────────────────────")

    # ── Pré-classement par sévérité économique ────────────────────────
    # Trier les scénarios candidats par mean(PD) décroissant.
    # Cela garantit que les scénarios les plus sévères (Δ PD le plus élevé)
    # sont considérés en priorité comme severe, et les intermédiaires comme adverse.
    scenario_severity = {
        sc: float(pd_matrix[sc].dropna().mean())
        for sc in stressed_names
    }
    stressed_sorted = sorted(stressed_names,
                             key=lambda x: scenario_severity[x],
                             reverse=True)

    log.info("── Sévérité économique des scénarios (mean PD) ──────────")
    for sc in stressed_sorted:
        delta = scenario_severity[sc] - float(pd_base.mean())
        log.info("  %-45s mean=%.4f  Δ=%+.4f", f"'{sc}':",
                 scenario_severity[sc], delta)
    log.info("────────────────────────────────────────────────────────")

    log.info(
        "ScenarioSelector: evaluating %d candidate scenarios → %d ordered pairs.",
        len(stressed_names),
        len(stressed_names) * (len(stressed_names) - 1),
    )

    # ── Enumerate all ordered pairs (adverse, severe) ─────────────────
    weights = cfg.weight_vector
    score_keys = list(weights.keys())

    valid_records:   List[Dict] = []
    rejected_records: List[Dict] = []

    for adv_name, sev_name in permutations(stressed_names, 2):
        # ── Contrainte inter-famille (CT uniquement) ─────────────────
        if enforce_cross_family:
            fam_adv = adv_name.split("_")[0].upper()
            fam_sev = sev_name.split("_")[0].upper()
            if fam_adv == fam_sev:
                rejected_records.append({
                    "adverse": adv_name, "severe": sev_name,
                    "reason": f"C0: same family '{fam_adv}'",
                })
                continue

        pd_adv = pd_matrix[adv_name].dropna().sort_index()
        pd_sev = pd_matrix[sev_name].dropna().sort_index()

        passed, constraint_detail, reject_reason = _check_constraints(
            pd_base, pd_adv, pd_sev, cfg
        )

        if not passed:
            rejected_records.append({
                "adverse":  adv_name,
                "severe":   sev_name,
                "reason":   reject_reason,
                **{k: v for k, v in constraint_detail.items()},
            })
            # Logger les 5 premières raisons de rejet pour diagnostic
            if len(rejected_records) <= 5:
                log.warning("  REJECTED ('%s', '%s'): %s",
                            adv_name, sev_name, reject_reason)
            continue

        soft = _compute_soft_scores(pd_base, pd_adv, pd_sev,
                                   severity_ranks=cfg.scenario_severity_ranks,
                                   adv_name=adv_name, sev_name=sev_name)
        valid_records.append({
            "adverse":  adv_name,
            "severe":   sev_name,
            **constraint_detail,
            **soft,
        })

    log.info(
        "ScenarioSelector: %d valid pairs / %d rejected.",
        len(valid_records), len(rejected_records),
    )

    rejected_df = pd.DataFrame(rejected_records) if rejected_records else pd.DataFrame()

    if not valid_records:
        log.warning(
            "No valid (adverse, severe) pair found under current constraints. "
            "Attempting auto-relaxation of hard constraints..."
        )

        # ── Auto-relaxation : relâcher les seuils et réessayer ────────────
        relaxed_cfg = ScenarioSelectorConfig(
            min_stress_spread=0.0,          # plus de seuil minimum de spread
            min_adv_sev_spread=0.0,         # plus de seuil entre adverse et severe
            min_monotonicity_ratio=0.0,     # plus de ratio minimum
            max_early_stress_lag=999,       # plus de contrainte temporelle
            # Garder les poids soft identiques
            w_spread_abs=cfg.w_spread_abs,
            w_spread_rel=cfg.w_spread_rel,
            w_differentiation=cfg.w_differentiation,
            w_persistence_adv=cfg.w_persistence_adv,
            w_persistence_sev=cfg.w_persistence_sev,
            w_convexity=cfg.w_convexity,
            w_monotonicity=cfg.w_monotonicity,
            w_regularity=cfg.w_regularity,
            w_stability=cfg.w_stability,
        )

        relaxed_valid: List[Dict] = []
        relaxed_rejected: List[Dict] = []

        for adv_name, sev_name in permutations(stressed_names, 2):
            # Contrainte inter-famille (CT)
            if enforce_cross_family:
                fam_adv = adv_name.split("_")[0].upper()
                fam_sev = sev_name.split("_")[0].upper()
                if fam_adv == fam_sev:
                    relaxed_rejected.append({
                        "adverse": adv_name, "severe": sev_name,
                        "reason": "same family (relaxed mode)",
                    })
                    continue

            pd_adv = pd_matrix[adv_name].dropna().sort_index()
            pd_sev = pd_matrix[sev_name].dropna().sort_index()

            # Contrainte minimale : ordering moyen (B < A et A < S en moyenne)
            if pd_adv.mean() <= pd_base.mean() or pd_sev.mean() <= pd_adv.mean():
                relaxed_rejected.append({
                    "adverse": adv_name, "severe": sev_name,
                    "reason": "ordering violated (relaxed mode)",
                })
                continue

            soft = _compute_soft_scores(pd_base, pd_adv, pd_sev,
                                       severity_ranks=cfg.scenario_severity_ranks,
                                       adv_name=adv_name, sev_name=sev_name)
            relaxed_valid.append({
                "adverse": adv_name, "severe": sev_name, **soft,
            })

        if relaxed_valid:
            log.warning(
                "Auto-relaxation: found %d valid pairs with relaxed constraints.",
                len(relaxed_valid),
            )
            relaxed_valid = _normalise_scores(
                relaxed_valid,
                [k for k in cfg.weight_vector.keys()],
            )
            weights_r = cfg.weight_vector
            for r in relaxed_valid:
                r["score"] = sum(
                    weights_r[k] * r.get(f"{k}_norm", 0.0)
                    for k in weights_r
                )
            ranking_df = (
                pd.DataFrame(relaxed_valid)
                .sort_values("score", ascending=False)
                .reset_index(drop=True)
            )
            best = ranking_df.iloc[0]
            best_adv = str(best["adverse"])
            best_sev  = str(best["severe"])
            score_keys_r = list(weights_r.keys())
            best_soft = {k: float(best.get(f"{k}_norm", 0.0)) for k in score_keys_r}
            log.warning(
                "Auto-relaxation ✓ | adverse='%s'  severe='%s'  score=%.4f",
                best_adv, best_sev, float(best["score"]),
            )
            return ScenarioPairResult(
                valid=True,
                baseline=baseline_name,
                adverse=best_adv,
                severe=best_sev,
                score=float(best["score"]),
                constraints={},
                soft_scores=best_soft,
                ranking_df=ranking_df,
                rejected_df=pd.DataFrame(relaxed_rejected),
                n_candidates=len(stressed_names) * (len(stressed_names) - 1),
            )

        # Même en mode relaxé, aucune paire ne respecte l'ordering moyen
        # ── Last resort : sélection par écart absolu au Baseline ─────────
        log.error(
            "Auto-relaxation failed: no pair satisfies mean ordering. "
            "Switching to last-resort selection by absolute PD deviation."
        )

        # Trier les scénarios par |mean(stressed) - mean(baseline)| décroissant
        base_mean = float(pd_base.mean())
        abs_deviations = {
            sc: abs(float(pd_matrix[sc].dropna().mean()) - base_mean)
            for sc in stressed_names
        }
        sorted_by_deviation = sorted(
            abs_deviations.items(), key=lambda x: x[1], reverse=True
        )

        if len(sorted_by_deviation) >= 2:
            # En mode cross-family : s'assurer que severe et adverse
            # viennent de familles différentes
            if enforce_cross_family:
                best_sev = None
                best_adv = None
                for sc, dev in sorted_by_deviation:
                    if best_sev is None:
                        best_sev = sc
                        continue
                    fam_sev = best_sev.split("_")[0].upper()
                    fam_cand = sc.split("_")[0].upper()
                    if fam_cand != fam_sev:
                        best_adv = sc
                        break
                if best_adv is None:
                    # Toutes les variantes sont de la même famille
                    best_adv = sorted_by_deviation[1][0]
            else:
                best_sev = sorted_by_deviation[0][0]
                best_adv = sorted_by_deviation[1][0]

            pd_adv_lr = pd_matrix[best_adv].dropna().sort_index()
            pd_sev_lr = pd_matrix[best_sev].dropna().sort_index()
            soft_lr   = _compute_soft_scores(pd_base, pd_adv_lr, pd_sev_lr,
                                             severity_ranks=cfg.scenario_severity_ranks,
                                             adv_name=best_adv, sev_name=best_sev)
            soft_norm_lr = {k: soft_lr.get(k, 0.0) for k in soft_lr}
            score_lr = float(np.mean(list(soft_norm_lr.values())))

            log.warning(
                "Last-resort ✓ | adverse='%s' (|Δ|=%.4f)  "
                "severe='%s' (|Δ|=%.4f)  score=%.4f",
                best_adv, abs_deviations[best_adv],
                best_sev, abs_deviations[best_sev],
                score_lr,
            )
            return ScenarioPairResult(
                valid=True,
                baseline=baseline_name,
                adverse=best_adv,
                severe=best_sev,
                score=score_lr,
                constraints={},
                soft_scores=soft_norm_lr,
                ranking_df=pd.DataFrame([{
                    "adverse": best_adv, "severe": best_sev, "score": score_lr
                }]),
                rejected_df=rejected_df,
                n_candidates=len(stressed_names) * (len(stressed_names) - 1),
            )

        # Pas assez de scénarios pour former une paire
        log.error("Last-resort failed: fewer than 2 non-baseline scenarios available.")
        return ScenarioPairResult(
            valid=False,
            baseline=baseline_name,
            ranking_df=pd.DataFrame(),
            rejected_df=rejected_df,
            n_candidates=len(stressed_names) * (len(stressed_names) - 1),
        )

    # ── Normalise soft scores across all valid pairs ──────────────────
    valid_records = _normalise_scores(valid_records, score_keys)

    # ── Compute composite weighted score ─────────────────────────────
    for r in valid_records:
        r["score"] = sum(
            weights[k] * r[f"{k}_norm"] for k in score_keys
        )

    # ── Build ranking DataFrame ───────────────────────────────────────
    display_cols = (
        ["adverse", "severe", "score"]
        + [f"{k}_norm" for k in score_keys]
        + list(next(iter(valid_records), {}).get("C1_ordering", {}) and [] or
               [c for c in valid_records[0] if c.startswith("C")])
    )
    # simpler: just all columns
    ranking_df = (
        pd.DataFrame(valid_records)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    # ── Best pair ─────────────────────────────────────────────────────
    best = ranking_df.iloc[0]
    best_adv = str(best["adverse"])
    best_sev = str(best["severe"])

    constraint_cols = [c for c in ranking_df.columns if c.startswith("C")]
    soft_norm_cols  = [f"{k}_norm" for k in score_keys]

    best_constraints = {c: bool(best[c]) for c in constraint_cols if c in best}
    best_soft = {k: float(best[f"{k}_norm"]) for k in score_keys
                 if f"{k}_norm" in best}

    log.info(
        "ScenarioSelector ✓ | adverse='%s'  severe='%s'  score=%.4f",
        best_adv, best_sev, float(best["score"]),
    )
    _log_ranking_summary(ranking_df, top_n=5)

    return ScenarioPairResult(
        valid=True,
        baseline=baseline_name,
        adverse=best_adv,
        severe=best_sev,
        score=float(best["score"]),
        constraints=best_constraints,
        soft_scores=best_soft,
        ranking_df=ranking_df,
        rejected_df=rejected_df,
        n_candidates=len(stressed_names) * (len(stressed_names) - 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_ranking_summary(ranking_df: pd.DataFrame, top_n: int = 5) -> None:
    """Log a compact top-N ranking table."""
    cols = ["adverse", "severe", "score"]
    cols = [c for c in cols if c in ranking_df.columns]
    top = ranking_df[cols].head(top_n)
    lines = ["", "── ScenarioSelector: Top pairs ──────────────────"]
    lines.append(f"  {'#':<4} {'adverse':<30} {'severe':<30} {'score':>6}")
    lines.append("  " + "─" * 72)
    for i, row in top.iterrows():
        lines.append(
            f"  {i+1:<4} {str(row['adverse']):<30} "
            f"{str(row['severe']):<30} {row['score']:>6.4f}"
        )
    log.info("\n".join(lines))


def build_pd_matrix(
    ngfs_country:            "pd.DataFrame",
    chosen_model:            "FittedModel",
    baseline_name:           str,
    ngfs_to_hist:            Dict[str, str],
    best_per_var:            Dict,
    hist_df:                 "pd.DataFrame",
    hist_year_col:           str,
    target_logit_transformed: bool,
    is_ct:                   bool,
    scenario_channel:        Optional[str],
    hist_anchor:             Optional["pd.Series"] = None,
    project_fn_lt:           Optional[callable] = None,
    project_fn_ct:           Optional[callable] = None,
    baseline_series:         Optional["pd.Series"] = None,
    candidate_scenarios:     Optional[List[str]] = None,
) -> Dict[str, "pd.Series"]:
    """
    Build the pd_matrix by projecting PD for every scenario.

    Parameters
    ----------
    candidate_scenarios : liste filtrée des scénarios à projeter (optionnel).
                          Si None, utilise tous les scénarios de ngfs_country.
                          En mode CT, doit contenir les scénarios filtrés par
                          scenario_family_filter (sans _run, 1 par famille).

    Returns
    -------
    pd_matrix : {scenario_name: pd.Series(index=year, dtype=float)}
                Only scenarios with at least 2 non-null PD values are kept.
    """
    if candidate_scenarios is not None:
        available_scenarios = sorted(candidate_scenarios)
    else:
        available_scenarios = sorted(ngfs_country["scenario"].dropna().unique().tolist())
    log.info("build_pd_matrix: projecting PD for %d scenarios.", len(available_scenarios))

    pd_matrix: Dict[str, pd.Series] = {}

    # ── Injecter le Baseline directement (pas via _project_model) ────────
    if baseline_series is not None and len(baseline_series.dropna()) >= 2:
        pd_matrix[baseline_name] = baseline_series.sort_index().astype(float)
        log.info(
            "build_pd_matrix: baseline '%s' injected directly (%d points).",
            baseline_name, len(baseline_series),
        )
    else:
        log.warning(
            "build_pd_matrix: no valid baseline_series provided for '%s' — "
            "will attempt projection via project_fn.", baseline_name,
        )

    # ── Projeter tous les autres scénarios (en parallèle — projections
    #    indépendantes, ne lisent que des objets partagés en lecture seule) ──
    def _project_one(sc_name: str):
        if is_ct:
            if project_fn_ct is None:
                raise ValueError("project_fn_ct must be provided in CT mode.")
            proj = project_fn_ct(
                model=chosen_model,
                ngfs_country=ngfs_country,
                scenario_name=sc_name,
                ngfs_to_hist=ngfs_to_hist,
                best_per_var=best_per_var,
                hist_df=hist_df,
                hist_year_col=hist_year_col,
                target_logit_transformed=target_logit_transformed,
            )
        else:
            if project_fn_lt is None:
                raise ValueError("project_fn_lt must be provided in LT mode.")
            proj = project_fn_lt(
                model=chosen_model,
                ngfs_country=ngfs_country,
                scenario_name=sc_name,
                scenario_channel=scenario_channel,
                ngfs_to_hist=ngfs_to_hist,
                best_per_var=best_per_var,
                hist_anchor=hist_anchor,
                target_logit_transformed=target_logit_transformed,
            )
        return sc_name, proj

    to_project = [sc for sc in available_scenarios if sc not in pd_matrix]
    n_workers = min(len(to_project), 8) if to_project else 1
    futures = {}
    if to_project:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for sc_name in to_project:
                futures[executor.submit(_project_one, sc_name)] = sc_name

    for fut in as_completed(futures):
        sc_name = futures[fut]
        try:
            sc_name, proj = fut.result()

            if not proj:
                log.debug("build_pd_matrix: empty projection for '%s' — skipped.", sc_name)
                continue

            series = pd.Series(
                {p["year"]: p["value"] for p in proj if p["value"] is not None}
            ).sort_index().astype(float)

            if series.dropna().__len__() < 2:
                log.debug("build_pd_matrix: '%s' has < 2 valid PD values — skipped.", sc_name)
                continue

            pd_matrix[sc_name] = series
            log.debug("build_pd_matrix: '%s' → %d PD values.", sc_name, len(series))

        except Exception as exc:
            log.warning("build_pd_matrix: projection failed for '%s': %s", sc_name, exc)

    log.info(
        "build_pd_matrix: %d / %d scenarios projected successfully.",
        len(pd_matrix), len(available_scenarios),
    )

    if baseline_name not in pd_matrix:
        log.error(
            "Baseline '%s' not found in pd_matrix. Keys: %s",
            baseline_name, list(pd_matrix.keys()),
        )

    return pd_matrix
