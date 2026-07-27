"""
historical_validation.py
========================
STAGE 2 — Statistical validation of variables on HISTORICAL data.

Mise à jour :
  - Logit transform de la cible si dans (0,1)
  - Retourne TOUS les modèles + le tableau de comparaison
  - Pas de sélection automatique — l'utilisateur choisit dans multi_scenario
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .mapping import MappingResult, map_variables
from .mapping_validator import validate_mapping
from .model_estimation import FittedModel, estimate_all
from .model_ranking import build_model_table
from .preprocessing import align_and_clean
from .stepwise import backward_elimination
from .transform_selector import (
    TransformSelectionResult,
    select_best_transforms,
)
from .correlation import correlation_matrix
from .utils import EngineConfig, get_logger
from .vif import remove_high_vif

log = get_logger(__name__)


@dataclass
class Stage2Result:
    mapping: MappingResult = None
    correlation_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    vif_final: pd.DataFrame = field(default_factory=pd.DataFrame)
    vif_removed: List[str] = field(default_factory=list)
    backward_kept: List[str] = field(default_factory=list)
    backward_trace: pd.DataFrame = field(default_factory=pd.DataFrame)
    fitted_models: List[FittedModel] = field(default_factory=list)
    model_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    transform_result: Optional[TransformSelectionResult] = None
    target_logit_transformed: bool = False
    stationarity_report: Dict[str, Dict] = field(default_factory=dict)



def _filter_zero_coef_models(
    models: List[FittedModel],
    X: pd.DataFrame,
    y: pd.Series,
    std_threshold: float = 0.01,
    contrib_threshold: float = 0.02,
) -> Tuple[List[FittedModel], int]:
    """
    Supprime les modèles estimés dont une variable a un coefficient
    standardisé nul ou négligeable.

    Critères (OR) :
      β_std_i = |β_i| × σ(x_i) / σ(y) < std_threshold      (absolu)
      contribution_i = |β_std_i| / Σ|β_std_j| < contrib_threshold  (relatif)

    Un modèle est supprimé si AU MOINS UNE variable échoue l'un des critères.

    Parameters
    ----------
    models           : liste de FittedModel à filtrer
    X                : DataFrame des variables (même index que y)
    y                : série cible (pour σ(y))
    std_threshold    : seuil absolu sur β_std (défaut 0.01)
    contrib_threshold: seuil relatif de contribution (défaut 0.02 = 2%)

    Returns
    -------
    (models_ok, n_dropped)
    """
    sigma_y = float(y.std(ddof=1)) if float(y.std(ddof=1)) > 1e-12 else 1.0
    sigma_x = X.std(ddof=1)

    models_ok = []
    n_dropped = 0

    for m in models:
        if not m.converged or not m.coefficients:
            models_ok.append(m)
            continue

        coefs_no_const = {
            v: c for v, c in m.coefficients.items() if v != "const"
        }
        if not coefs_no_const:
            models_ok.append(m)
            continue

        # β standardisé par variable
        beta_std = {}
        for v, beta in coefs_no_const.items():
            sx = float(sigma_x.get(v, 0.0))
            beta_std[v] = abs(beta) * sx / sigma_y

        total = sum(beta_std.values()) or 1.0
        contrib = {v: b / total for v, b in beta_std.items()}

        # Vérifier si une variable échoue
        bad_vars = [
            v for v in coefs_no_const
            if beta_std.get(v, 0.0) < std_threshold
            or contrib.get(v, 0.0) < contrib_threshold
        ]

        if bad_vars:
            log.debug(
                "Zero-coef filter: drop model '%s' — vars with near-zero β_std: %s",
                m.name, bad_vars,
            )
            n_dropped += 1
        else:
            models_ok.append(m)

    return models_ok, n_dropped


def run_stage2(stage1_variables: List[str],
               hist_df: pd.DataFrame,
               cfg: EngineConfig,
               pre_mapping: Optional[MappingResult] = None) -> Stage2Result:

    log.info("=== STAGE 2: historical validation (target='%s', risk='%s') ===",
             cfg.target_variable, cfg.risk_type)

    # ── 1) Mapping ───────────────────────────────────────────────────
    hist_cols = [c for c in hist_df.columns
                 if c not in (cfg.hist_year_col, cfg.target_variable)]

    if pre_mapping is not None and pre_mapping.matches:
        mapping = pre_mapping
        log.info("Stage 2: using pre-computed mapping (%d matches).",
                 len(mapping.matches))
    else:
        mapping = validate_mapping(
            stage1_variables, hist_cols, cfg,
            auto_accept=getattr(cfg, "auto_accept_mapping", True),
        )

    if not mapping.matches:
        raise ValueError(
            "Stage 2: no NGFS variables could be mapped to historical columns."
        )

    candidate_hist_cols = sorted(set(mapping.matches.values()))
    log.info("Stage 2: %d distinct historical columns after mapping.",
             len(candidate_hist_cols))

    # ── 1b) Restriction cross-risque (climat uniquement) ──────────────
    # cfg.restrict_candidate_vars, quand fourni par climate/wrapper.py, ne
    # contient que les variables acceptées à la fois par PD (crédit), au
    # moins une cible liquidité et au moins un beta marché (voir
    # cross_risk_selector.select_unanimous_variables). But : un même choc
    # macro doit être crédible pour les 3 risques, pas juste celui qui
    # corrèle le mieux avec la cible de ce module. Dégradation : si
    # l'intersection est vide, on ignore la restriction plutôt que de
    # bloquer tout le module crédit.
    _restrict = getattr(cfg, "restrict_candidate_vars", None)
    if _restrict:
        _restricted_cols = [c for c in candidate_hist_cols if c in _restrict]
        if _restricted_cols:
            log.info(
                "Stage 2: restriction cross-risque (PD + liquidité + marché) "
                "→ %d/%d colonnes retenues: %s",
                len(_restricted_cols), len(candidate_hist_cols), _restricted_cols,
            )
            candidate_hist_cols = _restricted_cols
        else:
            log.warning(
                "Stage 2: restriction cross-risque vide (aucune colonne "
                "candidate commune aux 3 risques) — restriction ignorée, "
                "pool de mapping complet conservé."
            )

    # ── 2) Align + clean ─────────────────────────────────────────────
    X_full, y = align_and_clean(hist_df, candidate_hist_cols, cfg)
    if X_full.shape[1] == 0:
        raise ValueError("Stage 2: no usable historical columns after cleaning.")

    # ── 2b) Pas de logit transform — Logit/Beta/Vasicek gèrent (0,1) nativement
    target_logit_transformed = False

    # ── 3) Best transforms ───────────────────────────────────────────
    # select_best_transforms() choisit la transformation par corrélation avec
    # la cible PUIS teste réellement (ADF+KPSS) la stationnarité de la série
    # retenue, en différenciant itérativement (jusqu'à d=2) si nécessaire.
    # Le rapport de stationnarité (transform_result.final_stationarity)
    # reflète donc la situation RÉELLE de la série effectivement utilisée par
    # le modèle — pas une hypothèse calculée sur le niveau brut avant
    # sélection.
    log.info(
        "Stage 2: sélection des transformations + vérification de "
        "stationnarité réelle sur %d macro variables...",
        X_full.shape[1],
    )
    transform_result = select_best_transforms(X_full, y, cfg)
    stationarity_report = transform_result.final_stationarity

    if transform_result.X_transformed.empty:
        raise ValueError(
            "Stage 2: aucune variable ne passe le filtre de corrélation "
            "après toutes les transformations."
        )

    X_work = transform_result.X_transformed
    y_work = y.reindex(X_work.index).dropna()
    X_work = X_work.reindex(y_work.index)

    log.info("Stage 2: %d variables transformées, %d observations.",
             X_work.shape[1], X_work.shape[0])

    # ── 4) VIF ────────────────────────────────────────────────────────
    X_vif, vif_final, vif_removed = remove_high_vif(X_work, cfg)
    if X_vif.shape[1] == 0:
        raise ValueError("Stage 2: VIF removed every variable.")

    # ── 5) Backward ──────────────────────────────────────────────────
    backward_kept, trace = backward_elimination(X_vif, y_work, cfg)
    if not backward_kept:
        backward_kept = X_vif.columns.tolist()
        log.warning("Backward empty; using VIF-survivors.")
    log.info("Stage 2: %d variables after backward: %s",
             len(backward_kept), backward_kept)

    backward_kept = backward_kept[: cfg.max_candidate_vars]

    # ── 5b) Re-vérification VIF sur le set final ──────────────────────
    # backward_elimination() ne teste jamais la colinéarité (seulement signe
    # économique / p-value / BIC). Le VIF n'est vérifié qu'UNE FOIS, avant
    # l'élimination, sur le pool initial complet — deux variables peuvent y
    # avoir un VIF individuellement acceptable (leur variance partagée étant
    # en partie absorbée par d'autres candidats du pool), puis se retrouver
    # ensemble à l'arrivée une fois tous les autres candidats éliminés, sans
    # que leur colinéarité mutuelle n'ait jamais été revérifiée isolément
    # (ex: imports/exports biens & services, souvent co-mouvants).
    if len(backward_kept) >= 2:
        _final_kept_df, _final_vif_table, _final_vif_removed = remove_high_vif(
            X_work[backward_kept], cfg
        )
        if _final_vif_removed:
            log.warning(
                "Stage 2: %d variable(s) retirée(s) après re-vérification VIF "
                "post-backward (colinéarité mutuelle masquée par le pool "
                "initial plus large): %s",
                len(_final_vif_removed), _final_vif_removed,
            )
            backward_kept = _final_kept_df.columns.tolist()

    # ── 6) Deux sets de variables : original + standardisé ───────────
    X_base = X_work[backward_kept]

    # Paramètres de standardisation calculés sur les données d'entraînement
    X_mean   = X_base.mean()
    X_std_s  = X_base.std(ddof=1).replace(0.0, 1.0)  # éviter division /0
    X_std    = (X_base - X_mean) / X_std_s            # z-score

    # Scaler complet : {col: (mean, std)} — utilisé dans la projection
    scaler_full = {
        col: (float(X_mean[col]), float(X_std_s[col]))
        for col in X_base.columns
    }

    log.info(
        "Stage 2: %d variables after backward — 2 sets (orig + std).",
        len(backward_kept),
    )

    # ── 7) Taille max des combos selon n_obs/k ────────────────────────
    n_obs      = len(y_work)
    k          = getattr(cfg, "min_obs_per_var", 5)
    max_vars   = max(1, min(n_obs // k, cfg.max_combo_size, len(backward_kept)))

    # min_combo_size : préférence utilisateur, mais ne peut pas dépasser
    # le nombre de variables qui ont survécu au backward elimination.
    min_combo_pref = getattr(cfg, "min_combo_size", 2)
    min_combo      = min(min_combo_pref, len(backward_kept))

    if min_combo < min_combo_pref:
        log.warning(
            "Stage 2: seulement %d variable(s) après backward — "
            "min_combo_size réduit de %d à %d.",
            len(backward_kept), min_combo_pref, min_combo,
        )

    # max_vars doit être >= min_combo
    if max_vars < min_combo:
        max_vars = min_combo

    log.info(
        "Stage 2: combo range=[%d, %d] (n_obs=%d, k=%d, backward_kept=%d).",
        min_combo, max_vars, n_obs, k, len(backward_kept),
    )

    # ── 8) Combinaisons + estimation (Logit / Beta / Vasicek) ────────
    all_models: List[FittedModel] = []
    n_combos = 0

    for r in range(min_combo, max_vars + 1):
        for combo in combinations(backward_kept, r):
            combo_list = list(combo)
            tag        = "+".join(combo_list)
            n_combos  += 1

            # Set A : variables originales (non standardisées)
            all_models.extend(
                estimate_all(
                    X_base[combo_list], y_work, cfg,
                    model_tag=tag,
                    variable_set="orig",
                    scaler=None,
                )
            )

            # Set B : variables standardisées (z-score)
            combo_scaler = {c: scaler_full[c] for c in combo_list}
            all_models.extend(
                estimate_all(
                    X_std[combo_list], y_work, cfg,
                    model_tag=f"std|{tag}",
                    variable_set="std",
                    scaler=combo_scaler,
                )
            )

    log.info(
        "Stage 2: %d combos × 2 sets × 3 familles → %d fits estimés.",
        n_combos, len(all_models),
    )

    # ── 9) Filtre coefficients nuls ──────────────────────────────────
    std_thresh    = getattr(cfg, "zero_coef_std_threshold",    0.01)
    contrib_thresh = getattr(cfg, "zero_coef_contrib_threshold", 0.02)
    all_models, n_zero_dropped = _filter_zero_coef_models(
        all_models, X_work, y_work,
        std_threshold=std_thresh,
        contrib_threshold=contrib_thresh,
    )
    log.info(
        "Stage 2: zero-coef filter → %d modèles supprimés, %d restants.",
        n_zero_dropped, len(all_models),
    )
    if not all_models:
        log.warning(
            "Zero-coef filter a tout supprimé — seuils trop stricts ? "
            "(std=%.3f, contrib=%.3f). Ajustez via cfg.zero_coef_std_threshold.",
            std_thresh, contrib_thresh,
        )

    # ── 10) Construire le tableau (PAS de sélection auto) ──────────────
    model_table = build_model_table(all_models, cfg)
    log.info("Stage 2: %d models in comparison table.", len(model_table))

    # Corrélations pour diagnostics
    corr_table = (
        transform_result.transform_table[transform_result.transform_table["best"]]
        .rename(columns={"abs_r": "pearson_abs_r", "transform": "best_transform"})
        [["variable", "best_transform", "pearson_abs_r", "selected"]]
        .reset_index(drop=True)
    )

    return Stage2Result(
        mapping=mapping,
        correlation_table=corr_table,
        correlation_matrix=correlation_matrix(X_full),
        vif_final=vif_final,
        vif_removed=vif_removed,
        backward_kept=backward_kept,
        backward_trace=trace,
        fitted_models=all_models,
        model_table=model_table,
        transform_result=transform_result,
        target_logit_transformed=target_logit_transformed,
        stationarity_report=stationarity_report,
    )
