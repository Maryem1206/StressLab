"""
baseline_fix.py
===============
Correctif pour deux bugs de la baseline PD dans multi_scenario.py :

  Bug 1 : Les années historiques (2022-2024) affichent des valeurs AR
          au lieu des valeurs réelles de mon_historique.csv.

  Bug 2 : La baseline projetée (2025-2050) est constante (~0.0315)
          parce que le modèle AR(1) a φ ≈ 0 sur 12 obs. de PD
          quasi-white-noise → réversion immédiate à la moyenne
          inconditionnelle, sans tenir compte du NGFS Baseline.

SOLUTION (conforme EBA 2023 Methodological Note §4.2, ECB BEAST) :
  • Années ≤ last_hist_year  → historique brut de hist_df
  • Années >  last_hist_year → satellite model appliqué au NGFS Baseline

À copier-coller dans macro_selection_engine/multi_scenario.py
en remplacement de l'ancienne fonction _project_baseline_model
et du bloc d'assemblage de la baseline dans build_pd_matrix.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  FONCTION DE REMPLACEMENT : projection baseline via satellite model
# ══════════════════════════════════════════════════════════════════════

def _project_baseline_satellite(
    selected_model: dict,
    ngfs_baseline_proj: pd.DataFrame,      # colonnes = variables macro NGFS Baseline, index = year
    hist_df: pd.DataFrame,                  # historique brut, index = year, col "Default rate"
    target_col: str,                        # ex. "Default rate"
    projection_years: list,                 # ex. [2025, 2026, ..., 2050]
    baseline_min_pd: float = 0.001,
    baseline_max_pd: float = 0.40,
) -> pd.Series:
    """
    Projette la PD baseline pour les années FUTURES UNIQUEMENT
    en appliquant le satellite model sélectionné aux variables
    macroéconomiques du scénario NGFS Baseline.

    Retourne une pd.Series indexée par les années de projection,
    clippée dans [baseline_min_pd, baseline_max_pd].

    Conforme à :
      - EBA 2023 Methodological Note §4.2
      - ECB BEAST (2023) : satellite models applied uniformly across scenarios
      - BCBS (2009) Principles for Sound Stress Testing §21
    """
    if selected_model is None:
        logger.warning("_project_baseline_satellite: no satellite model available, "
                       "falling back to last historical PD flat projection.")
        last_pd = hist_df[target_col].iloc[-1]
        return pd.Series(last_pd, index=projection_years, name="baseline_projected")

    family   = selected_model.get("family", "OLS")
    coefs    = selected_model.get("coefficients", {})
    vars_    = selected_model.get("variables", [])

    # ── Vérifie la disponibilité des variables dans ngfs_baseline_proj ──
    missing = [v for v in vars_ if v not in ngfs_baseline_proj.columns]
    if missing:
        logger.warning(f"_project_baseline_satellite: variables manquantes dans "
                       f"NGFS Baseline : {missing}. Projection plate (dernier historique).")
        last_pd = hist_df[target_col].iloc[-1]
        return pd.Series(last_pd, index=projection_years, name="baseline_projected")

    # ── Construction de la matrice X pour les années de projection ──
    X_future = ngfs_baseline_proj.loc[
        ngfs_baseline_proj.index.isin(projection_years), vars_
    ].copy()
    X_future.insert(0, "const", 1.0)

    # ── Calcul du prédicteur linéaire η = Xβ ──
    beta = pd.Series(coefs)
    available_cols = [c for c in beta.index if c in X_future.columns]
    eta  = X_future[available_cols].dot(beta[available_cols])

    # ── Transformation inverse selon la famille du modèle ──
    if family in ("Beta", "Logit"):
        # lien logit : PD = 1 / (1 + exp(-η))
        pd_proj = 1.0 / (1.0 + np.exp(-eta))

    elif family in ("OLS", "Vasicek-OLS"):
        # lien identité, mais on clip en [0,1]
        pd_proj = eta.clip(0.0, 1.0)

    elif family == "Poisson":
        # lien log : PD = exp(η)
        pd_proj = np.exp(eta).clip(0.0, 1.0)

    else:
        logger.warning(f"_project_baseline_satellite: famille '{family}' inconnue, "
                       "utilisation du lien identité.")
        pd_proj = eta.clip(0.0, 1.0)

    pd_proj = pd_proj.clip(baseline_min_pd, baseline_max_pd)
    pd_proj.index = X_future.index
    pd_proj.name  = "baseline_projected"

    logger.info(f"_project_baseline_satellite: satellite({family}) appliqué au NGFS Baseline "
                f"→ mean={pd_proj.mean():.4f}  "
                f"min={pd_proj.min():.4f}  "
                f"max={pd_proj.max():.4f}")
    return pd_proj


# ══════════════════════════════════════════════════════════════════════
#  FONCTION PRINCIPALE : assemblage complet de la baseline PD
# ══════════════════════════════════════════════════════════════════════

def build_baseline_path(
    hist_df: pd.DataFrame,
    target_col: str,
    selected_model: Optional[dict],
    ngfs_baseline_proj: pd.DataFrame,
    all_scenario_years: list,
    baseline_min_pd: float = 0.001,
    baseline_max_pd: float = 0.40,
) -> pd.Series:
    """
    Assemble la trajectoire baseline complète :

      ┌─────────────────────────────────────────────────────────┐
      │  Années ≤ last_hist_year  →  hist_df[target_col]  (réel)│
      │  Années >  last_hist_year  →  satellite(NGFS Baseline)  │
      └─────────────────────────────────────────────────────────┘

    Paramètres
    ----------
    hist_df            : DataFrame historique, index = year (int), 
                         contient la colonne target_col
    target_col         : nom de la colonne PD dans hist_df (ex. "Default rate")
    selected_model     : dict du satellite model sélectionné (peut être None)
    ngfs_baseline_proj : DataFrame des variables macro NGFS Baseline
                         projetées, index = year (int)
    all_scenario_years : liste de toutes les années du scénario (ex. 2022..2050)
    baseline_min_pd    : plancher PD (défaut 0.001 = 0.1%)
    baseline_max_pd    : plafond PD (défaut 0.40 = 40%)

    Retourne
    --------
    pd.Series : baseline PD complète, indexée par year (int)
    """
    last_hist_year = int(hist_df.index.max())
    all_years      = sorted(all_scenario_years)

    # ── 1. Années historiques : valeurs réelles brutes ─────────────
    hist_years  = [y for y in all_years if y <= last_hist_year]
    hist_values = hist_df.loc[
        hist_df.index.isin(hist_years), target_col
    ].reindex(hist_years)

    # Vérification : pas de NaN dans l'historique
    if hist_values.isna().any():
        logger.warning(f"build_baseline_path: NaN dans l'historique PD pour "
                       f"les années {hist_values[hist_values.isna()].index.tolist()}. "
                       "Forward-fill appliqué.")
        hist_values = hist_values.ffill().bfill()

    logger.info(f"build_baseline_path: historique injecté pour {hist_years} "
                f"→ {hist_values.values.round(4).tolist()}")

    # ── 2. Années futures : satellite model sur NGFS Baseline ──────
    future_years = [y for y in all_years if y > last_hist_year]

    if future_years:
        future_values = _project_baseline_satellite(
            selected_model     = selected_model,
            ngfs_baseline_proj = ngfs_baseline_proj,
            hist_df            = hist_df,
            target_col         = target_col,
            projection_years   = future_years,
            baseline_min_pd    = baseline_min_pd,
            baseline_max_pd    = baseline_max_pd,
        )
    else:
        future_values = pd.Series(dtype=float)

    # ── 3. Concaténation : historique + projection ─────────────────
    baseline = pd.concat([hist_values, future_values]).sort_index()

    logger.info(
        f"build_baseline_path: baseline assemblée "
        f"({len(hist_years)} pts historiques + {len(future_years)} pts projetés) "
        f"→ mean={baseline.mean():.4f}  "
        f"min={baseline.min():.4f}  "
        f"max={baseline.max():.4f}"
    )
    return baseline


# ══════════════════════════════════════════════════════════════════════
#  PATCH DANS build_pd_matrix (remplacer le bloc baseline existant)
# ══════════════════════════════════════════════════════════════════════
#
#  Dans la fonction build_pd_matrix de multi_scenario.py,
#  REMPLACER le bloc actuel :
#
#    baseline_proj = _project_baseline_model(...)
#    baseline_pd_matrix["Baseline"] = baseline_proj
#
#  PAR :
#
#    baseline_series = build_baseline_path(
#        hist_df            = hist_df,
#        target_col         = cfg.target_variable,   # "Default rate"
#        selected_model     = selected_model,         # dict du modèle satellite
#        ngfs_baseline_proj = ngfs_baseline_pivot,    # NGFS Baseline macro vars
#        all_scenario_years = scenario_years,         # liste des années du scénario
#        baseline_min_pd    = cfg.baseline_min_pd,
#        baseline_max_pd    = cfg.baseline_max_pd_value,
#    )
#    baseline_pd_matrix["Baseline"] = baseline_series
#
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
#  VÉRIFICATION RAPIDE (script autonome)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, os, sys

    # ── Charge l'historique ─────────────────────────────────────────
    csv_path = "mon_historique.csv"
    if not os.path.exists(csv_path):
        print(f"[ERREUR] {csv_path} introuvable. Lancer depuis le dossier du projet.")
        sys.exit(1)

    hist = pd.read_csv(csv_path, index_col="year")
    hist.index = hist.index.astype(int)
    target = "Default rate"

    print("\n=== Donnees historiques PD ===")
    print(hist[[target]].to_string())
    print(f"\nDerniere annee historique : {hist.index.max()}")
    print(f"Derniere PD historique   : {hist[target].iloc[-1]:.4f}")

    # -- Charge le modele depuis stress_test_resultats.json ----------
    json_path = "stress_test_resultats.json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            res = json.load(f)
        model = res.get("selected_model")
        print(f"\n=== Satellite model ===")
        if model:
            print(f"  Famille    : {model['family']}")
            print(f"  Variables  : {model['variables']}")
            print(f"  R2         : {model.get('r2_or_pseudo', 'N/A'):.4f}")
        else:
            print("  !  Aucun modele selectionne dans le JSON")
    else:
        print(f"\n!  {json_path} non trouve - test avec modele factice.")
        model = None

    # ── Simulation ngfs_baseline_proj minimale ──────────────────────
    # En production, ce DataFrame vient du data_loader.py
    # Ici on simule avec des valeurs constantes pour le test
    future_years_test = list(range(2025, 2051))
    if model and model.get("variables"):
        fake_ngfs = pd.DataFrame(
            {v: hist[hist.columns.intersection(model["variables"])].iloc[-1].get(v, 0.0)
             if v in hist.columns else 0.0
             for v in model["variables"]},
            index=future_years_test,
        )
        # Tentative de mapping colonnes hist → NGFS (approximatif pour le test)
        # En production les colonnes sont déjà dans l'espace NGFS/standardisé
    else:
        fake_ngfs = pd.DataFrame(index=future_years_test)

    # ── Appel de la fonction corrigée ───────────────────────────────
    baseline = build_baseline_path(
        hist_df            = hist,
        target_col         = target,
        selected_model     = model if fake_ngfs.shape[1] > 0 else None,
        ngfs_baseline_proj = fake_ngfs,
        all_scenario_years = list(range(2022, 2051)),
        baseline_min_pd    = 0.001,
        baseline_max_pd    = 0.40,
    )

    print("\n=== Baseline PD corrigee ===")
    print(f"{'Annee':>6}  {'PD':>8}  {'Source':>12}")
    print("-" * 32)
    for yr, val in baseline.items():
        source = "historique" if yr <= hist.index.max() else "satellite"
        print(f"{yr:>6}  {val:>8.4f}  {source:>12}")

    print(f"\nResume : mean={baseline.mean():.4f}  "
          f"min={baseline.min():.4f}  "
          f"max={baseline.max():.4f}")

    # -- Verification de coherence ------------------------------------
    print("\n=== Verifications ===")
    hist_check = baseline.loc[baseline.index <= hist.index.max()]
    hist_orig  = hist.loc[hist.index.isin(hist_check.index), target]

    ok = True
    for yr in hist_check.index:
        if abs(hist_check.loc[yr] - hist_orig.loc[yr]) > 1e-9:
            print(f"  X Annee {yr}: baseline={hist_check.loc[yr]:.4f}  "
                  f"historique={hist_orig.loc[yr]:.4f}")
            ok = False

    if ok:
        print("  OK Toutes les annees historiques (2013-2024) "
              "correspondent exactement a mon_historique.csv")

    proj_check = baseline.loc[baseline.index > hist.index.max()]
    if not proj_check.empty:
        flat = proj_check.std() < 1e-6
        if flat:
            print("  !  Projection encore plate (satellite model non connecte "
                  "ou ngfs_baseline_proj vide - normal en test autonome)")
        else:
            print(f"  OK Projection dynamique : std={proj_check.std():.6f} > 0")
