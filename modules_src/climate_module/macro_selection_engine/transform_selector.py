"""
transform_selector.py
=====================
Sélectionne, pour chaque variable macro historique, la meilleure transformation
parmi {level, lag1, growth, cycle} — celle qui maximise |corrélation de Pearson|
avec la cible (PD, Default rate, etc.).

Convention de nommage : les variables retenues portent un suffixe "__method"
(ex: "cpi_inflation__growth"), rejoué à l'identique sur les trajectoires NGFS
par replay_transforms_on_ngfs() pour rester cohérent entre estimation et
projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .baseline_projector import _adf, _kpss_test
from .utils import EngineConfig, get_logger

log = get_logger(__name__)

_MAX_DIFF_ORDER = 2  # au-delà de I(2), la variable est économiquement peu interprétable


# ─────────────────────────────────────────────────────────────────────
# Transformations disponibles
# ─────────────────────────────────────────────────────────────────────

TRANSFORMATIONS = ["level", "lag1", "growth", "cycle"]
_EWMA_SPAN = 3


def apply_transform(series: pd.Series, method: str,
                    ewma_span: int = _EWMA_SPAN) -> pd.Series:
    """
    Apply one transform to a raw macro series.

    Parameters
    ----------
    series    : série temporelle brute
    method    : "level" | "lag1" | "growth" | "cycle"
    ewma_span : fenêtre EWMA pour la composante cyclique

    Returns
    -------
    série transformée (même index). lag1/growth n'ont pas de valeur t-1 pour
    la toute première observation — plutôt que de laisser un NaN (qui forcerait
    à supprimer cette observation partout en aval, coûteux sur un échantillon
    déjà court), ce point unique est comblé par bfill() : on lui attribue la
    valeur du point suivant, ce qui préserve la taille d'échantillon sans
    affecter aucune autre observation.
    """
    s = series.astype(float)
    if method == "level":
        return s.copy()
    elif method == "lag1":
        return s.shift(1).bfill()
    elif method == "growth":
        denom = s.shift(1).abs().clip(lower=1e-9)
        return (s.diff() / denom).bfill()
    elif method == "cycle":
        trend = s.ewm(span=ewma_span, min_periods=1).mean()
        return s - trend
    else:
        raise ValueError(f"Unknown transform: {method}")


def _pearson_r(x: pd.Series, y: pd.Series) -> float:
    """Corrélation de Pearson sur l'intersection non-NaN."""
    mask = x.notna() & y.notna()
    if mask.sum() < 4:
        return np.nan
    if x[mask].std() == 0 or y[mask].std() == 0:
        return np.nan
    return float(np.corrcoef(x[mask].values, y[mask].values)[0, 1])


_KNOWN_METHODS = frozenset(TRANSFORMATIONS)


def parse_transformed_colname(col: str):
    """
    Split a "{hist_col}__{method}" feature name back into its parts.

    Falls back to (col, "level") when col carries no recognised "__method"
    suffix — e.g. a raw historical column name passed through unchanged.

    Returns
    -------
    (hist_col, method)
    """
    if "__" in col:
        hist_col, _, method = col.rpartition("__")
        if method in _KNOWN_METHODS:
            return hist_col, method
    return col, "level"


# ─────────────────────────────────────────────────────────────────────
# Dataclasses de résultat
# ─────────────────────────────────────────────────────────────────────

@dataclass
class VariableBestTransform:
    """Meilleure transformation trouvée pour une variable historique."""
    variable: str          # nom de la colonne historique originale
    best_method: str       # "level" | "lag1" | "growth" | "cycle"
    best_abs_r: float      # |r| avec la cible
    all_results: Dict[str, float] = field(default_factory=dict)
    # all_results : {method -> abs_r} pour traçabilité complète
    diff_order: int = 0
    # diff_order : nb de différenciations supplémentaires appliquées APRÈS le
    # choix de best_method, pour atteindre une stationnarité réellement
    # vérifiée (ADF+KPSS sur la série finale, pas une hypothèse sur le niveau
    # brut). 0 si déjà stationnaire sans différenciation additionnelle.
    final_stationary: bool = False
    final_adf_pval: float = float("nan")


@dataclass
class TransformSelectionResult:
    """
    Résultat complet de la sélection de transformations.

    X_transformed : DataFrame avec colonnes nommées '{hist_col}__{method}'
                    aligné sur y (dropna appliqué). Les valeurs reflètent déjà
                    toute différenciation supplémentaire appliquée pour
                    atteindre la stationnarité (voir diff_order).
    best_per_var  : {hist_col -> VariableBestTransform}
    rejected_vars : variables pour lesquelles aucune transformation
                    n'atteint cfg.min_abs_correlation
    transform_table : tableau complet de toutes les transformations
                      testées (pour diagnostic/export JSON)
    final_stationarity : {hist_col -> {adf_stationary, adf_pval,
                          kpss_stationary, order, method, final_stationary}}
                          — verdict RÉEL vérifié sur la série effectivement
                          utilisée par le modèle (post-sélection par
                          corrélation + différenciation itérative), pas une
                          hypothèse calculée sur le niveau brut avant sélection.
    """
    X_transformed: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_per_var: Dict[str, VariableBestTransform] = field(default_factory=dict)
    rejected_vars: List[str] = field(default_factory=list)
    transform_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_stationarity: Dict[str, Dict] = field(default_factory=dict)


def _stationarity_test(series: pd.Series) -> Tuple[bool, float, bool]:
    """ADF + KPSS sur une série (dropna appliqué). Retourne (adf_ok, adf_pval, kpss_ok)."""
    arr = series.dropna().values
    adf_ok, adf_pval = _adf(arr)
    kpss_ok = _kpss_test(arr)
    return adf_ok, adf_pval, kpss_ok


# ─────────────────────────────────────────────────────────────────────
# Fonction principale : sélection des meilleures transformations
# ─────────────────────────────────────────────────────────────────────

def select_best_transforms(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: EngineConfig,
) -> TransformSelectionResult:
    """
    Pour chaque variable macro, teste {level, lag1, growth, cycle} et retient
    la transformation maximisant |corrélation de Pearson| avec y.

    Parameters
    ----------
    X   : DataFrame des variables macro historiques (niveaux bruts)
    y   : Series de la cible (PD, Default rate, etc.)
    cfg : EngineConfig (utilise cfg.min_abs_correlation)

    Returns
    -------
    TransformSelectionResult — une transformation par variable, celle qui
    corrèle le mieux avec la cible.
    """
    log.info(
        "Transform selection: testing %d transforms × %d variables "
        "(min |r|=%.2f).",
        len(TRANSFORMATIONS), X.shape[1], cfg.min_abs_correlation,
    )

    best_per_var: Dict[str, VariableBestTransform] = {}
    rejected: List[str] = []
    transformed_cols: Dict[str, pd.Series] = {}
    table_rows: List[Dict] = []
    final_stationarity: Dict[str, Dict] = {}

    for col in X.columns:
        raw = X[col].astype(float)
        results: Dict[str, float] = {}
        series_by_method: Dict[str, pd.Series] = {}

        for method in TRANSFORMATIONS:
            transformed = apply_transform(raw, method)
            r = _pearson_r(transformed, y)
            abs_r = abs(r) if not np.isnan(r) else np.nan
            results[method] = abs_r
            series_by_method[method] = transformed

        valid_results = {m: v for m, v in results.items() if not np.isnan(v)}
        if not valid_results:
            for method in TRANSFORMATIONS:
                table_rows.append({
                    "variable": col, "transform": method, "abs_r": np.nan,
                    "best": method == "level", "selected": False,
                })
            log.debug("Variable '%s': aucune transformation exploitable → rejetée.", col)
            rejected.append(col)
            continue

        best_method = max(valid_results, key=valid_results.get)
        best_abs_r = valid_results[best_method]
        selected = best_abs_r >= cfg.min_abs_correlation

        for method in TRANSFORMATIONS:
            table_rows.append({
                "variable": col, "transform": method, "abs_r": results[method],
                "best": method == best_method,
                "selected": selected if method == best_method else False,
            })

        if not selected:
            log.debug(
                "Variable '%s': best |r|=%.3f (%s) < %.2f → rejected.",
                col, best_abs_r, best_method, cfg.min_abs_correlation,
            )
            rejected.append(col)
            continue

        # ── Vérification RÉELLE de stationnarité + différenciation itérative ──
        # Le choix de best_method ci-dessus est fondé uniquement sur la
        # corrélation avec la cible. On teste maintenant, par ADF+KPSS, si la
        # série EFFECTIVEMENT retenue est stationnaire — et si ce n'est pas le
        # cas, on différencie (1ère puis 2ème fois si besoin) jusqu'à ce
        # qu'elle le devienne, plutôt que de supposer sa stationnarité à
        # partir du niveau brut avant sélection.
        working = series_by_method[best_method].copy()
        diff_order = 0
        adf_ok, adf_pval, kpss_ok = _stationarity_test(working)
        while not (adf_ok and kpss_ok) and diff_order < _MAX_DIFF_ORDER:
            working = working.diff().bfill()
            diff_order += 1
            adf_ok, adf_pval, kpss_ok = _stationarity_test(working)
        final_stationary = adf_ok and kpss_ok

        if diff_order > 0:
            log.info(
                "Variable '%s' (%s): non-stationnaire après sélection par "
                "corrélation → différenciation d=%d appliquée (ADF p=%.4f, "
                "KPSS=%s) → %s",
                col, best_method, diff_order, adf_pval,
                "✓" if kpss_ok else "✗",
                "stationnaire" if final_stationary
                else f"TOUJOURS NON-STATIONNAIRE (d={_MAX_DIFF_ORDER} max atteint)",
            )

        fname = f"{col}__{best_method}"
        best_per_var[col] = VariableBestTransform(
            variable=col,
            best_method=best_method,
            best_abs_r=best_abs_r,
            all_results=results,
            diff_order=diff_order,
            final_stationary=final_stationary,
            final_adf_pval=adf_pval,
        )
        transformed_cols[fname] = working
        final_stationarity[col] = {
            "adf_stationary": bool(adf_ok),
            "adf_pval": round(float(adf_pval), 4),
            "kpss_stationary": bool(kpss_ok),
            "order": diff_order,
            "method": best_method,
            "final_stationary": bool(final_stationary),
        }

        log.debug(
            "Variable '%s': %s (d=%d) → |r|=%.3f ✓ (autres: %s)",
            col, best_method, diff_order, best_abs_r,
            {m: (round(v, 3) if not np.isnan(v) else None)
             for m, v in results.items() if m != best_method},
        )

    X_transformed = pd.DataFrame(transformed_cols)
    if not X_transformed.empty:
        y_aligned = y.loc[X_transformed.index]
        mask = y_aligned.notna()
        X_transformed = X_transformed[mask]

    transform_table = pd.DataFrame(table_rows)

    log.info(
        "Transform selection: %d variables retained, %d rejected "
        "(min |r|=%.2f). Méthodes retenues: %s",
        len(best_per_var), len(rejected), cfg.min_abs_correlation,
        {v.variable: v.best_method for v in best_per_var.values()},
    )
    n_still_ns = sum(1 for v in final_stationarity.values() if not v["final_stationary"])
    if n_still_ns:
        log.warning(
            "Stationnarité réelle (post-sélection): %d/%d variable(s) "
            "toujours non-stationnaire(s) après différenciation max (d=%d).",
            n_still_ns, len(final_stationarity), _MAX_DIFF_ORDER,
        )

    return TransformSelectionResult(
        X_transformed=X_transformed,
        best_per_var=best_per_var,
        rejected_vars=rejected,
        transform_table=transform_table,
        final_stationarity=final_stationarity,
    )


# ─────────────────────────────────────────────────────────────────────
# Replay des transformations sur les séries NGFS
# ─────────────────────────────────────────────────────────────────────

def replay_transforms_on_ngfs(
    ngfs_pivot: pd.DataFrame,
    best_per_var: Dict[str, VariableBestTransform],
    ngfs_to_hist: Dict[str, str],
    units: Dict[str, str],
    hist_anchor: Optional[pd.Series] = None,
    ngfs_baseline_pivot: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Convertit les déviations NGFS en inputs satellite selon le type d'unité,
    puis rejoue la même transformation ({level, lag1, growth, cycle}) que
    celle retenue en estimation pour chaque variable.

    Deux cas selon l'unité NGFS de la variable (reconstruction du NIVEAU
    WB-cohérent avant transformation) :

    Cas 1 — Abs. difference (variables en %, ex: taux d'intérêt, inflation)
        Les deux sources (WB historique et NGFS) sont en %.
        Stressed level = baseline_NGFS(t) + abs_diff(t)
        Si baseline_NGFS indisponible : fallback sur hist_anchor + abs_diff.

    Cas 2 — % difference (variables en Bn USD, ex: GDP, Imports, Exports)
        WB historique en USD courant ≠ NGFS en 2017 PPP USD → unités incompatibles.
        Solution : appliquer le choc relatif sur l'ancre WB.
        stressed_WB(t) = anchor_WB × (1 + pct_diff(t) / 100)

    Le niveau ainsi reconstruit est ensuite transformé via apply_transform()
    avec la même méthode que celle retenue par select_best_transforms() pour
    cette variable. Pour lag1/growth/cycle, la valeur d'ancrage historique
    (hist_anchor) est préfixée comme point t-1 avant transformation puis
    retirée, afin que la première année de projection ne soit pas NaN.

    Parameters
    ----------
    ngfs_pivot          : DataFrame NGFS (index=year, columns=ngfs_var) — déviations
    best_per_var        : {hist_col: VariableBestTransform}
    ngfs_to_hist        : {ngfs_var: hist_col}
    units               : {ngfs_var: unit_str}
    hist_anchor         : dernières valeurs WB historiques (point t-1 pour le replay
                          des transformations à mémoire courte, et fallback Cas 1
                          si pas de baseline)
    ngfs_baseline_pivot : DataFrame NGFS Baseline (index=year, columns=var_base) — niveaux

    Returns
    -------
    DataFrame (index=year, columns='{hist_col}__{method}') prêt pour le satellite
    """
    # Inverser le mapping : hist_col → [ngfs_vars]
    hist_to_ngfs: Dict[str, List[str]] = {}
    for ngfs_var, hist_col in ngfs_to_hist.items():
        hist_to_ngfs.setdefault(hist_col, []).append(ngfs_var)

    result_cols: Dict[str, pd.Series] = {}

    for hist_col, bpv in best_per_var.items():
        ngfs_sources = hist_to_ngfs.get(hist_col, [])
        present = [v for v in ngfs_sources if v in ngfs_pivot.columns]

        if not present:
            log.warning("Replay: no NGFS column for hist_col='%s'. Skipping.", hist_col)
            continue

        # Série NGFS brute (déviation — moyenne si plusieurs sources)
        raw_ngfs = ngfs_pivot[present].mean(axis=1)

        # Détecter le type de déviation NGFS selon l'unité
        ref_unit = str(units.get(present[0], "")).strip()
        is_pct_diff = ref_unit.lower().startswith("% difference")
        is_abs_diff = ref_unit.lower().startswith("abs.")

        has_anchor = hist_anchor is not None and hist_col in hist_anchor.index
        anchor_val = float(hist_anchor[hist_col]) if has_anchor else None

        if is_pct_diff:
            # % difference (GDP, Imports, Exports en Bn USD 2017 PPP)
            # stressed_WB(t) = anchor_WB × (1 + pct_diff(t) / 100)
            if has_anchor:
                series_out = anchor_val * (1.0 + raw_ngfs / 100.0)
                log.info(
                    "Replay: '%s' %% diff → anchor_WB(%.4f) × (1 + pct/100) "
                    "[ex 2030: %.4f].",
                    present[0], anchor_val,
                    float(series_out.get(2030, float("nan"))),
                )
            else:
                series_out = 1.0 + raw_ngfs / 100.0
                log.warning("Replay: '%s' %% diff sans anchor WB → ratio brut.", present[0])

        elif is_abs_diff:
            # Abs. difference (taux en %, ex: lending_rate +0.54 pp)
            # stressed_WB(t) = anchor_WB + abs_diff(t)
            if has_anchor:
                series_out = raw_ngfs + anchor_val
                log.info(
                    "Replay: '%s' Abs. diff → anchor_WB(%.4f) + abs_diff "
                    "[ex 2030: %.4f].",
                    present[0], anchor_val,
                    float(series_out.get(2030, float("nan"))),
                )
            else:
                series_out = raw_ngfs
                log.warning("Replay: '%s' Abs. diff sans anchor WB → déviation brute.", present[0])
        else:
            series_out = raw_ngfs
            log.warning("Replay: '%s' unité inconnue '%s' → valeur brute.", present[0], ref_unit)

        # ── Rejouer la transformation retenue en estimation ──────────────────
        method = bpv.best_method
        series_out = series_out.sort_index()
        if method == "level":
            transformed = series_out
        elif has_anchor:
            # Préfixer l'ancre comme point t-1 pour éviter un NaN en tête de
            # projection (lag1/growth ont besoin d'une valeur précédente ;
            # cycle en profite pour amorcer l'EWMA).
            anchor_year = int(series_out.index.min()) - 1
            s_ext = pd.concat([
                pd.Series([anchor_val], index=[anchor_year]), series_out,
            ])
            transformed = apply_transform(s_ext, method).drop(index=anchor_year)
        else:
            transformed = apply_transform(series_out, method)
            log.warning(
                "Replay: '%s' transform=%s sans hist_anchor — la première "
                "année de projection peut être NaN.", hist_col, method,
            )

        # Rejouer la différenciation supplémentaire appliquée en estimation
        # pour atteindre la stationnarité réelle (voir bpv.diff_order).
        for _ in range(bpv.diff_order):
            transformed = transformed.diff().bfill()

        result_cols[f"{hist_col}__{method}"] = transformed

    if not result_cols:
        return pd.DataFrame()

    return pd.DataFrame(result_cols).dropna()
