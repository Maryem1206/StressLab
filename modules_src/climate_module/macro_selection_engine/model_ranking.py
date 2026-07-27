"""
model_ranking.py
================
Affiche un tableau de tous les modèles estimés et laisse l'utilisateur
choisir le modèle satellite.

Colonnes du tableau :
    Rang | Modèle | Variables | R² | RMSE | p-value | Signe OK | AIC

Pas de score composite automatique — l'utilisateur décide.

p-value : p-value maximale parmi les coefficients non-intercept
          (le "pire" coefficient — si même celui-là est significatif,
          tout le modèle l'est).

Signe OK : nombre de variables dont le signe du coefficient respecte
             le prior économique, au format 'n/total' (ex: '3/3', '2/3').
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .model_estimation import FittedModel
from .utils import EngineConfig, expected_sign, get_logger

log = get_logger(__name__)


def _econ_score(model: FittedModel, risk_type: str) -> str:
    """Nombre de variables avec signe économique correct, format 'n/total'."""
    if not model.coefficients:
        return "0/0"
    ok = total = 0
    for var, coef in model.coefficients.items():
        if var == "const":
            continue
        exp = expected_sign(var, risk_type)
        if exp == 0:
            # Signe neutre : on compte 0.5 → arrondi à l'entier le plus proche
            ok += 0.5
            total += 1
        else:
            total += 1
            if np.sign(coef) == exp:
                ok += 1
    n_ok = round(ok)
    return f"{n_ok}/{total}" if total > 0 else "0/0"


def _max_pvalue(model: FittedModel) -> float:
    """p-value maximale parmi les variables (hors intercept)."""
    pvals = [p for v, p in model.pvalues.items() if v != "const"]
    if not pvals:
        return np.nan
    return float(max(pvals))


def _sign_ratio(sens_eco_str: str) -> float:
    """Convertit '2/2' → 1.0, '1/2' → 0.5, '0/0' → 0.5 (neutre)."""
    try:
        n, d = sens_eco_str.split("/")
        return float(n) / float(d) if float(d) > 0 else 0.5
    except Exception:
        return 0.5


def _quality_score(row: dict, all_rows: list) -> float:
    """
    Score composite de qualité statistique — utilisé comme tri primaire
    quand stress_score=0 pour tous les modèles, ou comme tiebreaker sinon.

    Critères (tous normalisés [0,1]) :
      Q1 — R²              : plus élevé = mieux         (poids 0.35)
      Q2 — Signes éco      : ratio n_ok/total            (poids 0.25)
      Q3 — p-value         : plus basse = mieux          (poids 0.25)
      Q4 — AIC             : plus bas = mieux            (poids 0.15)

    La normalisation est relative à l'ensemble des modèles candidats.
    """
    # Extraire les valeurs de tous les modèles pour normalisation
    r2_vals  = [r.get("R2")      for r in all_rows if r.get("R2")      is not None]
    pv_vals  = [r.get("p_value") for r in all_rows if r.get("p_value") is not None]
    aic_vals = [r.get("AIC")     for r in all_rows if r.get("AIC")     is not None]

    def norm_asc(val, vals):
        """Normalise val dans vals : plus bas = 1.0 (meilleur)."""
        if not vals or val is None:
            return 0.0
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return 1.0
        return 1.0 - (val - mn) / (mx - mn)

    def norm_desc(val, vals):
        """Normalise val dans vals : plus haut = 1.0 (meilleur)."""
        if not vals or val is None:
            return 0.0
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return 1.0
        return (val - mn) / (mx - mn)

    q1 = norm_desc(row.get("R2"),      r2_vals)   # R² haut = bien
    q2 = _sign_ratio(str(row.get("sens_eco", "0/0")))  # signes éco
    q3 = norm_asc(row.get("p_value"),  pv_vals)   # p-value basse = bien
    q4 = norm_asc(row.get("AIC"),      aic_vals)  # AIC bas = bien

    return 0.35 * q1 + 0.25 * q2 + 0.25 * q3 + 0.15 * q4


_VERDICT_RANK: Dict[str, int] = {
    "VALIDATED":               0,
    "VALIDATED_WITH_WARNINGS": 1,
    "REJECTED":                2,
}


def build_model_table(models: List[FittedModel],
                      cfg: EngineConfig,
                      stress_scores: Optional[Dict[str, float]] = None,
                      verdicts: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
    """
    Construit le tableau de comparaison des modèles.

    Logique de tri (primaire → secondaire → tertiaire) :
    - verdict ASC (VALIDATED < VALIDATED_WITH_WARNINGS < REJECTED)
    - stress_score DESC (si disponible et non-nul)
    - quality_score DESC (R², signes, p-value, AIC)

    verdicts : {model_name: {"verdict": str, "n_fail": int, ...}} — issu de
    PostEstimationValidator (8/9 tests pour le tri de tournoi; B3.3 monotonie des
    scénarios vérifiée en Phase 1 sur le modèle sélectionné uniquement).
    """
    if not models:
        return pd.DataFrame()

    rows = []
    for m in models:
        _vd = verdicts.get(m.name, {}) if verdicts else {}
        row = {
            "set":           getattr(m, "variable_set", "orig"),
            "modele":        m.family,
            "variables":     ", ".join(v for v in m.variables),
            "R2":            round(m.r2, 4) if not np.isnan(m.r2) else None,
            "RMSE":          round(m.rmse, 6) if not np.isnan(m.rmse) else None,
            "p_value":       round(_max_pvalue(m), 4),
            "sens_eco":      _econ_score(m, cfg.risk_type),
            "AIC":           round(m.aic, 2) if not np.isnan(m.aic) else None,
            "verdict":       _vd.get("verdict",       "VALIDATED_WITH_WARNINGS"),
            "n_fail":        _vd.get("n_fail",         0),
            "n_warn":        _vd.get("n_warn",         0),
            "n_pass":        _vd.get("n_pass",         0),
            "verdict_scope": _vd.get("verdict_scope",  ""),
            "_name":         m.name,
        }
        if stress_scores is not None:
            row["stress_score"] = round(stress_scores.get(m.name, 0.0), 4)
        rows.append(row)

    # Calculer quality_score et verdict_rank pour chaque modèle
    for row in rows:
        row["_quality_score"] = _quality_score(row, rows)
        row["_verdict_rank"]  = _VERDICT_RANK.get(row.get("verdict", "VALIDATED_WITH_WARNINGS"), 1)

    df = pd.DataFrame(rows)

    # Logique de tri : verdict ASC → stress DESC → quality DESC
    has_stress = "stress_score" in df.columns
    all_stress_zero = has_stress and (df["stress_score"].fillna(0.0) == 0.0).all()

    if has_stress and not all_stress_zero:
        df = df.sort_values(
            ["_verdict_rank", "stress_score", "_quality_score"],
            ascending=[True, False, False],
            na_position="last",
        )
        sort_label = "verdict puis stress_score puis quality_score"
    else:
        df = df.sort_values(
            ["_verdict_rank", "_quality_score"],
            ascending=[True, False],
            na_position="last",
        )
        sort_label = "verdict puis quality_score (R², signes, p-value, AIC)"

    df = df.reset_index(drop=True)
    df.insert(0, "rang", range(1, len(df) + 1))

    log.debug("build_model_table: trié par %s.", sort_label)

    priority = ["rang", "set", "modele", "variables",
                "verdict", "n_fail", "n_warn", "n_pass", "verdict_scope",
                "stress_score", "R2", "RMSE", "p_value", "sens_eco", "AIC",
                "_name", "_quality_score", "_verdict_rank"]
    ordered = [c for c in priority if c in df.columns]
    df = df[ordered]

    return df


def display_model_table(table: pd.DataFrame) -> None:
    """Affiche le tableau de modèles de manière lisible en console."""
    if table.empty:
        print("\n  !  Aucun modele estime.")
        return

    has_stress = "stress_score" in table.columns
    has_quality = "_quality_score" in table.columns
    all_stress_zero = has_stress and (table["stress_score"].fillna(0.0) == 0.0).all()

    print("\n" + "=" * 110)
    print("  MODELES SATELLITES ESTIMES - Choisissez le modele a utiliser")
    print("=" * 110)

    stress_hdr  = f"{'Stress':>7} " if has_stress else ""
    quality_hdr = f"{'Quality':>8} " if has_quality else ""
    header = (f"  {'Rang':>4}  {'Modele':<14} {'Variables':<35} "
              f"{stress_hdr}{quality_hdr}{'R2':>8} {'RMSE':>10} {'p-value':>9} "
              f"{'Signe OK':>9} {'AIC':>10}")
    print(header)
    print("  " + "-" * 106)

    for _, row in table.iterrows():
        stress_str = ""
        if has_stress:
            v = row.get("stress_score")
            stress_str = f"{v:.4f} " if v is not None else "   N/A "

        quality_str = ""
        if has_quality:
            q = row.get("_quality_score")
            quality_str = f"{q:.4f} " if q is not None else "   N/A "

        r2_str   = f"{row['R2']:.4f}"   if row["R2"]   is not None else "  N/A "
        rmse_str = f"{row['RMSE']:.6f}" if row["RMSE"] is not None else "   N/A  "
        pv_str   = f"{row['p_value']:.4f}" if row["p_value"] is not None else " N/A "
        eco_str  = str(row["sens_eco"])
        aic_str  = f"{row['AIC']:.2f}"  if row["AIC"]  is not None else "   N/A  "

        vars_str = str(row["variables"])
        if len(vars_str) > 33:
            vars_str = vars_str[:30] + "..."

        set_str = str(row.get("set", "orig"))
        print(f"  {row['rang']:>4}  {set_str:<5} {row['modele']:<12} {vars_str:<33} "
              f"{stress_str}{quality_str}{r2_str:>8} {rmse_str:>10} {pv_str:>9} "
              f"{eco_str:>9} {aic_str:>10}")

    if has_stress and not all_stress_zero:
        print("\n  * Classe par stress_score (coherence NGFS) puis quality_score (R2, signes, p-value, AIC)")
    else:
        print("\n  * Stress=0 pour tous - classe par quality_score (R2, signes eco, p-value, AIC)")
        print("    -> Le rang 1 est le modele statistiquement le plus robuste.")
    print()


def select_model_interactive(
    models: List[FittedModel],
    table: pd.DataFrame,
    auto_select: bool = False,
    forced_rank: Optional[int] = None,
) -> FittedModel:
    """
    Affiche le tableau et demande à l'utilisateur de choisir un modèle.

    Parameters
    ----------
    models      : liste complète des FittedModel
    table       : DataFrame construit par build_model_table
    auto_select : si True, sélectionne automatiquement le rang 1 (mode batch)
    forced_rank : si défini (0-based), force le modèle de ce rang

    Returns
    -------
    FittedModel choisi par l'utilisateur
    """
    display_model_table(table)

    # Index rapide : name -> FittedModel
    model_by_name = {m.name: m for m in models}

    # Forced rank from web UI takes priority
    if forced_rank is not None and 0 <= forced_rank < len(table):
        chosen_name = table.iloc[forced_rank]["_name"]
        log.info("FORCED model rank #%d by user -> '%s'.", forced_rank + 1, chosen_name)
        print(f"  -> Modele force par l'utilisateur : rang #{forced_rank + 1} ({chosen_name})")
        return model_by_name[chosen_name]

    if auto_select:
        if table.empty or not models:
            raise RuntimeError(
                "Aucun modèle satellite estimé disponible. "
                "Vérifiez que la variable cible est exprimée en proportion (0–1) "
                "et que les données historiques contiennent suffisamment d'observations."
            )
        chosen_name = table.iloc[0]["_name"]
        log.info("Auto-select: rang #1 -> '%s'.", chosen_name)
        print(f"  -> Mode automatique : rang #1 selectionne ({chosen_name})")
        return model_by_name[chosen_name]

    while True:
        try:
            choice = input("  Entrez le numero de rang du modele choisi : ").strip()
            rank = int(choice)
            if 1 <= rank <= len(table):
                chosen_name = table.loc[table["rang"] == rank, "_name"].iloc[0]
                chosen = model_by_name[chosen_name]
                log.info("User selected model: rang #%d -> '%s'.", rank, chosen.name)
                print(f"\n  OK Modele selectionne : {chosen.family} "
                      f"[{', '.join(chosen.variables)}]")
                return chosen
            else:
                print(f"  !  Rang invalide. Choisissez entre 1 et {len(table)}.")
        except ValueError:
            print("  !  Entrez un nombre entier.")
        except KeyboardInterrupt:
            # Fallback to rank #1
            chosen_name = table.iloc[0]["_name"]
            log.info("Interrupted — defaulting to rang #1: '%s'.", chosen_name)
            return model_by_name[chosen_name]


# ──────────────────────────────────────────────────────────────────────
# Sélection interactive des scénarios (adverse, severe)
# ──────────────────────────────────────────────────────────────────────

def select_scenario_pair_interactive(
    sel_result: Any,
    pd_matrix:  Dict[str, "pd.Series"],
    baseline_name: str,
    auto_accept: bool = False,
    fallback_adverse: Optional[str] = None,
    fallback_severe:  Optional[str] = None,
) -> Tuple[str, str]:
    """
    Explorateur interactif de combinaisons (adverse, severe).

    L'utilisateur parcourt toutes les paires disponibles, voit la
    trajectoire PD de chacune depuis pd_matrix (déjà calculé, pas de
    recalcul), et confirme la combinaison qui lui convient.

    Parameters
    ----------
    sel_result    : ScenarioPairResult issu de find_optimal_scenario_pair
    pd_matrix     : {scenario_name: pd.Series} — projections déjà calculées
    baseline_name : nom du scénario Baseline
    auto_accept   : si True, accepte la sélection automatique sans prompt
    fallback_adverse, fallback_severe : paire déjà déterminée par l'appelant
        (ex. le fallback "2 scénarios les plus déviés" de run_three_scenarios)
        à utiliser en mode auto si sel_result n'a pas de paire valide. Évite
        de tomber dans la boucle input() ci-dessous quand auto_accept=True
        mais que la sélection auto a échoué (sel_result.valid=False) —
        cette boucle bloque indéfiniment un run web/batch (aucun terminal
        attaché ne peut jamais répondre au prompt).

    Returns
    -------
    (adverse_name, severe_name) : tuple de la paire choisie
    """
    # ── Construire la liste complète des paires à explorer ─────────────
    ranking_df = getattr(sel_result, "ranking_df", pd.DataFrame())
    auto_adverse = getattr(sel_result, "adverse", None)
    auto_severe  = getattr(sel_result, "severe",  None)

    # Si ranking_df vide (auto-relaxation ou last-resort), construire
    # toutes les paires possibles depuis pd_matrix
    stressed = [k for k in pd_matrix if k != baseline_name]
    if ranking_df is None or ranking_df.empty:
        from itertools import permutations as _perms
        rows = [{"rank": i+1, "adverse": a, "severe": s, "score": None}
                for i, (a, s) in enumerate(_perms(stressed, 2))
                if a != s]
        ranking_df = pd.DataFrame(rows)
    else:
        ranking_df = ranking_df.copy().reset_index(drop=True)
        ranking_df.insert(0, "rank", range(1, len(ranking_df) + 1))

    n_pairs = len(ranking_df)

    # ── Mode auto : ne JAMAIS bloquer sur input() quand auto_accept=True.
    # Avant ce fix, si sel_result.valid=False (auto_adverse/severe=None),
    # le code tombait dans la boucle interactive ci-dessous même en mode
    # web/batch (auto_accept=True) — un run Phase 1 pouvait alors rester
    # bloqué indéfiniment sur "Entrez le rang a explorer...", en attente
    # d'un stdin qu'aucune requête HTTP ne peut jamais fournir.
    if auto_accept:
        if auto_adverse and auto_severe:
            log.info("Auto-accept: adverse='%s'  severe='%s'.",
                     auto_adverse, auto_severe)
            return auto_adverse, auto_severe
        if fallback_adverse and fallback_severe:
            log.warning(
                "Auto-accept: pas de paire valide depuis sel_result -- "
                "utilisation du fallback appelant: adverse='%s' severe='%s'.",
                fallback_adverse, fallback_severe,
            )
            return fallback_adverse, fallback_severe
        if n_pairs > 0:
            first = ranking_df.iloc[0]
            log.warning(
                "Auto-accept: pas de paire valide -- utilisation du rang #1 "
                "du classement: adverse='%s' severe='%s'.",
                first["adverse"], first["severe"],
            )
            return str(first["adverse"]), str(first["severe"])
        if len(stressed) >= 2:
            log.warning(
                "Auto-accept: aucune paire classée -- dernier recours "
                "(2 premiers scénarios disponibles): adverse='%s' severe='%s'.",
                stressed[0], stressed[1],
            )
            return stressed[0], stressed[1]
        log.error(
            "Auto-accept: aucune paire de scénarios disponible (0 candidat) "
            "-- retour (None, None)."
        )
        return None, None

    # ── Affichage de l'en-tête (uniquement atteint si auto_accept=False) ──
    _print_separator()
    print("  EXPLORATION DES SCENARIOS - Essayez toutes les combinaisons")
    _print_separator()

    if auto_adverse and auto_severe:
        score_str = (f"  score={sel_result.score:.4f}"
                     if getattr(sel_result, "score", None) else "")
        print(f"  Selection auto    : adverse='{auto_adverse}'"
              f"  severe='{auto_severe}'{score_str}")
    print()

    # ── Afficher le classement complet ──────────────────────────────────
    _print_ranking_table(ranking_df, auto_adverse, auto_severe)

    # ── Afficher la PD baseline ─────────────────────────────────────────
    if baseline_name in pd_matrix:
        b = pd_matrix[baseline_name].dropna()
        print(f"\n  Baseline '{baseline_name}' :"
              f"  mean={b.mean():.4f}  min={b.min():.4f}  max={b.max():.4f}")

    # ── Boucle d'exploration ─────────────────────────────────────────────
    current_adverse = auto_adverse
    current_severe  = auto_severe

    while True:
        print()
        print(f"  Entrez le rang a explorer [1-{n_pairs}],"
              f" ou ENTREE pour confirmer la selection actuelle"
              f" ({current_adverse or '?'} / {current_severe or '?'}) :")

        try:
            raw = input("  > ").strip()

            # ENTREE sans valeur -> confirmer
            if raw == "":
                if current_adverse and current_severe:
                    log.info("User confirmed: adverse='%s'  severe='%s'.",
                             current_adverse, current_severe)
                    print(f"\n  OK Combinaison confirmee :"
                          f" adverse='{current_adverse}'"
                          f"  severe='{current_severe}'")
                    return current_adverse, current_severe
                else:
                    print("  !  Aucune combinaison selectionnee. "
                          "Entrez un rang.")
                    continue

            rank = int(raw)
            if not (1 <= rank <= n_pairs):
                print(f"  !  Rang invalide. Choisissez entre 1 et {n_pairs}.")
                continue

            # Recuperer la paire choisie
            row = ranking_df[ranking_df["rank"] == rank].iloc[0]
            adv_name = str(row["adverse"])
            sev_name = str(row["severe"])
            score    = row.get("score")

            current_adverse = adv_name
            current_severe  = sev_name

            # Afficher la trajectoire PD de cette paire
            _print_trajectory(
                pd_matrix=pd_matrix,
                baseline_name=baseline_name,
                adverse_name=adv_name,
                severe_name=sev_name,
                score=score,
            )

        except ValueError:
            print("  !  Entrez un nombre entier ou ENTREE pour confirmer.")
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C -> accepter la selection auto
            fallback_adv = current_adverse or auto_adverse
            fallback_sev = current_severe  or auto_severe
            if fallback_adv and fallback_sev:
                log.info("Interrupted -- using: adverse='%s'  severe='%s'.",
                         fallback_adv, fallback_sev)
                return fallback_adv, fallback_sev
            # Dernier recours : 2 premiers scénarios disponibles
            if len(stressed) >= 2:
                return stressed[0], stressed[1]
            raise


# ── Helpers d'affichage ───────────────────────────────────────────────

def _print_separator() -> None:
    print("\n" + "=" * 62)


def _print_ranking_table(
    ranking_df: "pd.DataFrame",
    auto_adverse: str,
    auto_severe:  str,
) -> None:
    """Affiche le tableau des paires disponibles."""
    print(f"  {'Rang':>4}  {'Adverse':<28}  {'Severe':<28}  {'Score':>7}")
    print(f"  {'----':>4}  {'-'*28}  {'-'*28}  {'-'*7}")
    for _, row in ranking_df.iterrows():
        rank = int(row["rank"])
        adv  = str(row["adverse"])
        sev  = str(row["severe"])
        sc   = row.get("score")
        score_s = f"{sc:.4f}" if sc is not None and not _is_nan(sc) else "  N/A "
        marker = "  *" if adv == auto_adverse and sev == auto_severe else ""
        print(f"  {rank:>4}  {adv:<28}  {sev:<28}  {score_s:>7}{marker}")


def _print_trajectory(
    pd_matrix:     Dict[str, "pd.Series"],
    baseline_name: str,
    adverse_name:  str,
    severe_name:   str,
    score:         Any = None,
) -> None:
    """Affiche la trajectoire PD (B, A, S) pour une paire."""
    b_ser = pd_matrix.get(baseline_name, pd.Series(dtype=float))
    a_ser = pd_matrix.get(adverse_name,  pd.Series(dtype=float))
    s_ser = pd_matrix.get(severe_name,   pd.Series(dtype=float))

    all_years = sorted(
        set(b_ser.index) | set(a_ser.index) | set(s_ser.index)
    )

    score_str = f"  (score={score:.4f})" if score and not _is_nan(score) else ""
    print(f"\n  Rang selectionne : adverse='{adverse_name}'"
          f"  severe='{severe_name}'{score_str}")
    print(f"\n  {'Annee':>6}  {'Baseline':>10}  {'Adverse':>10}  "
          f"{'Severe':>10}  {'Ordre':>6}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")

    fmt = lambda v: f"{v:.4f}" if v is not None and not _is_nan(v) else "  N/A "
    for yr in all_years:
        b = b_ser.get(yr)
        a = a_ser.get(yr)
        s = s_ser.get(yr)
        ok = ""
        if b is not None and a is not None and s is not None:
            if not (_is_nan(b) or _is_nan(a) or _is_nan(s)):
                ok = "OK" if b <= a <= s else "X"
        print(f"  {yr:>6}  {fmt(b):>10}  {fmt(a):>10}  {fmt(s):>10}  {ok:>6}")

    # Resume
    if not a_ser.empty and not b_ser.empty:
        delta = a_ser.mean() - b_ser.mean()
        print(f"\n  Adverse : mean={a_ser.mean():.4f}  d_baseline={delta:+.4f}")
    if not s_ser.empty and not b_ser.empty:
        delta = s_ser.mean() - b_ser.mean()
        print(f"  Severe  : mean={s_ser.mean():.4f}  d_baseline={delta:+.4f}")


def _is_nan(v: Any) -> bool:
    """Retourne True si v est NaN ou None."""
    import math
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return v is None
