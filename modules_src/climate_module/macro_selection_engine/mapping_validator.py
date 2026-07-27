"""
mapping_validator.py
====================
Interface de validation du mapping NGFS -> historique.

Expose validate_mapping() qui :
  1. Affiche le mapping automatique proposé
  2. Demande à l'utilisateur de valider / corriger chaque entrée
  3. Sauvegarde le mapping validé en YAML (réutilisable)
  4. Retourne un MappingResult corrigé prêt pour le pipeline

C'est un composant plateforme générique : fonctionne pour n'importe
quel dataset, n'importe quel pays, n'importe quel type de risque.
"""

from __future__ import annotations

import os
from typing import List

import yaml

from .mapping import MappingResult, map_variables
from .utils import EngineConfig, get_logger

log = get_logger(__name__)


def validate_mapping(
    ngfs_vars: List[str],
    hist_cols: List[str],
    cfg: EngineConfig,
    auto_accept: bool = False,
) -> MappingResult:
    """
    Propose le mapping automatique à l'utilisateur et permet de le corriger.

    Parameters
    ----------
    ngfs_vars   : variables NGFS sélectionnées par Stage 1
    hist_cols   : colonnes disponibles dans le fichier historique
    cfg         : EngineConfig (contient mapping_yaml_path)
    auto_accept : si True, skip la validation interactive (mode batch/CI)

    Returns
    -------
    MappingResult validé et corrigé par l'utilisateur
    """

    # ── Étape 1 : mapping automatique initial ──────────────────────
    auto_result = map_variables(ngfs_vars, hist_cols, cfg)

    if auto_accept:
        log.info("Mapping validation: auto_accept=True, skipping interactive.")
        return auto_result

    # ── Étape 2 : affichage du mapping proposé ─────────────────────
    _print_mapping_summary(auto_result, hist_cols)

    # ── Étape 3 : demande de validation ───────────────────────────
    answer = input(
        "\n  Voulez-vous corriger ce mapping ? [o/N] : "
    ).strip().lower()

    if answer not in ("o", "oui", "y", "yes"):
        log.info("Mapping accepted as-is by user.")
        _maybe_save_yaml(auto_result, cfg)
        return auto_result

    # ── Étape 4 : correction interactive ──────────────────────────
    corrected = _interactive_correction(auto_result, hist_cols)

    # ── Étape 5 : sauvegarde YAML ─────────────────────────────────
    _maybe_save_yaml(corrected, cfg)

    return corrected


# ─────────────────────────────────────────────────────────────────────
# Helpers privés
# ─────────────────────────────────────────────────────────────────────

def _print_mapping_summary(result: MappingResult, hist_cols: List[str]) -> None:
    """Affiche le mapping proposé de manière lisible."""
    print("\n" + "=" * 72)
    print("  VALIDATION DU MAPPING  -  NGFS variable -> colonne historique")
    print("=" * 72)

    if result.matches:
        print(f"\n  OK {len(result.matches)} variable(s) mappee(s) automatiquement :\n")
        for i, (ngfs, hist) in enumerate(result.matches.items(), 1):
            method = result.method_per_match.get(ngfs, "?")
            print(f"  [{i:>2}] {ngfs}")
            print(f"        -> {hist}   (methode: {method})")
    else:
        print("\n  !  Aucune variable mappee automatiquement.")

    if result.unmatched:
        print(f"\n  X {len(result.unmatched)} variable(s) non mappee(s) "
              f"(ignorees pour l'instant) :\n")
        for i, v in enumerate(result.unmatched, 1):
            print(f"  [{i:>2}] {v}")

    print(f"\n  Colonnes disponibles dans votre fichier historique :")
    for i, c in enumerate(hist_cols, 1):
        print(f"       {i:>2}. {c}")
    print()


def _interactive_correction(
    result: MappingResult,
    hist_cols: List[str],
) -> MappingResult:
    """
    Boucle interactive : l'utilisateur peut :
      - corriger un mapping existant (entrer un nouveau numéro de colonne)
      - mapper une variable non-mappée (choisir une colonne)
      - supprimer un mapping erroné (entrer 0)
      - passer (appuyer sur Entrée)
    """
    matches = dict(result.matches)
    method_per_match = dict(result.method_per_match)
    unmatched = list(result.unmatched)

    col_index = {str(i): c for i, c in enumerate(hist_cols, 1)}

    print("\n  -- Correction des mappings existants --------------------")
    print("  (Entree = garder tel quel | numero = changer | 0 = supprimer)\n")

    for ngfs_var, hist_col in list(matches.items()):
        print(f"  NGFS : {ngfs_var}")
        print(f"  -> actuellement mappe sur : {hist_col}")
        choice = input("  Votre choix (Entree/numero/0) : ").strip()
        if choice == "":
            pass  # keep
        elif choice == "0":
            del matches[ngfs_var]
            method_per_match.pop(ngfs_var, None)
            unmatched.append(ngfs_var)
            print(f"  X Supprime.\n")
        elif choice in col_index:
            matches[ngfs_var] = col_index[choice]
            method_per_match[ngfs_var] = "user"
            print(f"  OK Remplace par : {col_index[choice]}\n")
        else:
            print(f"  !  Choix invalide - mapping inchange.\n")

    if unmatched:
        print("\n  -- Mapping des variables non-mappees --------------------")
        print("  (numero = assigner une colonne | Entree = laisser non-mappee)\n")

        still_unmatched = []
        for ngfs_var in unmatched:
            print(f"  NGFS : {ngfs_var}")
            choice = input("  Votre choix (numero/Entree) : ").strip()
            if choice == "" or choice not in col_index:
                still_unmatched.append(ngfs_var)
            else:
                matches[ngfs_var] = col_index[choice]
                method_per_match[ngfs_var] = "user"
                print(f"  OK Mappe sur : {col_index[choice]}\n")
        unmatched = still_unmatched

    corrected = MappingResult(
        matches=matches,
        unmatched=unmatched,
        method_per_match=method_per_match,
    )
    log.info("Mapping after user correction: %d matched, %d unmatched.",
             len(matches), len(unmatched))
    return corrected


def _maybe_save_yaml(result: MappingResult, cfg: EngineConfig) -> None:
    """
    Sauvegarde le mapping validé en YAML si l'utilisateur le demande.
    Le fichier YAML peut être rechargé directement lors des prochaines
    exécutions via cfg.mapping_yaml_path → skip la validation interactive.
    """
    save_path = cfg.mapping_yaml_path or "mapping_validated.yaml"

    answer = input(
        f"\n  Sauvegarder ce mapping en YAML pour les prochaines "
        f"executions ? [{save_path}]  [O/n] : "
    ).strip().lower()

    if answer in ("", "o", "oui", "y", "yes"):
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(
                result.matches,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=True,
            )
        log.info("Mapping saved to %s", save_path)
        print(f"  OK Mapping sauvegarde : {save_path}")
        print(f"    -> Ajoutez mapping_yaml_path='{save_path}' dans votre "
              f"EngineConfig pour le reutiliser.")
    else:
        print("  Mapping non sauvegarde.")