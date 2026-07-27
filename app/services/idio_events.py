"""
Bibliothèque des événements idiosyncratiques + constructeur DAG Plotly.
7 événements couvrant liquidité, crédit, opérationnel, cyber,
réputation, risque pays et ESG.
"""
from __future__ import annotations
import math
import json
from typing import Dict, Any, Optional

import plotly.graph_objects as go

# ── Node type → fill color  (brand palette orange / grey) ────────────
NODE_COLORS: Dict[str, str] = {
    "event":     "#F06423",   # orange vif     – déclencheur
    "reaction":  "#F59A6A",   # orange clair   – réaction
    "asset":     "#F8C2A4",   # pêche clair    – impact actif
    "liability": "#A6A6A6",   # gris moyen     – impact passif
    "capital":   "#7A7A7A",   # gris foncé     – érosion capital
    "ratio":     "#F06423",   # orange vif     – ratio réglementaire
    "process":   "#D0D0D0",   # gris clair     – processus
    "income":    "#F59A6A",   # orange clair   – impact P&L
}

# ── Node type → text color (dark on light bg, white on dark bg) ───────
NODE_TEXT_COLORS: Dict[str, str] = {
    "event":     "#ffffff",
    "reaction":  "#ffffff",
    "asset":     "#5C2A00",   # brun foncé sur fond pêche clair
    "liability": "#ffffff",
    "capital":   "#ffffff",
    "ratio":     "#ffffff",
    "process":   "#4A4A4A",   # gris foncé sur fond gris clair
    "income":    "#ffffff",
}

HIGHLIGHT_COLORS: Dict[str, str] = {
    "critical": "#F06423",   # orange vif
    "high":     "#F59A6A",   # orange clair
    "medium":   "#F8C2A4",   # pêche
    "low":      "#A6A6A6",   # gris moyen
}

# ── Helpers for compact event definition ────────────────────────────
def _n(nid: str, label: str, ntype: str, x: float, y: float,
       tooltip: str = "") -> Dict:
    return {"id": nid, "label": label, "type": ntype,
            "x": x, "y": y,
            "tooltip": tooltip or label.replace("<br>", " ")}

def _e(src: str, tgt: str) -> Dict:
    return {"from": src, "to": tgt}


# ═══════════════════════════════════════════════════════════════════════
# EVENT LIBRARY
# ═══════════════════════════════════════════════════════════════════════
IDIO_EVENTS: Dict[str, Any] = {   'bank_run': {   'balance_sheet': [   'Dépôts à vue (passif) : retraits massifs → ↓ ressources stables',
                                         "Lignes de crédit (passif) : tirages d'urgence par contreparties",
                                         'HQLA – Obligations souveraines (actif) : vente forcée pour couvrir les '
                                         'sorties'],
                    'category': 'Risque de Liquidité',
                    'dag': {   'edges': [   {'from': 'n0', 'to': 'n1'},
                                            {'from': 'n0', 'to': 'n2'},
                                            {'from': 'n1', 'to': 'n3'},
                                            {'from': 'n1', 'to': 'n4'},
                                            {'from': 'n2', 'to': 'n4'},
                                            {'from': 'n2', 'to': 'n5'},
                                            {'from': 'n3', 'to': 'n6'},
                                            {'from': 'n4', 'to': 'n6'},
                                            {'from': 'n4', 'to': 'n7'},
                                            {'from': 'n5', 'to': 'n7'}],
                               'nodes': [   {   'id': 'n0',
                                                'label': 'Rumeur<br>solvabilité',
                                                'tooltip': 'Événement déclencheur : rumeur ou perte annoncée',
                                                'type': 'event',
                                                'x': 0,
                                                'y': 0},
                                            {   'id': 'n1',
                                                'label': 'Retraits<br>massifs',
                                                'tooltip': 'Déposants retail et corporate retirent leurs fonds',
                                                'type': 'reaction',
                                                'x': 2,
                                                'y': 0.7},
                                            {   'id': 'n2',
                                                'label': 'Tirages<br>lignes crédit',
                                                'tooltip': 'Contreparties tirent sur les lignes de crédit confirmées',
                                                'type': 'liability',
                                                'x': 2,
                                                'y': -0.7},
                                            {   'id': 'n3',
                                                'label': '↓ HQLA<br>vendus',
                                                'tooltip': "Vente forcée d'actifs liquides pour couvrir les sorties",
                                                'type': 'asset',
                                                'x': 4,
                                                'y': 1.0},
                                            {   'id': 'n4',
                                                'label': '↑ Outflows<br>nets 30j',
                                                'tooltip': 'Les sorties nettes sur 30 jours augmentent fortement',
                                                'type': 'process',
                                                'x': 4,
                                                'y': 0.0},
                                            {   'id': 'n5',
                                                'label': '↑ Coût<br>funding',
                                                'tooltip': 'Le spread de refinancement monte avec la perte de '
                                                           'confiance',
                                                'type': 'income',
                                                'x': 4,
                                                'y': -1.0},
                                            {   'id': 'n6',
                                                'label': '↓ LCR',
                                                'tooltip': 'LCR = HQLA / Outflows nets 30j → dégradation critique',
                                                'type': 'ratio',
                                                'x': 6,
                                                'y': 0.7},
                                            {   'id': 'n7',
                                                'label': '↓ NSFR<br>↓ CET1',
                                                'tooltip': 'Funding instable + perte NII érodent NSFR et CET1',
                                                'type': 'capital',
                                                'x': 6,
                                                'y': -0.7}]},
                    'dag_severity_map': {   'fail': {   'n0': 'critical',
                                                        'n1': 'critical',
                                                        'n3': 'critical',
                                                        'n4': 'critical',
                                                        'n6': 'critical',
                                                        'n7': 'high'},
                                            'ok': {'n0': 'medium', 'n1': 'low', 'n3': 'low', 'n4': 'low', 'n6': 'low'},
                                            'warn': {   'n0': 'high',
                                                        'n1': 'high',
                                                        'n3': 'high',
                                                        'n4': 'high',
                                                        'n6': 'high',
                                                        'n7': 'medium'}},
                    'description': 'Perte soudaine de confiance des déposants provoquant des retraits massifs de '
                                   'liquidités. La banque est contrainte de vendre ses actifs liquides (HQLA) à prix '
                                   'bradé, ce qui comprime le LCR et le NSFR tout en renchérissant le coût de '
                                   'refinancement.',
                    'drivers': ['lcr', 'nsfr', 'cet1'],
                    'icon': '🏃',
                    'name': 'Bank Run',
                    'required_base': [   'lcr',
                                         'nsfr',
                                         'hqla',
                                         'asf',
                                         'rsf',
                                         'deposits',
                                         'outflows_30j',
                                         'cet1_ratio',
                                         'capital',
                                         'rwa'],
                    'shock_inputs': [
                        {'id': 'run_off_retail',
                         'label': 'Δ Taux de run-off retail (pp)',
                         'tooltip': 'Bâle III LCR (BCBS 238). Run-off rate retail. EBA stress = +10 à +25 pp',
                         'min': 0.0, 'max': 50.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                        {'id': 'run_off_corp',
                         'label': 'Δ Taux de run-off corporate (pp)',
                         'tooltip': 'LCR §107. Dépôts corporate moins sticky. EBA = +15 à +40 pp',
                         'min': 0.0, 'max': 75.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                        {'id': 'haircut_add',
                         'label': 'Δ Haircut HQLA additionnel (pp)',
                         'tooltip': 'BCBS 238 §50. Fire sales (Shleifer & Vishny 2011)',
                         'min': 0.0, 'max': 30.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                    ],
                    'trigger': "Rumeur sur la solvabilité, annonce d'une perte inattendue, ou faillite d'un "
                               'établissement similaire déclenchant un effet contagion.'},
    'cyber_attack': {   'balance_sheet': [   'Systèmes IT (actif immatériel) : dommages et coûts de remédiation',
                                             'Dépôts (passif) : retraits partiels suite à perte de confiance',
                                             "Revenus d'exploitation : interruption pendant la paralysie"],
                        'category': 'Risque Cyber',
                        'dag': {   'edges': [   {'from': 'n0', 'to': 'n1'},
                                                {'from': 'n0', 'to': 'n2'},
                                                {'from': 'n1', 'to': 'n3'},
                                                {'from': 'n2', 'to': 'n4'},
                                                {'from': 'n3', 'to': 'n5'},
                                                {'from': 'n4', 'to': 'n6'},
                                                {'from': 'n3', 'to': 'n6'}],
                                   'nodes': [   {   'id': 'n0',
                                                    'label': 'Cyberattaque<br>Ransomware',
                                                    'tooltip': "Attaque malveillante sur les systèmes d'information",
                                                    'type': 'event',
                                                    'x': 0,
                                                    'y': 0},
                                                {   'id': 'n1',
                                                    'label': 'Paralysie<br>systèmes',
                                                    'tooltip': 'Systèmes critiques hors service : paiements, trading',
                                                    'type': 'process',
                                                    'x': 2,
                                                    'y': 0.7},
                                                {   'id': 'n2',
                                                    'label': 'Perte de<br>confiance',
                                                    'tooltip': 'Les clients et contreparties perdent confiance',
                                                    'type': 'reaction',
                                                    'x': 2,
                                                    'y': -0.7},
                                                {   'id': 'n3',
                                                    'label': 'Coûts<br>récupération',
                                                    'tooltip': 'Rançon éventuelle + remédiation IT + coûts légaux',
                                                    'type': 'income',
                                                    'x': 4,
                                                    'y': 0.7},
                                                {   'id': 'n4',
                                                    'label': 'Retraits<br>clients',
                                                    'tooltip': 'Retraits de dépôts accélérés par la perte de confiance',
                                                    'type': 'liability',
                                                    'x': 4,
                                                    'y': -0.7},
                                                {   'id': 'n5',
                                                    'label': '↓ CET1<br>↓ CAR',
                                                    'tooltip': 'Pertes opérationnelles érodent le capital '
                                                               'réglementaire',
                                                    'type': 'capital',
                                                    'x': 6,
                                                    'y': 0.7},
                                                {   'id': 'n6',
                                                    'label': '↓ LCR',
                                                    'tooltip': 'Outflows accrus + utilisation HQLA dégradent le LCR',
                                                    'type': 'ratio',
                                                    'x': 6,
                                                    'y': -0.7}]},
                        'dag_severity_map': {   'fail': {   'n0': 'critical',
                                                            'n1': 'high',
                                                            'n2': 'critical',
                                                            'n3': 'critical',
                                                            'n4': 'high',
                                                            'n5': 'critical',
                                                            'n6': 'critical'},
                                                'ok': {   'n0': 'medium',
                                                          'n1': 'low',
                                                          'n2': 'medium',
                                                          'n3': 'low',
                                                          'n4': 'low',
                                                          'n5': 'low',
                                                          'n6': 'low'},
                                                'warn': {   'n0': 'high',
                                                            'n1': 'medium',
                                                            'n2': 'high',
                                                            'n3': 'high',
                                                            'n4': 'medium',
                                                            'n5': 'high',
                                                            'n6': 'high'}},
                        'description': 'Attaque informatique (ransomware, APT) paralysant les systèmes critiques de la '
                                       'banque. Génère des coûts de récupération élevés, une perte de revenus pendant '
                                       'la paralysie et un run-off partiel des dépôts clients par perte de confiance.',
                        'drivers': ['lcr', 'cet1', 'car'],
                        'icon': '💻',
                        'name': 'Cyberattaque / Ransomware',
                        'required_base': [   'cet1_ratio',
                                             'car',
                                             'capital',
                                             'rwa',
                                             'total_cap',
                                             'lcr',
                                             'hqla',
                                             'asf',
                                             'rsf',
                                             'outflows_30j'],
                        'shock_inputs': [
                            {'id': 'run_off_retail',
                             'label': 'Δ Taux de run-off retail (pp)',
                             'tooltip': 'Panique clientèle post-cyberattaque (cas Travelex 2020)',
                             'min': 0.0, 'max': 40.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                            {'id': 'run_off_corp',
                             'label': 'Δ Taux de run-off corporate (pp)',
                             'tooltip': 'Corporates retirent par précaution (BCE TRIM 2020)',
                             'min': 0.0, 'max': 50.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                            {'id': 'inflows_cont',
                             'label': 'Δ Inflows contingents (pp)',
                             'tooltip': 'Systèmes de paiement bloqués → cash inflows interrompus',
                             'min': -60.0, 'max': 0.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                            {'id': 'op_loss_amount',
                             'label': 'Pertes cyber + remédiation + amendes RGPD (M)',
                             'tooltip': 'BCBS 424 + EBA ICT Risk Guidelines. Inclut rançon + remédiation + amendes RGPD',
                             'min': 0.0, 'max': 100000.0, 'step': 10.0, 'default': 0.0, 'unit': 'M'},
                        ],
                        'trigger': "Infiltration des systèmes d'information (phishing, zero-day exploit) ou attaque de "
                                   "la chaîne d'approvisionnement logicielle."},
    'esg_stranded': {   'balance_sheet': [   'Prêts secteurs carbonés (actif) : hausse PD + provisions ECL',
                                             'Portefeuille AFS carbone (actif) : dépréciation forcée',
                                             'RWA : multiplicateur risque climatique (EBA pilier 2)'],
                        'category': 'Risque ESG',
                        'dag': {   'edges': [   {'from': 'n0', 'to': 'n1'},
                                                {'from': 'n0', 'to': 'n2'},
                                                {'from': 'n1', 'to': 'n3'},
                                                {'from': 'n2', 'to': 'n4'},
                                                {'from': 'n3', 'to': 'n5'},
                                                {'from': 'n4', 'to': 'n5'},
                                                {'from': 'n4', 'to': 'n6'}],
                                   'nodes': [   {   'id': 'n0',
                                                    'label': 'Réglementation<br>carbone',
                                                    'tooltip': 'Choc réglementaire ou de transition climatique',
                                                    'type': 'event',
                                                    'x': 0,
                                                    'y': 0},
                                                {   'id': 'n1',
                                                    'label': '↓ Actifs<br>carbonés',
                                                    'tooltip': 'Dépréciation des actifs liés aux secteurs à fort CO₂',
                                                    'type': 'asset',
                                                    'x': 2,
                                                    'y': 0.7},
                                                {   'id': 'n2',
                                                    'label': '↑ Défauts<br>secteurs',
                                                    'tooltip': 'Hausse des défauts sur les emprunteurs des secteurs '
                                                               'exposés',
                                                    'type': 'reaction',
                                                    'x': 2,
                                                    'y': -0.7},
                                                {   'id': 'n3',
                                                    'label': 'Pertes<br>portefeuille',
                                                    'tooltip': 'Pertes réalisées sur le portefeuille carbone',
                                                    'type': 'income',
                                                    'x': 4,
                                                    'y': 0.7},
                                                {   'id': 'n4',
                                                    'label': '↑ NPL<br>↑ Provisions',
                                                    'tooltip': 'NPL en hausse → provisions ECL Stage 3 additionnelles',
                                                    'type': 'process',
                                                    'x': 4,
                                                    'y': -0.7},
                                                {   'id': 'n5',
                                                    'label': '↓ CET1<br>↓ CAR',
                                                    'tooltip': 'Capital réglementaire érodé par pertes + RWA '
                                                               'additionnel',
                                                    'type': 'capital',
                                                    'x': 6,
                                                    'y': 0.7},
                                                {   'id': 'n6',
                                                    'label': '↑ Coût<br>du risque',
                                                    'tooltip': 'Coût du risque (EL/EAD) dégradé de façon permanente',
                                                    'type': 'ratio',
                                                    'x': 6,
                                                    'y': -0.7}]},
                        'dag_severity_map': {   'fail': {   'n0': 'critical',
                                                            'n1': 'high',
                                                            'n2': 'critical',
                                                            'n3': 'high',
                                                            'n4': 'critical',
                                                            'n5': 'critical',
                                                            'n6': 'high'},
                                                'ok': {   'n0': 'medium',
                                                          'n1': 'low',
                                                          'n2': 'low',
                                                          'n3': 'low',
                                                          'n4': 'low',
                                                          'n5': 'low',
                                                          'n6': 'low'},
                                                'warn': {   'n0': 'high',
                                                            'n1': 'medium',
                                                            'n2': 'high',
                                                            'n3': 'medium',
                                                            'n4': 'high',
                                                            'n5': 'high',
                                                            'n6': 'medium'}},
                        'description': 'Réglementation climatique ou transition énergétique forcée dépréciant les '
                                       'actifs liés aux secteurs à forte intensité carbone (charbon, pétrole, ciment) '
                                       'dans le bilan. Hausse des NPL sur ces secteurs et érosion du capital '
                                       'réglementaire.',
                        'drivers': ['cet1', 'car', 'npl', 'cout_risque'],
                        'icon': '🌱',
                        'name': 'Stranded Assets ESG',
                        'required_base': [   'cet1_ratio',
                                             'car',
                                             'capital',
                                             'rwa',
                                             'total_cap',
                                             'npl_ratio',
                                             'cout_risque',
                                             'ead',
                                             'total_el'],
                        'shock_inputs': [
                            {'id': 'PD_shock',
                             'label': 'Multiplicateur PD secteurs carbonés (×)',
                             'tooltip': 'BCE Climate Stress Test 2022, NGFS Phase III. Hausse PD secteurs carbonés (Battiston et al. 2017)',
                             'min': 1.0, 'max': 4.0, 'step': 0.1, 'default': 1.0, 'unit': 'x'},
                            {'id': 'LGD_shock',
                             'label': 'Δ LGD stranded assets (pp)',
                             'tooltip': 'Stranded assets : collatéraux carbonés perdent leur valeur (Carbon Tracker 2013)',
                             'min': 0.0, 'max': 40.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                            {'id': 'securities',
                             'label': 'Δ Valeur obligations carbonées (pp)',
                             'tooltip': 'Dépréciation obligations corporate carbonées (NGFS Phase III)',
                             'min': -50.0, 'max': 0.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                            {'id': 'loans',
                             'label': 'Δ Valeur prêts secteurs exposés (pp)',
                             'tooltip': 'TCFD 2017. Dégradation économique des secteurs exposés',
                             'min': -30.0, 'max': 0.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                        ],
                        'trigger': 'Nouvelle taxe carbone, exclusions sectorielles SFDR, stress test climatique EBA ou '
                                   'révision du multiplicateur de risque ESG dans les modèles internes.'},
    'fraud_internal': {   'balance_sheet': [   'Fonds propres (capital) : déduction directe des pertes réalisées',
                                               "RWA opérationnel : hausse consécutive à l'événement de perte (Basel "
                                               'SMA)'],
                          'category': 'Risque Opérationnel',
                          'dag': {   'edges': [   {'from': 'n0', 'to': 'n1'},
                                                  {'from': 'n0', 'to': 'n2'},
                                                  {'from': 'n1', 'to': 'n3'},
                                                  {'from': 'n2', 'to': 'n3'},
                                                  {'from': 'n2', 'to': 'n4'},
                                                  {'from': 'n3', 'to': 'n5'},
                                                  {'from': 'n4', 'to': 'n6'},
                                                  {'from': 'n3', 'to': 'n6'}],
                                     'nodes': [   {   'id': 'n0',
                                                      'label': 'Fraude<br>interne',
                                                      'tooltip': 'Découverte de la fraude / position non autorisée',
                                                      'type': 'event',
                                                      'x': 0,
                                                      'y': 0},
                                                  {   'id': 'n1',
                                                      'label': 'Perte<br>financière',
                                                      'tooltip': 'Perte P&L directe et immédiate',
                                                      'type': 'income',
                                                      'x': 2,
                                                      'y': 0.7},
                                                  {   'id': 'n2',
                                                      'label': 'Enquête<br>judiciaire',
                                                      'tooltip': "Déclenchement d'une enquête réglementaire et "
                                                                 'judiciaire',
                                                      'type': 'process',
                                                      'x': 2,
                                                      'y': -0.7},
                                                  {   'id': 'n3',
                                                      'label': '↓ Capital<br>direct',
                                                      'tooltip': 'Déduction directe du Tier 1 / CET1',
                                                      'type': 'capital',
                                                      'x': 4,
                                                      'y': 0.7},
                                                  {   'id': 'n4',
                                                      'label': '↑ RWA<br>opérationnel',
                                                      'tooltip': "RWA opérationnel majoré par l'événement de perte "
                                                                 '(Basel SMA)',
                                                      'type': 'process',
                                                      'x': 4,
                                                      'y': -0.7},
                                                  {   'id': 'n5',
                                                      'label': '↓ CET1',
                                                      'tooltip': 'CET1 = Capital / RWA → doublement dégradé',
                                                      'type': 'ratio',
                                                      'x': 6,
                                                      'y': 0.7},
                                                  {   'id': 'n6',
                                                      'label': '↓ CAR<br>↓ Leverage',
                                                      'tooltip': 'CAR et Leverage ratio affectés simultanément',
                                                      'type': 'ratio',
                                                      'x': 6,
                                                      'y': -0.7}]},
                          'dag_severity_map': {   'fail': {   'n0': 'critical',
                                                              'n1': 'critical',
                                                              'n3': 'critical',
                                                              'n4': 'high',
                                                              'n5': 'critical',
                                                              'n6': 'critical'},
                                                  'ok': {   'n0': 'medium',
                                                            'n1': 'low',
                                                            'n3': 'low',
                                                            'n4': 'low',
                                                            'n5': 'low',
                                                            'n6': 'low'},
                                                  'warn': {   'n0': 'high',
                                                              'n1': 'high',
                                                              'n3': 'high',
                                                              'n4': 'medium',
                                                              'n5': 'high',
                                                              'n6': 'high'}},
                          'description': 'Détournement de fonds, trading non autorisé ou manipulation comptable par un '
                                         'ou plusieurs employés causant des pertes financières directes et une hausse '
                                         'du capital réglementaire opérationnel (RWA opérationnel).',
                          'drivers': ['cet1', 'car', 'leverage'],
                          'icon': '🔓',
                          'name': 'Fraude Interne Massive',
                          'required_base': [   'cet1_ratio',
                                               'car',
                                               'capital',
                                               'rwa',
                                               'tier1',
                                               'total_cap',
                                               'leverage_ratio',
                                               'total_exposure'],
                          'shock_inputs': [
                              {'id': 'op_loss_amount',
                               'label': 'Perte opérationnelle (M)',
                               'tooltip': 'Bâle III SMA (BCBS 424). Cas SocGen 2008 = €4.9bn, Barings 1995 = £827m',
                               'min': 0.0, 'max': 100000.0, 'step': 10.0, 'default': 0.0, 'unit': 'M'},
                          ],
                          'trigger': "Découverte d'une malversation interne : position non autorisée, détournement de "
                                     'trésorerie, falsification de comptes.'},
    'npl_surge': {   'balance_sheet': [   'Créances clients (actif) : reclassement en Stage 3',
                                          'Provisions pour pertes sur créances (passif) : augmentation forcée',
                                          'RWA crédit : hausse liée à la dégradation des notations internes'],
                     'category': 'Risque de Crédit',
                     'dag': {   'edges': [   {'from': 'n0', 'to': 'n1'},
                                             {'from': 'n1', 'to': 'n2'},
                                             {'from': 'n1', 'to': 'n3'},
                                             {'from': 'n2', 'to': 'n4'},
                                             {'from': 'n3', 'to': 'n4'},
                                             {'from': 'n3', 'to': 'n5'},
                                             {'from': 'n4', 'to': 'n6'},
                                             {'from': 'n5', 'to': 'n6'}],
                                'nodes': [   {   'id': 'n0',
                                                 'label': 'Choc<br>économique',
                                                 'tooltip': 'Choc macro ou sectoriel dépassant les niveaux attendus',
                                                 'type': 'event',
                                                 'x': 0,
                                                 'y': 0},
                                             {   'id': 'n1',
                                                 'label': '↑ Défauts<br>emprunteurs',
                                                 'tooltip': 'Hausse des défauts et reclassements internes',
                                                 'type': 'reaction',
                                                 'x': 2,
                                                 'y': 0},
                                             {   'id': 'n2',
                                                 'label': '↑ Créances<br>douteuses',
                                                 'tooltip': 'Stock NPL en hausse, passage en Stage 3 IFRS 9',
                                                 'type': 'asset',
                                                 'x': 4,
                                                 'y': 0.7},
                                             {   'id': 'n3',
                                                 'label': '↑ Provisions<br>ECL',
                                                 'tooltip': 'Provisions ECL Stage 3 passées en résultat',
                                                 'type': 'process',
                                                 'x': 4,
                                                 'y': -0.7},
                                             {   'id': 'n4',
                                                 'label': '↓ P&L<br>↓ Capital',
                                                 'tooltip': 'Impact négatif sur résultat net → érosion CET1',
                                                 'type': 'income',
                                                 'x': 6,
                                                 'y': 0.7},
                                             {   'id': 'n5',
                                                 'label': '↑ RWA<br>crédit',
                                                 'tooltip': 'RWA augmente avec la dégradation des notations',
                                                 'type': 'process',
                                                 'x': 6,
                                                 'y': -0.7},
                                             {   'id': 'n6',
                                                 'label': '↓ CET1<br>↓ CAR',
                                                 'tooltip': 'Capital réglementaire dégradé par les deux canaux',
                                                 'type': 'capital',
                                                 'x': 8,
                                                 'y': 0}]},
                     'dag_severity_map': {   'fail': {   'n0': 'critical',
                                                         'n1': 'critical',
                                                         'n2': 'high',
                                                         'n3': 'high',
                                                         'n4': 'critical',
                                                         'n5': 'high',
                                                         'n6': 'critical'},
                                             'ok': {   'n0': 'medium',
                                                       'n1': 'low',
                                                       'n2': 'low',
                                                       'n3': 'low',
                                                       'n4': 'low',
                                                       'n5': 'low',
                                                       'n6': 'low'},
                                             'warn': {   'n0': 'high',
                                                         'n1': 'high',
                                                         'n2': 'medium',
                                                         'n3': 'medium',
                                                         'n4': 'high',
                                                         'n5': 'medium',
                                                         'n6': 'high'}},
                     'description': 'Dégradation soudaine de la qualité du portefeuille crédit : hausse des créances '
                                    'douteuses (NPL), provisionnement forcé en ECL Stage 3, augmentation du RWA et '
                                    'érosion du capital réglementaire.',
                     'drivers': ['cet1', 'car', 'npl', 'cout_risque'],
                     'icon': '📉',
                     'name': 'Hausse Brutale des NPL',
                     'required_base': [   'cet1_ratio',
                                          'car',
                                          'capital',
                                          'rwa',
                                          'total_cap',
                                          'npl_ratio',
                                          'cout_risque',
                                          'ead',
                                          'total_el'],
                     'shock_inputs': [
                         {'id': 'PD_shock',
                          'label': 'Multiplicateur PD (×)',
                          'tooltip': 'BCBS 128 IRB. Levier central ASRF (Vasicek 2002, Gordy 2003). EBA = ×1.5 à ×3',
                          'min': 1.0, 'max': 5.0, 'step': 0.1, 'default': 1.0, 'unit': 'x'},
                         {'id': 'LGD_shock',
                          'label': 'Δ LGD (pp)',
                          'tooltip': 'Downturn LGD obligatoire (BCBS 128). Procyclicité (Altman et al. 2005)',
                          'min': 0.0, 'max': 30.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                     ],
                     'trigger': 'Choc économique sectoriel (immobilier, énergie, PME), hausse brutale des taux ou '
                                'dépression régionale dépassant les seuils internes.'},
    'reputation_crisis': {   'balance_sheet': [   'Dépôts (passif) : run-off accéléré (retail + corporate)',
                                                  'Funding marché (passif) : spread en forte hausse',
                                                  "NII (P&L) : compression des revenus d'intérêts"],
                             'category': 'Risque de Réputation',
                             'dag': {   'edges': [   {'from': 'n0', 'to': 'n1'},
                                                     {'from': 'n0', 'to': 'n2'},
                                                     {'from': 'n1', 'to': 'n3'},
                                                     {'from': 'n2', 'to': 'n4'},
                                                     {'from': 'n3', 'to': 'n5'},
                                                     {'from': 'n4', 'to': 'n5'},
                                                     {'from': 'n4', 'to': 'n6'},
                                                     {'from': 'n5', 'to': 'n6'}],
                                        'nodes': [   {   'id': 'n0',
                                                         'label': 'Scandale<br>médiatique',
                                                         'tooltip': 'Événement réputationnel déclencheur',
                                                         'type': 'event',
                                                         'x': 0,
                                                         'y': 0},
                                                     {   'id': 'n1',
                                                         'label': 'Perte clients<br>retail',
                                                         'tooltip': 'Fuite des déposants retail (confiance dégradée)',
                                                         'type': 'reaction',
                                                         'x': 2,
                                                         'y': 0.7},
                                                     {   'id': 'n2',
                                                         'label': '↓ Rating<br>agences',
                                                         'tooltip': "Dégradation de la note par S&P, Moody's ou Fitch",
                                                         'type': 'reaction',
                                                         'x': 2,
                                                         'y': -0.7},
                                                     {   'id': 'n3',
                                                         'label': 'Retraits<br>dépôts',
                                                         'tooltip': 'Run-off rate accéléré sur les dépôts non couverts',
                                                         'type': 'liability',
                                                         'x': 4,
                                                         'y': 0.7},
                                                     {   'id': 'n4',
                                                         'label': '↑ Spread<br>funding',
                                                         'tooltip': 'Prime de risque sur les nouvelles émissions de '
                                                                    'dette',
                                                         'type': 'income',
                                                         'x': 4,
                                                         'y': -0.7},
                                                     {   'id': 'n5',
                                                         'label': '↓ LCR<br>↓ NSFR',
                                                         'tooltip': 'Liquidité à court (LCR) et moyen terme (NSFR) '
                                                                    'dégradée',
                                                         'type': 'ratio',
                                                         'x': 6,
                                                         'y': 0.7},
                                                     {   'id': 'n6',
                                                         'label': '↓ NII<br>↓ CET1',
                                                         'tooltip': 'Hausse du coût de funding → baisse NII → érosion '
                                                                    'CET1',
                                                         'type': 'capital',
                                                         'x': 6,
                                                         'y': -0.7}]},
                             'dag_severity_map': {   'fail': {   'n0': 'critical',
                                                                 'n1': 'critical',
                                                                 'n2': 'high',
                                                                 'n3': 'critical',
                                                                 'n4': 'high',
                                                                 'n5': 'critical',
                                                                 'n6': 'high'},
                                                     'ok': {   'n0': 'medium',
                                                               'n1': 'low',
                                                               'n2': 'low',
                                                               'n3': 'low',
                                                               'n4': 'low',
                                                               'n5': 'low',
                                                               'n6': 'low'},
                                                     'warn': {   'n0': 'high',
                                                                 'n1': 'high',
                                                                 'n2': 'medium',
                                                                 'n3': 'high',
                                                                 'n4': 'medium',
                                                                 'n5': 'high',
                                                                 'n6': 'medium'}},
                             'description': "Scandale médiatique ou publication d'informations négatives provoquant "
                                            'une fuite accélérée des clients retail et wholesale, une dégradation de '
                                            'notation et un renchérissement brutal du coût de financement sur les '
                                            'marchés.',
                             'drivers': ['lcr', 'nsfr', 'cet1'],
                             'icon': '📰',
                             'name': 'Crise de Réputation',
                             'required_base': [   'lcr',
                                                  'nsfr',
                                                  'hqla',
                                                  'asf',
                                                  'rsf',
                                                  'deposits',
                                                  'outflows_30j',
                                                  'cet1_ratio',
                                                  'capital',
                                                  'rwa',
                                                  'total_cap'],
                             'shock_inputs': [
                                 {'id': 'run_off_retail',
                                  'label': 'Δ Taux de run-off retail (pp)',
                                  'tooltip': 'Sortie progressive de dépôts (Credit Suisse 2023, Wells Fargo 2016)',
                                  'min': 0.0, 'max': 30.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                                 {'id': 'run_off_corp',
                                  'label': 'Δ Taux de run-off corporate (pp)',
                                  'tooltip': 'Corporates sensibles à la réputation (BCBS 144)',
                                  'min': 0.0, 'max': 40.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                                 {'id': 'asf_factor_retail',
                                  'label': 'Δ Facteur ASF retail (pp)',
                                  'tooltip': 'NSFR (BCBS 295). Dépôts deviennent moins stables → ASF factor ↓',
                                  'min': -20.0, 'max': 0.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                                 {'id': 'asf_factor_corp',
                                  'label': 'Δ Facteur ASF corporate (pp)',
                                  'tooltip': 'Idem pour dépôts corporate (BCBS 295)',
                                  'min': -25.0, 'max': 0.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                             ],
                             'trigger': 'Scandale public, mauvaise presse persistante, enquête régulatrice médiatisée, '
                                        "ou contamination par la faillite d'un actionnaire clé."},
    'sovereign_downgrade': {   'balance_sheet': [   "Portefeuille AFS souverain (actif) : pertes MTM déduites de l'OCI "
                                                    '/ CET1',
                                                    'HQLA Level 1 (actif) : reclassement Level 2A avec haircut accru',
                                                    'Funding marché (passif) : spread souverain transmis à la banque'],
                               'category': 'Risque Pays',
                               'dag': {   'edges': [   {'from': 'n0', 'to': 'n1'},
                                                       {'from': 'n0', 'to': 'n2'},
                                                       {'from': 'n1', 'to': 'n3'},
                                                       {'from': 'n1', 'to': 'n4'},
                                                       {'from': 'n2', 'to': 'n5'},
                                                       {'from': 'n3', 'to': 'n6'},
                                                       {'from': 'n4', 'to': 'n7'},
                                                       {'from': 'n5', 'to': 'n7'}],
                                          'nodes': [   {   'id': 'n0',
                                                           'label': 'Dégradation<br>souveraine',
                                                           'tooltip': 'Révision à la baisse de la note souveraine',
                                                           'type': 'event',
                                                           'x': 0,
                                                           'y': 0},
                                                       {   'id': 'n1',
                                                           'label': '↓ Valeur<br>oblig. État',
                                                           'tooltip': 'Dépréciation MTM du portefeuille obligataire '
                                                                      "d'État",
                                                           'type': 'asset',
                                                           'x': 2,
                                                           'y': 0.7},
                                                       {   'id': 'n2',
                                                           'label': '↑ Spread<br>souverain',
                                                           'tooltip': 'Le spread CDS souverain augmente → funding plus '
                                                                      'cher',
                                                           'type': 'reaction',
                                                           'x': 2,
                                                           'y': -0.7},
                                                       {   'id': 'n3',
                                                           'label': '↓ HQLA<br>haircut ↑',
                                                           'tooltip': "Les obligations d'État passent Level 1→2A, "
                                                                      'haircut +5%/notch',
                                                           'type': 'asset',
                                                           'x': 4,
                                                           'y': 1.0},
                                                       {   'id': 'n4',
                                                           'label': 'Pertes<br>MTM OCI',
                                                           'tooltip': 'Pertes latentes passées en OCI et déduites du '
                                                                      'CET1',
                                                           'type': 'income',
                                                           'x': 4,
                                                           'y': 0.0},
                                                       {   'id': 'n5',
                                                           'label': '↑ Coût<br>refi',
                                                           'tooltip': 'Hausse du spread de refinancement de la banque',
                                                           'type': 'liability',
                                                           'x': 4,
                                                           'y': -1.0},
                                                       {   'id': 'n6',
                                                           'label': '↓ LCR',
                                                           'tooltip': 'HQLA ajustés / Outflows → LCR dégradé',
                                                           'type': 'ratio',
                                                           'x': 6,
                                                           'y': 0.7},
                                                       {   'id': 'n7',
                                                           'label': '↓ CET1<br>↓ CAR',
                                                           'tooltip': 'Pertes OCI et hausse coût funding érodent CET1 '
                                                                      'et CAR',
                                                           'type': 'capital',
                                                           'x': 6,
                                                           'y': -0.7}]},
                               'dag_severity_map': {   'fail': {   'n0': 'critical',
                                                                   'n1': 'critical',
                                                                   'n3': 'critical',
                                                                   'n4': 'critical',
                                                                   'n5': 'high',
                                                                   'n6': 'critical',
                                                                   'n7': 'critical'},
                                                       'ok': {   'n0': 'medium',
                                                                 'n1': 'low',
                                                                 'n3': 'low',
                                                                 'n4': 'low',
                                                                 'n5': 'low',
                                                                 'n6': 'low',
                                                                 'n7': 'low'},
                                                       'warn': {   'n0': 'high',
                                                                   'n1': 'high',
                                                                   'n3': 'high',
                                                                   'n4': 'high',
                                                                   'n5': 'medium',
                                                                   'n6': 'high',
                                                                   'n7': 'high'}},
                               'description': "Dégradation de la notation souveraine du pays d'implantation. Les "
                                              "obligations d'État détenues en HQLA subissent des haircuts accrus, des "
                                              'pertes MTM sont passées en OCI (déduites du CET1), et le coût de '
                                              'refinancement de la banque augmente en parallèle.',
                               'drivers': ['lcr', 'cet1', 'car'],
                               'icon': '🏛️',
                               'name': 'Dégradation Note Souveraine',
                               'required_base': [   'lcr',
                                                    'hqla',
                                                    'asf',
                                                    'rsf',
                                                    'outflows_30j',
                                                    'cet1_ratio',
                                                    'capital',
                                                    'rwa',
                                                    'total_cap',
                                                    'car'],
                               'shock_inputs': [
                                   {'id': 'haircut_add',
                                    'label': 'Δ Haircut HQLA additionnel (pp)',
                                    'tooltip': 'LCR §50. Downgrade → hausse haircuts HQLA (Grèce 2010, Égypte 2023)',
                                    'min': 0.0, 'max': 50.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                                   {'id': 'L1',
                                    'label': 'Δ Actifs Level 1 (pp)',
                                    'tooltip': 'BCBS 238 §50. Titres souverains downgradés sortent du Level 1',
                                    'min': -40.0, 'max': 0.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                                   {'id': 'securities',
                                    'label': 'Δ Valeur portefeuille souverain (pp)',
                                    'tooltip': 'MTM loss sur portefeuille souverain (sovereign-bank nexus, Acharya et al. 2014)',
                                    'min': -30.0, 'max': 0.0, 'step': 1.0, 'default': 0.0, 'unit': 'pp'},
                                   {'id': 'RWA_shock',
                                    'label': 'Δ RWA souverain (pp)',
                                    'tooltip': 'Bâle III SA (BCBS d424). Pondération souveraine plus élevée post-downgrade',
                                    'min': 0.0, 'max': 100.0, 'step': 5.0, 'default': 0.0, 'unit': 'pp'},
                               ],
                               'trigger': "Révision à la baisse de la note souveraine d'au moins un cran par une "
                                          "agence internationale (S&P, Moody's, Fitch)."}}

# -- Ordered list for dropdown display ---------------------------------------
EVENTS_LIST = [
    {"id": k, "name": v["name"], "category": v["category"], "icon": v["icon"]}
    for k, v in IDIO_EVENTS.items()
]


# ═══════════════════════════════════════════════════════════════════════
# DAG FIGURE BUILDER
# ═══════════════════════════════════════════════════════════════════════
def _hex_alpha(hex6: str, alpha: float) -> str:
    """Convert #rrggbb + alpha (0-1) to rgba(r,g,b,a) for Plotly."""
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def build_dag_figure(event: Dict, highlights: Optional[Dict[str, str]] = None) -> Dict:
    """
    Build a Plotly DAG figure for the transmission chain.

    Parameters
    ----------
    event      : event dict from IDIO_EVENTS
    highlights : {node_id: severity_str} — nodes to highlight after shock
                 severity ∈ {"critical","high","medium","low"}

    Returns
    -------
    dict  Plotly figure serialised as plain dict (JSON-safe)
    """
    dag = event.get("dag", {})
    nodes: list = dag.get("nodes", [])
    edges: list = dag.get("edges", [])

    if not nodes:
        return {}

    BG = "#ffffff"
    hl = highlights or {}

    # -- Helper: offset arrow so it starts/ends at box edge, not center
    def _ep(x0: float, y0: float, x1: float, y1: float, r: float = 0.45):
        dx, dy = x1 - x0, y1 - y0
        d = math.sqrt(dx * dx + dy * dy) or 1e-9
        return (x0 + dx / d * r, y0 + dy / d * r,
                x1 - dx / d * r, y1 - dy / d * r)

    node_map = {n["id"]: n for n in nodes}
    fig = go.Figure()

    # ── Draw edges first (behind nodes) ────────────────────────────
    for edge in edges:
        src = node_map.get(edge["from"])
        tgt = node_map.get(edge["to"])
        if not src or not tgt:
            continue
        ax, ay, ex, ey = _ep(src["x"], src["y"], tgt["x"], tgt["y"], r=0.50)

        active = (edge["from"] in hl) or (edge["to"] in hl)
        a_color = "rgba(240,100,35,0.95)" if active else "rgba(160,160,160,0.65)"
        a_width = 3.0 if active else 2.0

        fig.add_annotation(
            ax=ax, ay=ay, x=ex, y=ey,
            axref="x", ayref="y", xref="x", yref="y",
            arrowhead=3, arrowsize=1.3, arrowwidth=a_width,
            arrowcolor=a_color,
            showarrow=True, text="",
        )

    # ── Draw nodes grouped by type ──────────────────────────────────
    for ntype, base_color in NODE_COLORS.items():
        group = [n for n in nodes if n["type"] == ntype]
        if not group:
            continue

        text_color = NODE_TEXT_COLORS.get(ntype, "#ffffff")
        xs, ys, texts, hovers = [], [], [], []
        m_colors, b_colors, b_widths = [], [], []

        for nd in group:
            nid = nd["id"]
            severity = hl.get(nid)

            if severity:
                fill = HIGHLIGHT_COLORS.get(severity, base_color)
                border = "#2D2D2D"
                bw = 3.5
            elif hl:
                fill = _hex_alpha(base_color, 0.28)
                border = "rgba(200,200,200,0.35)"
                bw = 1.0
            else:
                fill = base_color
                border = "rgba(255,255,255,0.70)"
                bw = 2.0

            xs.append(nd["x"])
            ys.append(nd["y"])
            texts.append(f"<b>{nd['label']}</b>")
            hovers.append(nd.get("tooltip", nd["label"]))
            m_colors.append(fill)
            b_colors.append(border)
            b_widths.append(bw)

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(
                size=90,
                color=m_colors,
                symbol="square",
                line=dict(color=b_colors, width=b_widths),
            ),
            text=texts,
            textposition="middle center",
            textfont=dict(size=12, color=text_color,
                          family="Arial Black, Arial, sans-serif"),
            hovertext=hovers,
            hoverinfo="text",
            showlegend=False,
        ))

    # ── Legend strip at bottom ──────────────────────────────────────
    legend_items = [
        ("Déclencheur", "#F06423"),
        ("Réaction",    "#F59A6A"),
        ("Actif",       "#F8C2A4"),
        ("Passif",      "#A6A6A6"),
        ("Capital",     "#7A7A7A"),
        ("Ratio",       "#F06423"),
        ("Processus",   "#D0D0D0"),
        ("P&L",         "#F59A6A"),
    ]
    fig.add_trace(go.Scatter(
        x=[i * 1.18 for i in range(len(legend_items))],
        y=[-2.05] * len(legend_items),
        mode="markers+text",
        marker=dict(
            size=16,
            color=[c for _, c in legend_items],
            symbol="square",
            line=dict(color="rgba(120,120,120,0.5)", width=1),
        ),
        text=[f"  {lbl}" for lbl, _ in legend_items],
        textposition="middle right",
        textfont=dict(size=10, color="#7A7A7A", family="Arial, sans-serif"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # ── Layout ─────────────────────────────────────────────────────
    all_x = [n["x"] for n in nodes]
    all_y = [n["y"] for n in nodes]
    x_range = [min(all_x) - 1.2, max(all_x) + 1.2]
    y_range = [min(all_y) - 1.4, max(all_y) + 0.9]

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=16, r=16, t=12, b=12),
        height=420,
        xaxis=dict(visible=False, range=x_range),
        yaxis=dict(visible=False, range=y_range),
        font=dict(color="#2D2D2D"),
        hovermode="closest",
    )

    return json.loads(fig.to_json())
