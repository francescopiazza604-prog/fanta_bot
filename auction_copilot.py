"""
auction_copilot.py — Assistente Asta Live in Tempo Reale.

Gestisce la dinamica dell'Asta del Fantacalcio in diretta:
1. Calcolo del Target Price & Prezzo Massimo Consigliato (Max Bid) per ogni giocatore svincolato.
2. Ricalcolo dinamico del budget residuo ad ogni acquisto proprio o dei rivali.
3. Strategia di allocazione crediti salvati da rilanci mancati.
4. Tracciamento rosa e saturezione ruoli.
"""

import json
import logging
import os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROSA_TARGET = {'P': 3, 'D': 8, 'C': 8, 'A': 6}

# Budget indicativo per reparto base
def get_role_allocation(num_partecipanti: int = 8) -> dict:
    if num_partecipanti <= 8:
        return {'P': 0.05, 'D': 0.16, 'C': 0.32, 'A': 0.47}
    elif num_partecipanti == 10:
        return {'P': 0.09, 'D': 0.20, 'C': 0.28, 'A': 0.43}
    else:  # 12+
        return {'P': 0.11, 'D': 0.24, 'C': 0.25, 'A': 0.40}

# Indice di Integrità Fisica (Durability Index)
# Giocatori storicamente fragili (saltano molte partite ogni stagione)
# Il moltiplicatore taglia matematicamente il MAX BID (es. 0.65 taglia il tetto asta del 35%)
INJURY_PRONE_PLAYERS = {
    "dybala": 0.65,
    "berardi": 0.40,
    "milik": 0.55,
    "sensi": 0.30,
    "castrovilli": 0.50,
    "nico gonzalez": 0.75,
    "zapata": 0.75,
    "chiesa": 0.80,
    "spinazzola": 0.65,
    "pellegrini lo.": 0.85,
    "kalulu": 0.75,
    "bennacer": 0.70,
    "maignan": 0.85,
    "kvaratskhelia": 0.90,  # occasionali stop
}


class AuctionState:
    """Rappresenta lo stato in tempo reale dell'Asta Fantacalcio."""

    def __init__(self, total_budget: int = 500, my_team_name: str = "La Mia Squadra"):
        self.total_budget = total_budget
        self.remaining_budget = total_budget
        self.my_team_name = my_team_name
        self.my_roster = []      # Lista di dict dei giocatori comprati
        self.sold_players = {}   # dict: nome_giocatore -> {squadra_fanta, prezzo}
        self.role_counts = {'P': 0, 'D': 0, 'C': 0, 'A': 0}
        self.role_spent = {'P': 0, 'D': 0, 'C': 0, 'A': 0}

    def buy_player(self, player_dict: dict, price: int, buyer: str = "ME"):
        """Registra l'acquisto di un giocatore da parte mia o di un concorrente."""
        nome = player_dict['nome']
        ruolo = player_dict['ruolo']
        
        self.sold_players[nome] = {'buyer': buyer, 'price': price, 'ruolo': ruolo}

        if buyer.upper() in ["ME", "MY_TEAM", self.my_team_name.upper()]:
            self.my_roster.append({**player_dict, 'prezzo_acquisto': price})
            self.remaining_budget -= price
            self.role_counts[ruolo] = self.role_counts.get(ruolo, 0) + 1
            self.role_spent[ruolo] = self.role_spent.get(ruolo, 0) + price
            logger.info(f"Acquistato {nome} ({ruolo}) a {price} cr. Budget rimasto: {self.remaining_budget} cr.")

    def get_remaining_slots(self, ruolo: str) -> int:
        return max(0, ROSA_TARGET.get(ruolo, 0) - self.role_counts.get(ruolo, 0))

    def get_total_remaining_slots(self) -> int:
        return sum(self.get_remaining_slots(r) for r in ROSA_TARGET.keys())

    def to_json(self) -> str:
        return json.dumps({
            'total_budget': self.total_budget,
            'remaining_budget': self.remaining_budget,
            'my_team_name': self.my_team_name,
            'my_roster': self.my_roster,
            'sold_players': self.sold_players,
            'role_counts': self.role_counts,
            'role_spent': self.role_spent
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        state = cls(total_budget=data['total_budget'], my_team_name=data['my_team_name'])
        state.remaining_budget = data['remaining_budget']
        state.my_roster = data['my_roster']
        state.sold_players = data['sold_players']
        state.role_counts = data['role_counts']
        state.role_spent = data['role_spent']
        return state


def calculate_copilot_bids(
    df_players: pd.DataFrame,
    auction_state: AuctionState,
    num_partecipanti: int = 8
) -> pd.DataFrame:
    """
    Calcola per ciascun giocatore libero nel listone:
    1. Value Above Replacement (VAR): marginalità di FM attesa rispetto al replacement level.
    2. Target Bid Price: prezzo fanta-equo stimato.
    3. Max Recommended Bid: soffitto d'asta massimo invalicabile per mantenere la sostenibilità.
    """
    df = df_players.copy()

    # Rimuoviamo i già venduti
    if auction_state.sold_players:
        df = df[~df['nome'].isin(auction_state.sold_players.keys())].copy()

    if 'previsione_ia' not in df.columns:
        df['previsione_ia'] = df.get('fanta_media', 6.0)

    # Calcolo Replacement Level per ruolo
    replacement_levels = {}
    for r in ['P', 'D', 'C', 'A']:
        role_df = df[df['ruolo'] == r]
        if not role_df.empty:
            # Livello di rimpiazzo: il percentile 40 del pool del ruolo
            replacement_levels[r] = role_df['previsione_ia'].quantile(0.40)
        else:
            replacement_levels[r] = 5.50

    # Budget residuo disponibile
    rem_budget = auction_state.remaining_budget
    rem_slots = auction_state.get_total_remaining_slots()

    if rem_slots <= 0:
        df['target_bid'] = 0
        df['max_bid'] = 0
        df['var_score'] = 0.0
        return df

    # Riserva per coperture a 1 credito
    # Dobbiamo garantire almeno 1 credito per ciascuno slot rimanente dopo questo acquisto
    min_reserve = max(0, rem_slots - 1)
    usable_budget = max(1, rem_budget - min_reserve)

    target_bids = []
    max_bids = []
    var_scores = []

    for idx, row in df.iterrows():
        ruolo = row['ruolo']
        fm = row.get('previsione_ia', 6.0)
        costo_listone = float(row.get('costo_iniziale', 1))
        
        # VAR (Value Above Replacement)
        repl = replacement_levels.get(ruolo, 5.5)
        var = max(0.0, fm - repl)
        var_scores.append(round(var, 2))

        # Se non abbiamo slot in questo ruolo, il Max Bid è 0
        if auction_state.get_remaining_slots(ruolo) <= 0:
            target_bids.append(0)
            max_bids.append(0)
            continue

        # Quota budget allocata al reparto
        role_rem_slots = auction_state.get_remaining_slots(ruolo)
        
        # Target bid proporzionale a costo listone e VAR score
        target_bid = max(1, int(round(costo_listone * (1.0 + (var * 0.25)))))

        # Soffitto massimo (Max Bid) modulato sui partecipanti
        role_allocs = get_role_allocation(num_partecipanti)
        if ruolo == 'A':
            max_cap_pct = role_allocs['A'] + 0.08 if role_rem_slots == ROSA_TARGET.get('A', 6) else role_allocs['A'] - 0.05
        elif ruolo == 'C':
            max_cap_pct = role_allocs['C'] + 0.03 if role_rem_slots >= 6 else role_allocs['C'] - 0.07
        elif ruolo == 'D':
            max_cap_pct = role_allocs['D'] + 0.05 if role_rem_slots >= 6 else role_allocs['D'] - 0.02
        else: # P
            max_cap_pct = role_allocs['P'] + 0.05

        max_bid = min(usable_budget, max(1, int(round(target_bid * 1.35))))
        max_bid = min(max_bid, max(1, int(usable_budget * max_cap_pct)))

        # ── Applicazione Indice di Integrità Fisica ──
        nome_lower = str(row.get('nome', '')).lower().strip()
        durability_factor = 1.0
        for fragile_name, factor in INJURY_PRONE_PLAYERS.items():
            if fragile_name in nome_lower:
                durability_factor = factor
                break
        
        if durability_factor < 1.0:
            target_bid = max(1, int(round(target_bid * durability_factor)))
            max_bid = max(1, int(round(max_bid * durability_factor)))

        target_bids.append(target_bid)
        max_bids.append(max_bid)

    df['var_score'] = var_scores
    df['target_bid'] = target_bids
    df['max_bid'] = max_bids

    # Ordina per valore decrescente
    df = df.sort_values(by=['max_bid', 'previsione_ia'], ascending=[False, False])
    return df
