"""
auction_copilot.py — Assistente Asta Live in Tempo Reale.

Gestisce la dinamica dell'Asta del Fantacalcio in diretta:
1. Tracciamento Avversari: budget, slot rimanenti, focus (Strategia).
2. Inflazione Mercato: calcola quanti crediti "girano" rispetto ai posti vuoti.
3. Scarsità: se sei l'unico a cui manca un portiere, il max_bid crolla a 1.
4. Durability Dinamica: usa i dati storici (es. presenze) al posto di liste fisse.
"""

import json
import logging
import os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROSA_TARGET = {'P': 3, 'D': 8, 'C': 8, 'A': 6}

def get_role_allocation(num_partecipanti: int = 8) -> dict:
    if num_partecipanti <= 8:
        return {'P': 0.05, 'D': 0.16, 'C': 0.32, 'A': 0.47}
    elif num_partecipanti == 10:
        return {'P': 0.09, 'D': 0.20, 'C': 0.28, 'A': 0.43}
    else:  # 12+
        return {'P': 0.11, 'D': 0.24, 'C': 0.25, 'A': 0.40}

class AuctionState:
    """Rappresenta lo stato in tempo reale dell'Asta Fantacalcio."""

    def __init__(self, total_budget: int = 500, my_team_name: str = "ME", opponents: list[str] = None):
        self.total_budget = total_budget
        self.my_team_name = my_team_name
        
        # Inizializza tutte le squadre (ME + avversari)
        self.teams = {}
        self.teams[my_team_name] = self._empty_team_state(total_budget)
        
        if opponents is None:
            opponents = [f"RIVALE {i}" for i in range(1, 8)]
            
        for opp in opponents:
            if opp.strip() and opp != my_team_name:
                self.teams[opp] = self._empty_team_state(total_budget)
                
        self.sold_players = {}   # dict: nome_giocatore -> {buyer, price, ruolo}
        
    def _empty_team_state(self, budget: int) -> dict:
        return {
            'budget': budget,
            'roster': [],
            'role_counts': {'P': 0, 'D': 0, 'C': 0, 'A': 0},
            'role_spent': {'P': 0, 'D': 0, 'C': 0, 'A': 0}
        }

    @property
    def remaining_budget(self) -> int:
        return self.teams[self.my_team_name]['budget']
        
    @property
    def my_roster(self) -> list:
        return self.teams[self.my_team_name]['roster']

    @property
    def role_counts(self) -> dict:
        return self.teams[self.my_team_name]['role_counts']

    @property
    def role_spent(self) -> dict:
        return self.teams[self.my_team_name]['role_spent']

    def buy_player(self, player_dict: dict, price: int, buyer: str = "ME"):
        """Registra l'acquisto di un giocatore per uno specifico fantallenatore."""
        nome = player_dict['nome']
        ruolo = player_dict['ruolo']
        
        # Mappa "ME" nel nome effettivo
        if buyer.upper() in ["ME", "MY_TEAM"] or buyer == self.my_team_name:
            buyer_key = self.my_team_name
        else:
            buyer_key = buyer if buyer in self.teams else list(self.teams.keys())[1] # fallback

        self.sold_players[nome] = {'buyer': buyer_key, 'price': price, 'ruolo': ruolo}
        
        t = self.teams[buyer_key]
        t['roster'].append({**player_dict, 'prezzo_acquisto': price})
        t['budget'] -= price
        t['role_counts'][ruolo] = t['role_counts'].get(ruolo, 0) + 1
        t['role_spent'][ruolo] = t['role_spent'].get(ruolo, 0) + price
        
        logger.info(f"[{buyer_key}] Acquistato {nome} ({ruolo}) a {price} cr. Budget rimasto: {t['budget']} cr.")

    def get_remaining_slots(self, team_name: str, ruolo: str) -> int:
        counts = self.teams[team_name]['role_counts']
        return max(0, ROSA_TARGET.get(ruolo, 0) - counts.get(ruolo, 0))

    def get_total_remaining_slots(self, team_name: str) -> int:
        return sum(self.get_remaining_slots(team_name, r) for r in ROSA_TARGET.keys())

    def get_market_heat(self) -> float:
        """Calcola l'indicatore di Inflazione (Heat). >1 = Inflazione, <1 = Deflazione."""
        total_initial = len(self.teams) * self.total_budget
        total_left = sum(t['budget'] for t in self.teams.values())
        slots_left = sum(self.get_total_remaining_slots(t_name) for t_name in self.teams)
        
        if slots_left == 0:
            return 1.0
            
        avg_budget_per_slot = total_left / slots_left
        baseline_per_slot = total_initial / (len(self.teams) * 25)
        
        return avg_budget_per_slot / baseline_per_slot

    def to_json(self) -> str:
        return json.dumps({
            'total_budget': self.total_budget,
            'my_team_name': self.my_team_name,
            'teams': self.teams,
            'sold_players': self.sold_players
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        # Compatibilità con vecchia versione
        if 'teams' not in data:
            old_opp = ["RIVALE 1", "RIVALE 2", "RIVALE 3", "RIVALE 4", "RIVALE 5", "RIVALE 6", "RIVALE 7"]
            state = cls(total_budget=data['total_budget'], my_team_name=data.get('my_team_name', 'ME'), opponents=old_opp)
            state.teams[state.my_team_name]['budget'] = data.get('remaining_budget', data['total_budget'])
            state.teams[state.my_team_name]['roster'] = data.get('my_roster', [])
            state.teams[state.my_team_name]['role_counts'] = data.get('role_counts', {'P':0, 'D':0, 'C':0, 'A':0})
            state.teams[state.my_team_name]['role_spent'] = data.get('role_spent', {'P':0, 'D':0, 'C':0, 'A':0})
            state.sold_players = data.get('sold_players', {})
            return state
            
        state = cls(total_budget=data['total_budget'], my_team_name=data['my_team_name'], opponents=[])
        state.teams = data['teams']
        state.sold_players = data['sold_players']
        return state


def calculate_copilot_bids(
    df_players: pd.DataFrame,
    auction_state: AuctionState,
    num_partecipanti: int = 8
) -> pd.DataFrame:
    """
    Calcola target e max_bid integrando:
    - Inflazione globale
    - Scarsità (se nessun rivale cerca il ruolo, max_bid = 1)
    - Durability (da presenze)
    """
    df = df_players.copy()
    my_team = auction_state.my_team_name

    if auction_state.sold_players:
        df = df[~df['nome'].isin(auction_state.sold_players.keys())].copy()

    if 'previsione_ia' not in df.columns:
        df['previsione_ia'] = df.get('fanta_media', 6.0)

    replacement_levels = {}
    for r in ['P', 'D', 'C', 'A']:
        role_df = df[df['ruolo'] == r]
        if not role_df.empty:
            replacement_levels[r] = role_df['previsione_ia'].quantile(0.40)
        else:
            replacement_levels[r] = 5.50

    rem_budget = auction_state.teams[my_team]['budget']
    rem_slots = auction_state.get_total_remaining_slots(my_team)

    if rem_slots <= 0:
        df['target_bid'] = 0
        df['max_bid'] = 0
        df['var_score'] = 0.0
        return df

    min_reserve = max(0, rem_slots - 1)
    usable_budget = max(1, rem_budget - min_reserve)

    # Market Heat
    heat = auction_state.get_market_heat()
    heat_modifier = max(0.6, min(heat, 1.4))  # Limitiamo l'impatto tra 60% e 140%

    target_bids = []
    max_bids = []
    var_scores = []

    for idx, row in df.iterrows():
        ruolo = row['ruolo']
        fm = row.get('previsione_ia', 6.0)
        costo_listone = float(row.get('costo_iniziale', 1))
        
        repl = replacement_levels.get(ruolo, 5.5)
        var = max(0.0, fm - repl)
        var_scores.append(round(var, 2))

        my_role_slots = auction_state.get_remaining_slots(my_team, ruolo)
        if my_role_slots <= 0:
            target_bids.append(0)
            max_bids.append(0)
            continue

        # Scarcity Check: quanti slot liberi hanno I RIVALI in questo ruolo?
        rival_slots = sum(auction_state.get_remaining_slots(t, ruolo) for t in auction_state.teams if t != my_team)
        
        if rival_slots == 0:
            # EFFETTO SCARSITA': Nessun rivale ha bisogno di questo ruolo! 
            target_bids.append(1)
            max_bids.append(1)
            continue

        target_bid = max(1, int(round(costo_listone * (1.0 + (var * 0.25)) * heat_modifier)))

        role_allocs = get_role_allocation(len(auction_state.teams))
        
        # Calcolo budget dinamico del ruolo per evitare di prendere 2 Top player nello stesso ruolo
        my_spent_total = sum(auction_state.teams[my_team]['role_spent'].values())
        my_initial_budget = rem_budget + my_spent_total
        role_ideal_budget = my_initial_budget * role_allocs[ruolo]
        role_already_spent = auction_state.teams[my_team]['role_spent'].get(ruolo, 0)
        
        # Budget rimanente ideale per questo ruolo
        role_rem_budget = role_ideal_budget - role_already_spent
        
        if role_rem_budget <= 0:
            # Hai già sfondato o saturato il budget per questo ruolo (es. preso un Top Portiere a 30)
            # Copilot bloccherà target alti, lasciando al massimo l'1.5% del budget (per gli scarti a 1 cr)
            max_cap_pct = 0.015
        else:
            # Trasformiamo il budget ideale rimanente in percentuale sul budget usabile attuale
            # Aggiungiamo un buffer di flessibilità (+5%) per le aste combattute
            max_cap_pct = min(1.0, (role_rem_budget / usable_budget) + 0.05)

        max_bid = min(usable_budget, max(1, int(round(target_bid * 1.35))))
        max_bid = min(max_bid, max(1, int(usable_budget * max_cap_pct)))

        # Durability Dinamica: penalizza se presenze sono basse (< 20 su storici di 38)
        # Se 'presenze' non c'è, assumed 30.
        presenze = float(row.get('presenze', 30.0))
        if presenze < 20.0 and 'presenze' in row:
            durability_factor = max(0.4, presenze / 38.0)
            target_bid = max(1, int(round(target_bid * durability_factor)))
            max_bid = max(1, int(round(max_bid * durability_factor)))

        target_bids.append(target_bid)
        max_bids.append(max_bid)

    df['var_score'] = var_scores
    df['target_bid'] = target_bids
    df['max_bid'] = max_bids

    # Mantieni il CUI se esiste
    if 'cui' not in df.columns:
        df['cui'] = 0.0
    if 'cui_label' not in df.columns:
        df['cui_label'] = ''

    df = df.sort_values(by=['max_bid', 'previsione_ia'], ascending=[False, False])
    return df

