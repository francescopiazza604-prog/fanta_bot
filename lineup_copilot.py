"""
lineup_copilot.py — Copilota Schieramento Formazione (Matchday Lineup Optimizer)

Utilizza la Teoria dei Giochi e un risolutore matematico (MILP) per decidere chi schierare 
ogni settimana. Considera:
1. I giocatori della propria rosa.
2. I giocatori schierati dall'avversario (Hedging / Risk Adjustment).
3. Difficoltà del match della settimana.
"""

import pandas as pd
import numpy as np
import pulp
import logging

logger = logging.getLogger(__name__)

# Moduli validi al Fantacalcio (Difesa, Centrocampo, Attacco)
VALID_FORMATIONS = [
    (3, 4, 3),
    (3, 5, 2),
    (4, 3, 3),
    (4, 4, 2),
    (4, 5, 1),
    (5, 3, 2),
    (5, 4, 1)
]

# Difficoltà fittizia dei match per la giornata (Simulazione)
# 1.5 = Partita facilissima, 0.5 = Partita difficilissima
TEAM_MATCH_DIFFICULTY = {
    'Inter': 1.4, 'Juventus': 1.3, 'Milan': 1.3, 'Atalanta': 1.25, 
    'Napoli': 1.25, 'Roma': 1.15, 'Lazio': 1.1, 'Fiorentina': 1.05,
    'Bologna': 1.0, 'Torino': 0.95, 'Genoa': 0.9, 'Monza': 0.9,
    'Lecce': 0.85, 'Udinese': 0.85, 'Verona': 0.8, 'Empoli': 0.8,
    'Cagliari': 0.8, 'Parma': 0.75, 'Como': 0.75, 'Venezia': 0.7
}

def optimize_weekly_lineup(my_roster: pd.DataFrame, opponent_roster: pd.DataFrame = None) -> dict:
    """
    Calcola la formazione perfetta per la giornata in corso.
    my_roster: DataFrame dei giocatori acquistati.
    opponent_roster: DataFrame dei giocatori dell'avversario (opzionale).
    """
    if my_roster is None or my_roster.empty:
        return {"error": "La tua rosa è vuota. Registra gli acquisti prima di schierare la formazione."}
        
    df = my_roster.copy()
    
    # 1. Calcolo Proiezione Punti (FantaMedia di Base x Difficoltà Match)
    df['fm_attesa'] = df.get('previsione_ia', df.get('fanta_media', 6.0))
    df['fm_attesa'] = df['fm_attesa'].fillna(6.0)
    
    # Simula la difficoltà del match (bonus/malus basato sul team del giocatore)
    match_mults = []
    for _, row in df.iterrows():
        team = str(row.get('squadra', 'Unknown')).strip()
        # Se non abbiamo la squadra nel df (capita nei listoni raw), usiamo un multiplier neutro
        mult = TEAM_MATCH_DIFFICULTY.get(team, 1.0)
        match_mults.append(mult)
    
    df['match_multiplier'] = match_mults
    df['punti_proiettati'] = df['fm_attesa'] * (0.8 + (df['match_multiplier'] * 0.2))
    
    # 2. Risk Adjustment (Hedging contro l'avversario)
    # Se l'avversario ha un top player di una squadra X, potremmo voler alzare
    # leggermente il valore dei difensori o portieri di QUELLA squadra per coprirci.
    if opponent_roster is not None and not opponent_roster.empty:
        opp_teams = opponent_roster['squadra'].dropna().unique()
        # Bonus Hedging (+0.1 punti proiettati se gioca nella stessa squadra dell'avversario)
        df.loc[df['squadra'].isin(opp_teams), 'punti_proiettati'] += 0.15

    # 3. MILP Optimization (Selezione Top 11)
    prob = pulp.LpProblem("Lineup_Optimization", pulp.LpMaximize)
    
    player_vars = {}
    for i, row in df.iterrows():
        player_vars[i] = pulp.LpVariable(f"P_{i}", cat="Binary")
        
    # Obiettivo: Massimizzare punti_proiettati
    prob += pulp.lpSum([row['punti_proiettati'] * player_vars[i] for i, row in df.iterrows()])
    
    # Vincolo 1: Esattamente 11 giocatori
    prob += pulp.lpSum([player_vars[i] for i in df.index]) == 11
    
    # Vincolo 2: Esattamente 1 Portiere
    idx_p = df[df['ruolo'] == 'P'].index
    prob += pulp.lpSum([player_vars[i] for i in idx_p]) == 1
    
    # Variabili dummy per i moduli (quale stiamo usando?)
    module_vars = {fmt: pulp.LpVariable(f"Mod_{fmt}", cat="Binary") for fmt in VALID_FORMATIONS}
    
    # Deve essere scelto esattamente 1 modulo
    prob += pulp.lpSum(module_vars.values()) == 1
    
    idx_d = df[df['ruolo'] == 'D'].index
    idx_c = df[df['ruolo'] == 'C'].index
    idx_a = df[df['ruolo'] == 'A'].index
    
    # Vincoli di ruolo legati al modulo scelto
    # Difensori = Somma (D_mod * Mod_mod)
    prob += pulp.lpSum([player_vars[i] for i in idx_d]) == pulp.lpSum([fmt[0] * module_vars[fmt] for fmt in VALID_FORMATIONS])
    prob += pulp.lpSum([player_vars[i] for i in idx_c]) == pulp.lpSum([fmt[1] * module_vars[fmt] for fmt in VALID_FORMATIONS])
    prob += pulp.lpSum([player_vars[i] for i in idx_a]) == pulp.lpSum([fmt[2] * module_vars[fmt] for fmt in VALID_FORMATIONS])

    # Risoluzione
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        return {"error": "Nessuna formazione valida trovata. Hai abbastanza giocatori per ogni ruolo?"}
        
    # Risultati
    selected_indices = [i for i in df.index if player_vars[i].varValue == 1]
    bench_indices = [i for i in df.index if player_vars[i].varValue == 0]
    
    titolari = df.loc[selected_indices].copy()
    panchina = df.loc[bench_indices].copy()
    
    # Ordiniamo i titolari per ruolo (P, D, C, A)
    role_order = {'P': 1, 'D': 2, 'C': 3, 'A': 4}
    titolari['role_idx'] = titolari['ruolo'].map(role_order)
    titolari = titolari.sort_values(by=['role_idx', 'punti_proiettati'], ascending=[True, False])
    
    # Ordiniamo la panchina (prima chi ha punti proiettati più alti)
    panchina = panchina.sort_values(by=['punti_proiettati'], ascending=False)
    
    # Scopriamo il modulo
    best_fmt = None
    for fmt, var in module_vars.items():
        if var.varValue == 1:
            best_fmt = f"{fmt[0]}-{fmt[1]}-{fmt[2]}"
            break
            
    total_pts = sum(titolari['punti_proiettati'])
    
    return {
        "success": True,
        "modulo": best_fmt,
        "totale_proiettato": round(total_pts, 2),
        "titolari": titolari,
        "panchina": panchina
    }
