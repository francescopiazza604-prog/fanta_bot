"""
set_pieces.py — Rigoristi, Punizionisti e Tiratori di Piazzati Serie A.

Modulo per calcolare l'impatto dei calci di rigore (+3 pt) e delle punizioni dirette
sulle stime di FantaMedia attesa.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Gerarchia Rigoristi e Tiratori Serie A (Stagione Corrente) ─────────────────
# 1° rigorista: quota ~70% dei rigori della squadra
# 2° rigorista: quota ~25% dei rigori della squadra
# Media rigori pro-squadra in Serie A: ~6.5 a stagione
SET_PIECES_DB = {
    'Inter':        {'rigoristi': ['Calhanoglu', 'Lautaro', 'Taremi'], 'punizioni': ['Calhanoglu', 'Dimarco']},
    'Juventus':     {'rigoristi': ['Vlahovic', 'Koopmeiners', 'Douglas Luiz'], 'punizioni': ['Vlahovic', 'Koopmeiners']},
    'Milan':        {'rigoristi': ['Pulisic', 'Morata', 'Hernandez Theo'], 'punizioni': ['Hernandez Theo', 'Pulisic']},
    'Napoli':       {'rigoristi': ['Lukaku', 'Kvaratskhelia', 'Politano'], 'punizioni': ['Kvaratskhelia', 'Politano']},
    'Roma':         {'rigoristi': ['Dybala', 'Dovbyk', 'Pellegrini'], 'punizioni': ['Dybala', 'Pellegrini']},
    'Lazio':        {'rigoristi': ['Zaccagni', 'Castellanos', 'Pedro'], 'punizioni': ['Zaccagni', 'Tchaouna']},
    'Atalanta':     {'rigoristi': ['Retegui', 'Lookman', 'De Ketelaere'], 'punizioni': ['Lookman', 'Samardzic']},
    'Fiorentina':   {'rigoristi': ['Gudmundsson', 'Kean', 'Biraghi'], 'punizioni': ['Biraghi', 'Gudmundsson']},
    'Bologna':      {'rigoristi': ['Orsolini', 'Castro', 'Dallinga'], 'punizioni': ['Orsolini', 'Lykogiannis']},
    'Torino':       {'rigoristi': ['Zapata', 'Sanabria', 'Vlasic'], 'punizioni': ['Vlasic', 'Ilic']},
    'Genoa':        {'rigoristi': ['Pinamonti', 'Malinovskyi', 'Messias'], 'punizioni': ['Malinovskyi', 'Messias']},
    'Udinese':      {'rigoristi': ['Thauvin', 'Lucca', 'Sanchez'], 'punizioni': ['Thauvin', 'Lovric']},
    'Verona':       {'rigoristi': ['Tengstedt', 'Mosquera', 'Suslov'], 'punizioni': ['Suslov', 'Duda']},
    'Cagliari':     {'rigoristi': ['Piccoli', 'Viola', 'Lapadula'], 'punizioni': ['Viola', 'Marin']},
    'Monza':        {'rigoristi': ['Pessina', 'Djuric', 'Caprari'], 'punizioni': ['Caprari', 'Kyriakopoulos']},
    'Lecce':        {'rigoristi': ['Krstovic', 'Oudin', 'Rafia'], 'punizioni': ['Oudin', 'Gallo']},
    'Empoli':       {'rigoristi': ['Esposito', 'Colombo', 'Zurkowski'], 'punizioni': ['Esposito', 'Pezzella']},
    'Como':         {'rigoristi': ['Belotti', 'Strefezza', 'Cutrone'], 'punizioni': ['Strefezza', 'Da Cunha']},
    'Parma':        {'rigoristi': ['Man', 'Bonny', 'Hernani'], 'punizioni': ['Man', 'Bernabe']},
    'Venezia':      {'rigoristi': ['Pohjanpalo', 'Gytkjaer', 'Busio'], 'punizioni': ['Nicolussi Caviglia', 'Busio']},
}

AVG_PENALTIES_PER_TEAM = 6.5   # Media rigori a favore a stagione per club

def get_player_set_piece_role(nome: str, squadra: str) -> tuple[int, bool]:
    """
    Ritorna: (rigo_rank, is_punizionista)
    rigo_rank: 1 (primo rigorista), 2 (secondo), 3 (terzo), 0 (non rigorista)
    is_punizionista: True/False
    """
    squadra_clean = str(squadra).strip().title()
    nome_clean = str(nome).strip()
    
    # Match squadra
    info = None
    for team, data in SET_PIECES_DB.items():
        if team.lower() in squadra_clean.lower() or squadra_clean.lower() in team.lower():
            info = data
            break
            
    if not info:
        return 0, False
        
    rigo_rank = 0
    for idx, reg in enumerate(info['rigoristi'], start=1):
        if reg.lower() in nome_clean.lower() or nome_clean.lower() in reg.lower():
            rigo_rank = idx
            break
            
    is_pun = any(p.lower() in nome_clean.lower() or nome_clean.lower() in p.lower() for p in info['punizioni'])
    
    return rigo_rank, is_pun


def apply_set_pieces_boost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola il bonus atteso in FantaMedia per rigoristi e tiratori di punizione.
    - 1° rigorista: ~4.5 rigori attesi trasformati (4.5 * 3pt / 38g) -> +0.35 FM
    - 2° rigorista: ~1.5 rigori attesi (1.5 * 3pt / 38g) -> +0.12 FM
    - Tiratore punizioni: +0.08 FM
    """
    df = df.copy()
    if 'previsione_ia' not in df.columns:
        df['previsione_ia'] = df.get('fanta_media', 6.0)

    bonus_rigo_list = []
    bonus_pun_list = []
    
    for idx, row in df.iterrows():
        nome = row.get('nome', '')
        squadra = row.get('squadra', '')
        rank, is_pun = get_player_set_piece_role(nome, squadra)
        
        bonus_rigo = 0.0
        if rank == 1:
            bonus_rigo = 0.38   # ~4.8 rigori segnati su 38 giornate (+14.4 pt / 38 = +0.38 FM)
        elif rank == 2:
            bonus_rigo = 0.15   # ~1.9 rigori segnati (+5.7 pt / 38 = +0.15 FM)
        elif rank == 3:
            bonus_rigo = 0.06
            
        bonus_pun = 0.08 if is_pun else 0.0
        
        bonus_rigo_list.append(bonus_rigo)
        bonus_pun_list.append(bonus_pun)

    df['bonus_rigorista'] = bonus_rigo_list
    df['bonus_punizioni'] = bonus_pun_list
    df['previsione_ia'] += df['bonus_rigorista'] + df['bonus_punizioni']
    
    logger.info("Applicato boost Rigoristi & Punizionisti a tutti i giocatori.")
    return df
