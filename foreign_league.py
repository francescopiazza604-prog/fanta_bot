"""
foreign_league.py — Conversion Engine per Giocatori da Campionati Esteri.

Convertitore statistico e database dei nuovi acquisti esteri giunti in Serie A.
Calcola la resa attesa (FM, Gol, Assist, xG, Titolarità) in Serie A per giocatori
provenienti da Premier League, LaLiga, Bundesliga, Ligue 1, Eredivisie, ecc.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Conversione Lega → Serie A ────────────────────────────────────────────────
# Fatti di conversione tattico/difensivo rispetto alla Serie A
LEAGUE_CONVERSION_FACTORS = {
    'premier_league': {'gol_mult': 1.05, 'assist_mult': 1.00, 'fm_base_shift': +0.15, 'intensity': 'ALTA'},
    'laliga':         {'gol_mult': 1.00, 'assist_mult': 1.00, 'fm_base_shift': 0.00,  'intensity': 'MEDIA'},
    'bundesliga':     {'gol_mult': 0.90, 'assist_mult': 0.90, 'fm_base_shift': -0.10, 'intensity': 'OFFENSIVA'},
    'ligue_1':        {'gol_mult': 0.92, 'assist_mult': 0.95, 'fm_base_shift': -0.05, 'intensity': 'FISICA'},
    'eredivisie':     {'gol_mult': 0.78, 'assist_mult': 0.80, 'fm_base_shift': -0.25, 'intensity': 'MOLTO_OFFENSIVA'},
    'liga_portugal':  {'gol_mult': 0.85, 'assist_mult': 0.88, 'fm_base_shift': -0.15, 'intensity': 'TECNICA'},
    'super_lig':      {'gol_mult': 0.80, 'assist_mult': 0.82, 'fm_base_shift': -0.20, 'intensity': 'MEDIA'},
    'championship':   {'gol_mult': 0.90, 'assist_mult': 0.90, 'fm_base_shift': -0.10, 'intensity': 'ALTA'},
    'mls':            {'gol_mult': 0.75, 'assist_mult': 0.75, 'fm_base_shift': -0.30, 'intensity': 'BASSA'},
}

# ── Database Acquisti Esteri Recenti (Serie A 2024/25 & 2025/26) ─────────────
# Mappa di sicurezza per i giocatori trasferiti dall'estero che non hanno storico Serie A.
FOREIGN_ARRIVALS_DB = {
    "Dovbyk":         {"ruolo": "A", "squadra": "Roma", "lega_orig": "laliga", "gol_orig_p90": 0.65, "assist_orig_p90": 0.22, "titolarita": 0.88, "fm_est": 7.80, "costo_est": 32},
    "Morata":         {"ruolo": "A", "squadra": "Milan", "lega_orig": "laliga", "gol_orig_p90": 0.48, "assist_orig_p90": 0.18, "titolarita": 0.85, "fm_est": 7.60, "costo_est": 28},
    "Taremi":         {"ruolo": "A", "squadra": "Inter", "lega_orig": "liga_portugal", "gol_orig_p90": 0.52, "assist_orig_p90": 0.20, "titolarita": 0.65, "fm_est": 7.30, "costo_est": 18},
    "McTominay":      {"ruolo": "C", "squadra": "Napoli", "lega_orig": "premier_league", "gol_orig_p90": 0.28, "assist_orig_p90": 0.12, "titolarita": 0.90, "fm_est": 7.20, "costo_est": 20},
    "Gilmour":        {"ruolo": "C", "squadra": "Napoli", "lega_orig": "premier_league", "gol_orig_p90": 0.05, "assist_orig_p90": 0.15, "titolarita": 0.70, "fm_est": 6.35, "costo_est": 10},
    "Douglas Luiz":   {"ruolo": "C", "squadra": "Juventus", "lega_orig": "premier_league", "gol_orig_p90": 0.25, "assist_orig_p90": 0.22, "titolarita": 0.82, "fm_est": 7.10, "costo_est": 22},
    "Neres":          {"ruolo": "A", "squadra": "Napoli", "lega_orig": "liga_portugal", "gol_orig_p90": 0.35, "assist_orig_p90": 0.38, "titolarita": 0.75, "fm_est": 7.45, "costo_est": 21},
    "Greenwood":      {"ruolo": "A", "squadra": "Marseille", "lega_orig": "laliga", "gol_orig_p90": 0.38, "assist_orig_p90": 0.22, "titolarita": 0.85, "fm_est": 7.50, "costo_est": 24},
    "Adams":          {"ruolo": "A", "squadra": "Torino", "lega_orig": "championship", "gol_orig_p90": 0.42, "assist_orig_p90": 0.15, "titolarita": 0.80, "fm_est": 7.15, "costo_est": 16},
    "Varane":         {"ruolo": "D", "squadra": "Como", "lega_orig": "premier_league", "gol_orig_p90": 0.04, "assist_orig_p90": 0.02, "titolarita": 0.80, "fm_est": 6.30, "costo_est": 11},
    "Ser Sergi Roberto": {"ruolo": "C", "squadra": "Como", "lega_orig": "laliga", "gol_orig_p90": 0.10, "assist_orig_p90": 0.18, "titolarita": 0.85, "fm_est": 6.40, "costo_est": 12},
    "Nico Williams":  {"ruolo": "A", "squadra": "Athletic", "lega_orig": "laliga", "gol_orig_p90": 0.28, "assist_orig_p90": 0.35, "titolarita": 0.90, "fm_est": 7.70, "costo_est": 30},
    "Hermoso":        {"ruolo": "D", "squadra": "Roma", "lega_orig": "laliga", "gol_orig_p90": 0.06, "assist_orig_p90": 0.05, "titolarita": 0.80, "fm_est": 6.25, "costo_est": 10},
    "Hummels":        {"ruolo": "D", "squadra": "Roma", "lega_orig": "bundesliga", "gol_orig_p90": 0.08, "assist_orig_p90": 0.04, "titolarita": 0.82, "fm_est": 6.35, "costo_est": 12},
    "Conceicao":      {"ruolo": "A", "squadra": "Juventus", "lega_orig": "liga_portugal", "gol_orig_p90": 0.25, "assist_orig_p90": 0.30, "titolarita": 0.72, "fm_est": 7.25, "costo_est": 17},
    "Nico Gonzalez":  {"ruolo": "A", "squadra": "Juventus", "lega_orig": "serie_a", "gol_orig_p90": 0.35, "assist_orig_p90": 0.15, "titolarita": 0.80, "fm_est": 7.40, "costo_est": 22},
    "Tchaouna":       {"ruolo": "A", "squadra": "Lazio", "lega_orig": "serie_a", "gol_orig_p90": 0.22, "assist_orig_p90": 0.15, "titolarita": 0.70, "fm_est": 6.80, "costo_est": 11},
    "Noslin":         {"ruolo": "A", "squadra": "Lazio", "lega_orig": "serie_a", "gol_orig_p90": 0.30, "assist_orig_p90": 0.15, "titolarita": 0.72, "fm_est": 7.00, "costo_est": 14},
}

def convert_foreign_stats(
    gol_p90: float,
    assist_p90: float,
    league: str = 'laliga',
    ruolo: str = 'A'
) -> tuple[float, float, float]:
    """
    Converte le statistiche P90 del campionato estero in stime Serie A.
    Ritorna: (gol_pg_serie_a, assist_pg_serie_a, delta_fm_atteso)
    """
    factors = LEAGUE_CONVERSION_FACTORS.get(league.lower(), {'gol_mult': 0.88, 'assist_mult': 0.90, 'fm_base_shift': -0.10})
    
    gol_sa = gol_p90 * factors['gol_mult']
    assist_sa = assist_p90 * factors['assist_mult']
    
    # Stima voto base secondo ruolo
    base_voto = 6.00 + factors['fm_base_shift']
    if ruolo == 'A':
        fm_est = base_voto + (gol_sa * 3.0) + (assist_sa * 1.0)
    elif ruolo == 'C':
        fm_est = base_voto + (gol_sa * 3.0) + (assist_sa * 1.0) + 0.15
    elif ruolo == 'D':
        fm_est = base_voto + (gol_sa * 3.0) + (assist_sa * 1.0) + 0.10
    else:
        fm_est = 6.00
        
    return float(round(gol_sa, 3)), float(round(assist_sa, 3)), float(round(fm_est, 2))


def enrich_dataset_with_foreign_arrivals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arricchisce il dataframe dei giocatori con le stime dei nuovi acquisti esteri
    se mancano i dati storici Serie A.
    """
    df = df.copy()
    
    # Assicuriamo la presenza delle colonne necessarie
    if 'previsione_ia' not in df.columns:
        df['previsione_ia'] = df.get('fanta_media', 6.0)
    if 'titolarita_pct' not in df.columns:
        df['titolarita_pct'] = 0.75

    for idx, row in df.iterrows():
        nome = str(row.get('nome', '')).strip()
        # Cerchiamo match con il DB degli arrivi esteri
        matched_info = None
        for k, info in FOREIGN_ARRIVALS_DB.items():
            if k.lower() in nome.lower() or nome.lower() in k.lower():
                matched_info = info
                break
                
        if matched_info:
            # Se la fanta_media o la previsione è bassa/assente (tipico dei nuovi acquisti esteri)
            curr_fm = row.get('fanta_media', 0.0)
            if pd.isna(curr_fm) or curr_fm < 5.0 or row.get('previsione_ia', 0.0) < 5.5:
                df.loc[idx, 'previsione_ia'] = matched_info['fm_est']
                df.loc[idx, 'titolarita_pct'] = matched_info['titolarita']
                if 'flag_nuovo_estero' not in df.columns:
                    df['flag_nuovo_estero'] = 0
                df.loc[idx, 'flag_nuovo_estero'] = 1
                logger.info(f"Giocatore estero riconosciuto: {nome} -> Stima FM: {matched_info['fm_est']}")

    return df
