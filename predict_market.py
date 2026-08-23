"""
predict_market.py — Market-Aware AI Engine per Calciomercato Serie A.

Integra dati reali di trasferimento (Transfermarkt), reputazione club,
conversione campionati esteri, rigoristi/punizionisti e impatto tattico.
Rimuove completamente qualsiasi dato casuale/mock.
"""

import os
import logging
import pandas as pd
import numpy as np

from foreign_league import enrich_dataset_with_foreign_arrivals
from set_pieces import apply_set_pieces_boost
from scraper_transfermarkt import get_coach_tactical_bonus, get_coach_style

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRANSFERS_CACHE = os.path.join(DATA_DIR, "transfers_cache.csv")

def apply_real_market_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applica l'intelligenza di mercato reale al dataset:
    1. Foreign League Conversion (nuovi arrivi da Premier, LaLiga, ecc.)
    2. Modificatori da trasferimenti reali (Transfermarkt cache)
    3. Tactical Coach Bonus (es. quinti con Gasperini/Inzaghi, trequartisti con Conte)
    4. Rigoristi e Tiratori di Piazzati (+3 pt bonus)
    """
    df = df.copy()
    
    # 1. Arricchimento Nuovi Arrivi Esteri
    df = enrich_dataset_with_foreign_arrivals(df)

    # 2. Caricamento cache trasferimenti da Transfermarkt (se esistente)
    transfers_df = None
    if os.path.exists(TRANSFERS_CACHE):
        try:
            transfers_df = pd.read_csv(TRANSFERS_CACHE)
            logger.info("Cache trasferimenti caricata per l'analisi di mercato.")
        except Exception as e:
            logger.warning(f"Errore lettura cache trasferimenti: {e}")

    # Inizializziamo le colonne di mercato se mancanti
    if 'cambio_squadra' not in df.columns:
        df['cambio_squadra'] = 0
    if 'upgrade_squadra' not in df.columns:
        df['upgrade_squadra'] = 0
    if 'coach_bonus' not in df.columns:
        df['coach_bonus'] = 0.0

    # Processiamo ciascun giocatore
    for idx, row in df.iterrows():
        nome = str(row.get('nome', '')).strip()
        squadra = str(row.get('squadra', '')).strip()
        ruolo = str(row.get('ruolo', 'C')).strip()

        # Tactical coach bonus
        c_bonus = get_coach_tactical_bonus(squadra, ruolo)
        df.loc[idx, 'coach_bonus'] = c_bonus

        # Match trasferimenti reali
        if transfers_df is not None and isinstance(transfers_df, pd.DataFrame) and not transfers_df.empty:
            tm_col = None
            for candidate in ['nome_tm', 'nome', 'nome_giocatore', 'giocatore']:
                if candidate in transfers_df.columns:
                    tm_col = candidate
                    break
            if tm_col and nome:
                match = transfers_df[transfers_df[tm_col].astype(str).str.lower() == nome.lower()]
                if not match.empty:
                    t_row = match.iloc[0]
                    df.loc[idx, 'cambio_squadra'] = int(t_row.get('cambio_squadra', 1))
                    df.loc[idx, 'upgrade_squadra'] = int(t_row.get('upgrade_squadra', 0))

    # 3. Applicazione bonus/malus tattici sulla 'previsione_ia'
    # Base FM o previsione corrente
    if 'previsione_ia' not in df.columns:
        if 'fanta_media' in df.columns:
            df['previsione_ia'] = df['fanta_media'].fillna(6.0)
        else:
            df['previsione_ia'] = 6.0
    else:
        df['previsione_ia'] = df['previsione_ia'].fillna(6.0)

    # Applicazione Coach Bonus (es. +0.15 a +0.20 FM)
    df['previsione_ia'] += df['coach_bonus']

    # Upgrade di squadra: andare in un Top Club (es. Inter, Juve, Milan, Napoli, Atalanta) aumenta la produzione
    mask_upgrade_pos = df['upgrade_squadra'] == 1
    df.loc[mask_upgrade_pos, 'previsione_ia'] *= 1.04

    mask_downgrade = df['upgrade_squadra'] == -1
    df.loc[mask_downgrade, 'previsione_ia'] *= 0.96

    # 4. Rigoristi e Tiratori di Piazzati (+3 pt bonus)
    df = apply_set_pieces_boost(df)

    # Arrotondamento della previsione finale
    df['previsione_ia'] = df['previsione_ia'].round(2)
    return df


if __name__ == "__main__":
    # Self-test
    test_path = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        res = apply_real_market_logic(df_test)
        print("Market-Aware Logic completata con successo! Prime 5 righe:")
        print(res[['nome', 'ruolo', 'squadra', 'previsione_ia', 'bonus_rigorista']].head())
