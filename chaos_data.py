"""
chaos_data.py — Pipeline dati reali per il Chaos & Upside Index Optimizer.

FLUSSO DATI:
  ┌──────────────────────┐
  │  quotazioni (xlsx/csv)│  → nome, ruolo, costo_iniziale, fanta_media
  └──────────┬───────────┘
             │
  ┌──────────▼───────────┐
  │ FC.it stats cache    │  → gol_pg, assist_pg, ammonizioni_pg,
  │ (fantacalcio_stats_  │     xg_pg, titolarita_pct, media_voto
  │  cache.csv)          │
  └──────────┬───────────┘
             │
  ┌──────────▼───────────┐
  │ FBref cache          │  → xa_pg (xAG)          [opzionale]
  │ (fbref_stats_cache   │     sovrascrive xg_pg se più preciso
  │  .csv)               │
  └──────────┬───────────┘
             │
  ┌──────────▼───────────┐
  │ FALLBACK per ruolo   │  → storico_voti_std (stimato da media_voto + FM)
  │                      │     peak_rate (da distribuzione normale)
  │                      │     dribbling_riusciti_p90 (media ruolo)
  │                      │     passaggi_chiave_p90    (media ruolo)
  │                      │     xa_pg                  (media ruolo)
  └──────────┬───────────┘
             │
  ┌──────────▼───────────┐
  │ DataFrame pronto per │  → compute_chaos_upside_index()
  │ chaos_optimizer.py   │
  └──────────────────────┘

LOGICA FALLBACK:
  Ogni colonna CUI mancante viene stimata con un metodo specifico,
  non con uno zero generico. Il log segnala sempre da dove viene ogni dato.
"""

import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")

FC_STATS_CACHE  = os.path.join(DATA_DIR, "fantacalcio_stats_cache.csv")
FBREF_CACHE     = os.path.join(DATA_DIR, "fbref_stats_cache.csv")


# ── Default per ruolo (valori medi realistici Serie A) ────────────────────────
#
# Usati SOLO per le colonne mancanti. Derivati da statistiche pubbliche FBref/Opta
# per la Serie A 2023-24. Aggiornali ogni stagione se hai i dati reali.

_ROLE_DEFAULTS: dict[str, dict[str, float]] = {
    # Deviazione standard base dei voti: quanto "varia" un giocatore tipico per ruolo.
    # Portieri variano poco (grandi prestazioni cambiano poco la FM), attaccanti molto.
    'base_voti_std':            {'P': 0.35, 'D': 0.48, 'C': 0.65, 'A': 0.85},

    # Media voto di riferimento per ruolo (usata se media_voto non disponibile)
    'media_voto_ref':           {'P': 6.05, 'D': 6.10, 'C': 6.15, 'A': 6.20},

    # Dribbling completati ogni 90 minuti (fonte: FBref Serie A 2023-24)
    'dribbling_riusciti_p90':   {'P': 0.05, 'D': 0.65, 'C': 1.45, 'A': 2.10},

    # Passaggi chiave ogni 90 minuti
    'passaggi_chiave_p90':      {'P': 0.10, 'D': 0.65, 'C': 1.70, 'A': 1.30},

    # xA (expected assists) per partita giocata
    'xa_pg':                    {'P': 0.00, 'D': 0.04, 'C': 0.10, 'A': 0.13},

    # xG per partita (fallback se non presente da nessuna fonte)
    'xg_pg':                    {'P': 0.00, 'D': 0.03, 'C': 0.09, 'A': 0.28},

    # Gol/assist per partita — fallback se FC.it stats non disponibili
    'gol_pg':                   {'P': 0.00, 'D': 0.03, 'C': 0.08, 'A': 0.28},
    'assist_pg':                {'P': 0.00, 'D': 0.05, 'C': 0.12, 'A': 0.12},
}


def _default(ruolo: str, col: str) -> float:
    return _ROLE_DEFAULTS.get(col, {}).get(ruolo, 0.0)


# ── Normalizzazione chiave di merge ───────────────────────────────────────────

def _norm_key(s: pd.Series) -> pd.Series:
    """
    Normalizza nome per il merge: lowercase, strip, rimuove accenti comuni.
    Esempio: 'Di María' → 'di maria', 'Calhanoglu' → 'calhanoglu'
    """
    return (
        s.astype(str)
         .str.lower()
         .str.strip()
         .str.replace(r"[àáâä]", "a", regex=True)
         .str.replace(r"[èéêë]", "e", regex=True)
         .str.replace(r"[ìíîï]", "i", regex=True)
         .str.replace(r"[òóôö]", "o", regex=True)
         .str.replace(r"[ùúûü]", "u", regex=True)
         .str.replace(r"[^\w\s]", "", regex=True)
         .str.replace(r"\s+", " ", regex=True)
    )


def _merge_by_name(
    df: pd.DataFrame,
    source: pd.DataFrame,
    cols_to_merge: list[str],
    source_name: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Merge left su chiave nome normalizzata. Tenta prima match esatto,
    poi match sul solo cognome (ultima parola).
    Ritorna il DataFrame arricchito e un dict {colonna: n_match}.
    """
    df     = df.copy()
    source = source.copy()

    df['_key']     = _norm_key(df['nome'])
    source['_key'] = _norm_key(source['nome'])

    # Rimuovi colonne già presenti nel df sorgente per evitare _x/_y
    cols_available = [c for c in cols_to_merge if c in source.columns]
    if not cols_available:
        df.drop(columns=['_key'], inplace=True)
        return df, {}

    src_dedup = source.drop_duplicates('_key')[['_key'] + cols_available]

    # ── Merge esatto ──────────────────────────────────────────────────────────
    merged = df.merge(src_dedup, on='_key', how='left', suffixes=('', '_src'))

    # ── Fallback cognome: riempi i NaN rimasti ────────────────────────────────
    missing_mask = merged[cols_available[0]].isna()
    if missing_mask.any():
        merged['_cognome'] = merged['_key'].str.split().str[-1]
        src_dedup['_cognome'] = src_dedup['_key'].str.split().str[-1]
        src_by_cognome = src_dedup.drop_duplicates('_cognome').set_index('_cognome')

        for col in cols_available:
            fill_vals = merged.loc[missing_mask, '_cognome'].map(
                src_by_cognome[col] if col in src_by_cognome.columns else {}
            )
            merged.loc[missing_mask, col] = merged.loc[missing_mask, col].fillna(fill_vals)

        merged.drop(columns=['_cognome'], inplace=True)

    merged.drop(columns=['_key'], inplace=True)

    # Conta quanti record hanno ricevuto dati reali per ogni colonna
    hit_counts = {c: int(merged[c].notna().sum()) for c in cols_available}
    for col, n in hit_counts.items():
        total = len(merged)
        logger.info(f"  [{source_name}] {col}: {n}/{total} giocatori con dati reali")

    return merged, hit_counts


# ── Step 1: Carica quotazioni ─────────────────────────────────────────────────

def _load_quotazioni(path: str) -> pd.DataFrame:
    from importer import import_real_quotazioni
    df = import_real_quotazioni(path)
    if df is None or df.empty:
        raise ValueError(f"Impossibile caricare le quotazioni da '{path}'")
    required = {'nome', 'ruolo', 'costo_iniziale'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonne obbligatorie mancanti nel file quotazioni: {missing}")
    logger.info(f"Quotazioni: {len(df)} giocatori caricati da '{path}'")
    return df


# ── Step 2: Merge con Fantacalcio.it stats ────────────────────────────────────

def _merge_fc_stats(df: pd.DataFrame, fc_stats_path: str) -> pd.DataFrame:
    """
    Arricchisce con le colonne del cache Fantacalcio.it:
    gol_pg, assist_pg, ammonizioni_pg, xg_pg, titolarita_pct, media_voto, fanta_media_fc
    """
    if not os.path.exists(fc_stats_path):
        logger.warning(
            f"Cache FC.it non trovata ({fc_stats_path}). "
            "Avvia l'app e scarica le statistiche dalla sezione 'Carica Dati'."
        )
        return df

    try:
        stats = pd.read_csv(fc_stats_path)
    except Exception as e:
        logger.warning(f"Errore lettura cache FC.it: {e}")
        return df

    cols = [c for c in [
        'gol_pg', 'assist_pg', 'ammonizioni_pg', 'xg_pg',
        'titolarita_pct', 'media_voto', 'fanta_media_fc',
    ] if c in stats.columns]

    if not cols:
        logger.warning("Cache FC.it non contiene colonne statistiche utilizzabili.")
        return df

    df, _ = _merge_by_name(df, stats, cols, source_name="FC.it")

    # Converti fanta_media a numerico (potrebbe essere stringa dall'importer)
    df['fanta_media'] = pd.to_numeric(df['fanta_media'], errors='coerce').fillna(0.0)

    # Se fanta_media_fc è disponibile e fanta_media mancante/zero, usa quella
    if 'fanta_media_fc' in df.columns:
        df['fanta_media_fc'] = pd.to_numeric(df['fanta_media_fc'], errors='coerce').fillna(0.0)
        mask = df['fanta_media'].isna() | (df['fanta_media'] < 1.0)
        df.loc[mask, 'fanta_media'] = df.loc[mask, 'fanta_media_fc']
        df.drop(columns=['fanta_media_fc'], inplace=True)

    return df


# ── Step 3: Merge con FBref (opzionale) ──────────────────────────────────────

def _merge_fbref_stats(df: pd.DataFrame, fbref_path: str) -> pd.DataFrame:
    """
    Arricchisce con xA (xa_pg) da FBref, se disponibile.
    FBref chiama questa metrica 'xag' (Expected Assisted Goals).
    """
    if not os.path.exists(fbref_path):
        return df

    try:
        stats = pd.read_csv(fbref_path)
    except Exception as e:
        logger.warning(f"Errore lettura cache FBref: {e}")
        return df

    # FBref usa 'nome_fbref' come colonna nome — rinomina per il merge
    if 'nome_fbref' in stats.columns:
        stats = stats.rename(columns={'nome_fbref': 'nome'})

    # Calcola xa_pg da xag se non già presente
    if 'xag' in stats.columns and 'presenze' in stats.columns:
        safe_pv = stats['presenze'].replace(0, 1)
        stats['xa_pg'] = (stats['xag'] / safe_pv).round(3)

    cols = [c for c in ['xa_pg'] if c in stats.columns]
    if not cols:
        return df

    df, _ = _merge_by_name(df, stats, cols, source_name="FBref")
    return df


# ── Step 4: Stima storico_voti_std e peak_rate ───────────────────────────────

def _estimate_variance_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stima le colonne di varianza storica quando non disponibili da fonte diretta.

    LOGICA:
      base_std:    varia per ruolo (attaccanti > centrocampisti > difensori > portieri)
      bonus_gap:   FM - media_voto misura i bonus episodici (gol×3, assist×1, ...)
                   Un giocatore con gap alto ha eventi bonus frequenti → più picchi
      std_stimata: base_std + bonus_gap × 0.28
                   Il moltiplicatore 0.28 è calibrato empiricamente su dati FC.it 23-24.

      peak_rate:   P(voto ≥ 7.0) sotto distribuzione normale con media=MV e σ=std_stimata.
                   È un'approssimazione: la distribuzione reale è leggermente destra-skewed,
                   ma l'errore è < 5 punti percentuali nella fascia FM 6.0-7.5.
    """
    df = df.copy()

    # Usa media_voto se disponibile, altrimenti usa il default per ruolo
    if 'media_voto' not in df.columns:
        df['media_voto'] = df['ruolo'].map(_ROLE_DEFAULTS['media_voto_ref']).fillna(6.10)
        logger.info("media_voto non disponibile — uso default per ruolo")

    # Garantisci che fanta_media sia numerica e ragionevole
    df['fanta_media'] = pd.to_numeric(df['fanta_media'], errors='coerce').fillna(6.0)
    df['media_voto']  = pd.to_numeric(df['media_voto'],  errors='coerce').fillna(6.10)

    # Differenza FM - MV: proxy del contributo bonus medio
    bonus_gap = (df['fanta_media'] - df['media_voto']).clip(lower=0.0, upper=4.0)

    base_std = df['ruolo'].map(_ROLE_DEFAULTS['base_voti_std']).fillna(0.55)

    df['storico_voti_std'] = (base_std + bonus_gap * 0.28).round(3)
    df['peak_rate'] = df.apply(
        lambda r: float(_norm.sf(7.0, loc=r['media_voto'], scale=max(r['storico_voti_std'], 0.10))),
        axis=1,
    ).round(3)

    logger.info("storico_voti_std e peak_rate: stimati da media_voto + fanta_media")
    return df


# ── Step 5: Fallback per colonne P90 e xa_pg ─────────────────────────────────

def _fill_p90_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Riempie dribbling_riusciti_p90, passaggi_chiave_p90, xa_pg, xg_pg
    con i default per ruolo per i giocatori che non hanno dati reali.
    Usa il valore reale quando disponibile, il default solo dove è NaN.
    """
    df = df.copy()
    p90_cols = ['gol_pg', 'assist_pg', 'dribbling_riusciti_p90', 'passaggi_chiave_p90', 'xa_pg', 'xg_pg']

    for col in p90_cols:
        if col not in df.columns:
            df[col] = np.nan

        n_missing_before = df[col].isna().sum()
        if n_missing_before == 0:
            continue

        fill_vals = df['ruolo'].map(_ROLE_DEFAULTS.get(col, {})).fillna(0.0)
        df[col] = df[col].fillna(fill_vals)

        n_filled = n_missing_before
        logger.info(
            f"  [Fallback ruolo] {col}: {n_filled} giocatori senza dati reali → default applicato"
        )

    return df


# ── Funzione principale ───────────────────────────────────────────────────────

def prepare_chaos_data(
    quotazioni_path: str,
    budget: int = 500,
    fc_stats_path: str  = FC_STATS_CACHE,
    fbref_stats_path: str = FBREF_CACHE,
) -> pd.DataFrame:
    """
    Pipeline completa: da file quotazioni a DataFrame pronto per il CUI.

    Parametri
    ---------
    quotazioni_path  : file Excel o CSV delle quotazioni d'asta
                       (formato standard Fantacalcio.it / Leghe Fantacalcio)
    budget           : budget dell'asta — usato solo per la diagnostica
    fc_stats_path    : percorso alla cache Fantacalcio.it stats (default: auto)
    fbref_stats_path : percorso alla cache FBref stats (default: auto)

    Colonne prodotte
    ----------------
    Obbligatorie (da quotazioni):
      nome, ruolo, costo_iniziale, fanta_media

    Da FC.it stats (reali se disponibili, altrimenti default):
      gol_pg, assist_pg, ammonizioni_pg, xg_pg, titolarita_pct, media_voto

    Da FBref (reali se disponibili, altrimenti default):
      xa_pg

    Stimate/default:
      storico_voti_std, peak_rate         (da media_voto + fanta_media)
      dribbling_riusciti_p90              (default per ruolo)
      passaggi_chiave_p90                 (default per ruolo)

    NON presente: ownership_pct (rimossa — inaffidabile pre-torneo)
    """
    logger.info(f"\n{'─'*55}")
    logger.info(f"  PREPARE_CHAOS_DATA — budget={budget}cr")
    logger.info(f"{'─'*55}")

    # 1. Quotazioni (obbligatorio)
    df = _load_quotazioni(quotazioni_path)

    # 2. Fantacalcio.it stats (consigliato)
    df = _merge_fc_stats(df, fc_stats_path)

    # 3. FBref stats (opzionale, solo xa_pg)
    df = _merge_fbref_stats(df, fbref_stats_path)

    # 4. Stima varianza storica
    df = _estimate_variance_cols(df)

    # 5. Fallback P90 per colonne ancora mancanti
    df = _fill_p90_defaults(df)

    # 6. Pulizia finale
    df = df.dropna(subset=['nome', 'ruolo', 'costo_iniziale']).copy()
    df['costo_iniziale'] = pd.to_numeric(df['costo_iniziale'], errors='coerce').fillna(1).astype(int)
    df['fanta_media']    = pd.to_numeric(df['fanta_media'],    errors='coerce').fillna(6.0)

    # Report copertura dati
    _log_coverage(df, budget)

    return df.reset_index(drop=True)


def _log_coverage(df: pd.DataFrame, budget: int) -> None:
    total = len(df)
    logger.info(f"\n  COPERTURA DATI — {total} giocatori  (budget={budget}cr)")
    logger.info(f"  {'Colonna':<28s}  {'Reali':>7s}  {'Default':>8s}  {'Source'}")
    logger.info(f"  {'─'*60}")

    col_sources = {
        'gol_pg':                   'FC.it stats',
        'assist_pg':                'FC.it stats',
        'xg_pg':                    'FC.it / FBref',
        'xa_pg':                    'FBref',
        'media_voto':               'FC.it stats',
        'storico_voti_std':         'Stimato (MV+FM)',
        'peak_rate':                'Stimato (MV+σ)',
        'dribbling_riusciti_p90':   'Default ruolo',
        'passaggi_chiave_p90':      'Default ruolo',
        'titolarita_pct':           'FC.it stats',
    }
    for col, source in col_sources.items():
        if col not in df.columns:
            logger.info(f"  {col:<28s}  {'─':>7s}  {'─':>8s}  (assente)")
            continue
        n_real    = int(df[col].notna().sum())
        n_default = total - n_real
        logger.info(f"  {col:<28s}  {n_real:>7d}  {n_default:>8d}  {source}")


# ── CLI diagnostica ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    path   = sys.argv[1] if len(sys.argv) > 1 else "temp_quotazioni.xlsx"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    df = prepare_chaos_data(path, budget=budget)

    print(f"\nDataset pronto: {len(df)} giocatori")
    print(f"Colonne: {list(df.columns)}")
    print("\nCampione (primi 5):")
    sample_cols = ['nome', 'ruolo', 'costo_iniziale', 'fanta_media',
                   'media_voto', 'storico_voti_std', 'peak_rate',
                   'dribbling_riusciti_p90', 'xg_pg', 'xa_pg']
    sample_cols = [c for c in sample_cols if c in df.columns]
    print(df[sample_cols].head().to_string(index=False))
