import pandas as pd
import numpy as np
import os
from io import StringIO

RUOLI_VALIDI = {'P', 'D', 'C', 'A'}

def _leggi_csv_robusto(file_path):
    """
    Legge un CSV gestendo righe di larghezza variabile (es. riga di metadati
    seguita da righe con piu campi). Prova UTF-8, poi latin-1.
    """
    lines = None
    for enc in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            with open(file_path, 'r', encoding=enc, errors='strict') as f:
                lines = [l.rstrip('\r\n') for l in f if l.strip()]
            break
        except (UnicodeDecodeError, Exception):
            continue
    if lines is None:
        raise ValueError("Impossibile decodificare il file CSV.")

    best_sep, max_cols = ',', 1
    for sep in (';', ',', '\t'):
        cols = max((len(l.split(sep)) for l in lines), default=1)
        if cols > max_cols:
            max_cols, best_sep = cols, sep

    padded = '\n'.join(
        l + best_sep * max(0, max_cols - 1 - l.count(best_sep))
        for l in lines
    )
    return pd.read_csv(
        StringIO(padded), header=None, sep=best_sep,
        names=range(max_cols), dtype=str
    )


def import_real_quotazioni(file_path):
    """
    Importa il file delle quotazioni o statistiche con rilevamento automatico dell'header.
    Supporta xlsx e csv con separatori variabili e diverse convenzioni di colonne.
    """
    ALIASES_RUOLO = {'r', 'ruolo', 'role', 'pos', 'posizione'}
    ALIASES_NOME = {'nome', 'name', 'giocatore', 'player', 'cognome', 'calciatore'}
    ALIASES_COSTO = {'qt.a', 'qt. a', 'qt.i', 'qt. i', 'quota', 'quotazione', 'costo', 'cost',
                     'prezzo', 'q.a', 'qta', 'qt_a', 'valore', 'value', 'costo_iniziale', 'fvm', 'fvm m', 'fvm_m'}
    ALIASES_FM = {'fm', 'fantamedia', 'fanta_media', 'fanta media', 'media fanta',
                  'mf', 'fm m', 'fm_m', 'f.m.', 'f.m', 'fm_reale', 'fanta_media_reale',
                  'fm_25_26', 'fm_24_25', 'fm_23_24', 'fanta_media_25_26'}
    ALIASES_FVM = {'fvm', 'fvm_m', 'valore', 'valore_acquisto'}
    ALIASES_MV = {'mv', 'mediavoto', 'media_voto', 'media voto', 'voto media',
                  'm.v.', 'm.v', 'media_voto_reale'}
    ALIASES_PRESENZE = {'pg', 'presenze', 'partite', 'partite_giocate', 'p.g.', 'p.g'}
    ALIASES_GOL = {'g', 'gol', 'gol_segnati', 'reti', 'g.s.'}
    ALIASES_ASSIST = {'a', 'assist', 'ass'}

    try:
        if file_path.endswith('.xlsx'):
            df_raw = pd.read_excel(file_path, header=None)
        else:
            df_raw = _leggi_csv_robusto(file_path)

        # Trova la riga header
        header_idx = None
        for i, row in df_raw.head(15).iterrows():
            row_lower = [str(x).strip().lower() for x in row.values]
            has_nome = any(v in ALIASES_NOME for v in row_lower)
            has_ruolo_col = any(v in ALIASES_RUOLO for v in row_lower)
            has_costo = any(v in ALIASES_COSTO for v in row_lower)
            if has_nome and (has_ruolo_col or has_costo):
                header_idx = i
                break

        if header_idx is None:
            for i, row in df_raw.head(15).iterrows():
                row_str = [str(x).strip() for x in row.values]
                if 'R' in row_str or 'Nome' in row_str or 'FM' in row_str:
                    header_idx = i
                    break
        if header_idx is None:
            header_idx = 0

        df = df_raw.iloc[header_idx:].copy()
        df.columns = [str(c).strip() for c in df.iloc[0].values]
        df = df.iloc[1:].reset_index(drop=True)
        df = df.dropna(how='all').reset_index(drop=True)

        # Mappa colonne ai nomi interni (case-insensitive)
        rename_map = {}
        for col in df.columns:
            col_low = col.strip().lower()
            if col_low in ALIASES_RUOLO and 'ruolo' not in rename_map.values():
                rename_map[col] = 'ruolo'
            elif col_low in ALIASES_NOME and 'nome' not in rename_map.values():
                rename_map[col] = 'nome'
            elif col_low in ALIASES_COSTO and 'costo_iniziale' not in rename_map.values():
                rename_map[col] = 'costo_iniziale'
            elif col_low in ALIASES_FM and 'fanta_media' not in rename_map.values():
                rename_map[col] = 'fanta_media'
            elif col_low in ALIASES_FVM and 'fvm' not in rename_map.values():
                rename_map[col] = 'fvm'
            elif col_low in ALIASES_MV and 'media_voto' not in rename_map.values():
                rename_map[col] = 'media_voto'
            elif col_low in ALIASES_PRESENZE and 'presenze' not in rename_map.values():
                rename_map[col] = 'presenze'
            elif col_low in ALIASES_GOL and 'gol' not in rename_map.values():
                rename_map[col] = 'gol'
            elif col_low in ALIASES_ASSIST and 'assist' not in rename_map.values():
                rename_map[col] = 'assist'

        df = df.rename(columns=rename_map)

        # Fallbacks per contenuto
        if 'ruolo' not in df.columns:
            for col in df.columns:
                if col in ('nome', 'costo_iniziale', 'fanta_media', 'fvm'):
                    continue
                unique_vals = set(df[col].dropna().astype(str).str.upper().unique())
                if unique_vals.issubset(RUOLI_VALIDI | {'nan', ''}) and len(unique_vals & RUOLI_VALIDI) >= 2:
                    df = df.rename(columns={col: 'ruolo'})
                    break

        if 'nome' not in df.columns:
            for col in df.columns:
                if col in ('ruolo', 'costo_iniziale', 'fanta_media', 'fvm'):
                    continue
                serie = df[col].astype(str)
                numeric_ratio = pd.to_numeric(serie, errors='coerce').notna().mean()
                if numeric_ratio < 0.3 and serie.str.len().mean() > 4 and serie.nunique() > 5:
                    df = df.rename(columns={col: 'nome'})
                    break

        # Pulizia e conversione numerica dei campi
        if 'costo_iniziale' in df.columns:
            df['costo_iniziale'] = pd.to_numeric(
                df['costo_iniziale'].astype(str).str.replace(',', '.'), errors='coerce'
            ).fillna(1).astype(int)

        if 'fvm' in df.columns:
            df['fvm'] = pd.to_numeric(
                df['fvm'].astype(str).str.replace(',', '.'), errors='coerce'
            )

        if 'fanta_media' in df.columns:
            df['fanta_media'] = pd.to_numeric(
                df['fanta_media'].astype(str).str.replace(',', '.'), errors='coerce'
            )
            # Se fanta_media contiene valori > 20 (es. FVM scambiato per FM), non è una vera FantaMedia (3-10)
            if df['fanta_media'].max() > 20:
                df['fanta_media'] = np.nan
        else:
            df['fanta_media'] = np.nan

        if 'media_voto' in df.columns:
            df['media_voto'] = pd.to_numeric(
                df['media_voto'].astype(str).str.replace(',', '.'), errors='coerce'
            )

        if 'ruolo' in df.columns:
            df['ruolo'] = df['ruolo'].astype(str).str.strip().str.upper()
            df = df[df['ruolo'].isin(RUOLI_VALIDI)].reset_index(drop=True)

        # Garantisci colonne minime
        for col in ('ruolo', 'nome', 'costo_iniziale', 'fanta_media'):
            if col not in df.columns:
                df[col] = 1 if col == 'costo_iniziale' else np.nan if col == 'fanta_media' else 'N/D'

        if df.empty:
            print("Attenzione: nessun giocatore valido trovato nel file.")
            return None

        # Ritorna tutte le colonne rilevanti presenti
        keep_cols = [c for c in ['ruolo', 'nome', 'costo_iniziale', 'fanta_media', 'fvm', 'media_voto', 'presenze', 'gol_pg', 'assist_pg', 'titolarita_pct'] if c in df.columns]
        return df[keep_cols].copy()

    except Exception as e:
        print(f"Errore nell'importazione: {e}")
        return None
