"""
Genera un dataset completo di giocatori Serie A partendo da FBref.
- Scarica stats reali (gol, assist, xG, presenze, ammonizioni, titolarità)
- Stima la FantaMedia usando la formula Fantacalcio (MV + bonus - malus)
- Stima il costo asta proporzionale alla qualità e al ruolo
- Output: CSV pronto per ottimizzatore e backtest (~400-500 giocatori)
"""
import os
import logging

import numpy as np
import pandas as pd

from scraper_stats import fetch_and_save_stats, load_stats_for_season

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Parametri formula Fantacalcio ─────────────────────────────────────────────
GOL_BONUS   = {'P': 0.0, 'D': 4.0, 'C': 3.0, 'A': 3.0}
COSTO_MAX   = {'P': 25,  'D': 35,  'C': 45,  'A': 60}
COSTO_MIN   = 1

# ── Mapping posizione FBref → ruolo Fantacalcio ───────────────────────────────
def _map_role(fbref_pos: str) -> str:
    pos = str(fbref_pos).upper().strip()
    if 'GK' in pos:
        return 'P'
    if 'DF' in pos:
        return 'D'
    if 'FW' in pos:
        return 'A'
    if 'MF' in pos:
        # centrocampisti offensivi (MF,FW) → attaccante
        return 'A' if 'FW' in pos else 'C'
    return 'C'


# ── Stima MV ─────────────────────────────────────────────────────────────────
def _estimate_mv(ruolo: str, titolarita_pct: float, presenze: int,
                  gol_pg: float, assist_pg: float, xg_pg: float) -> float:
    """
    Stima la Media Voto in base ai dati di campo.
    Range realistico: 5.0 (riserva) → 7.5 (top player stagionale).
    """
    # Base: dipende dal tempo giocato
    if presenze >= 30:
        mv = 6.15
    elif presenze >= 20:
        mv = 6.00
    elif presenze >= 10:
        mv = 5.80
    else:
        mv = 5.55

    # Titolarità: chi parte sempre ha voto più alto
    mv += titolarita_pct * 0.55

    # Qualità offensiva (xG come proxy del livello del giocatore)
    mv += min(xg_pg * 1.5, 0.70)

    # Assist e gol portano voti alti nelle partite giocate
    mv += min(assist_pg * 0.8, 0.40)
    mv += min(gol_pg * 0.5, 0.50)

    # Portieri: correzione (MV media portieri ~6.0-6.3, meno varianza)
    if ruolo == 'P':
        mv = 5.8 + titolarita_pct * 0.6

    return round(min(mv, 7.80), 2)


# ── Stima FantaMedia ──────────────────────────────────────────────────────────
def _estimate_fm(ruolo: str, mv: float, gol_pg: float, assist_pg: float,
                  ammonizioni_pg: float, titolarita_pct: float) -> float:
    """
    FM = MV + gol/g × bonus_gol + assist/g × 1.0 - ammonizioni/g × 0.5
    Per portieri: FM ≈ MV + bonus_parate stimato
    """
    if ruolo == 'P':
        # Portieri: +0.5 medio per clean sheet (stimato), no gol
        fm = mv + titolarita_pct * 0.4
    else:
        gol_b = GOL_BONUS.get(ruolo, 3.0)
        fm = mv + gol_pg * gol_b + assist_pg * 1.0 - ammonizioni_pg * 0.5

    return round(max(4.0, min(fm, 10.0)), 2)


# ── Stima costo asta ─────────────────────────────────────────────────────────
def _estimate_cost(ruolo: str, fm: float, presenze: int) -> int:
    """
    Mappa la FM stimata al costo asta realistico per ruolo.
    Scala non lineare: i top player costano esponenzialmente di più.
    """
    fm_floor = {'P': 5.0, 'D': 5.5, 'C': 5.5, 'A': 5.5}.get(ruolo, 5.0)
    fm_ceil  = {'P': 7.5, 'D': 8.5, 'C': 9.5, 'A': 10.5}.get(ruolo, 9.0)

    normalized = max(0.0, min(1.0, (fm - fm_floor) / (fm_ceil - fm_floor)))
    costo_max = COSTO_MAX.get(ruolo, 30)

    # Scala potenza 2: i top player costano molto di più
    cost = COSTO_MIN + (costo_max - COSTO_MIN) * (normalized ** 2.0)

    # Riduzione per poche presenze (giocatori a rischio infortunio)
    if presenze < 10:
        cost *= 0.55
    elif presenze < 18:
        cost *= 0.80

    return max(COSTO_MIN, round(cost))


# ── Funzione principale ───────────────────────────────────────────────────────
def generate_complete_dataset(
    season: str = "current",
    min_presenze: int = 5,
) -> pd.DataFrame:
    """
    Genera un DataFrame con tutti i giocatori Serie A della stagione indicata.

    Parametri:
        season:       "current" oppure "YYYY-YYYY" (es. "2022-2023")
        min_presenze: soglia minima di partite giocate

    Ritorna un DataFrame con ~400-500 giocatori pronto per
    ottimizzatore e backtest.
    """
    # 1. Carica (o scarica) le stats FBref
    stats = load_stats_for_season(season)
    if stats is None or stats.empty:
        logger.info(f"Cache {season} assente, scaricamento FBref...")
        ok = fetch_and_save_stats(season)
        if not ok:
            raise RuntimeError(
                f"Impossibile scaricare le stats FBref per la stagione '{season}'.\n"
                "Controlla la connessione internet e riprova."
            )
        stats = load_stats_for_season(season)

    if stats is None or stats.empty:
        raise RuntimeError("Dati FBref vuoti dopo il download.")

    df = stats.copy()

    # 2. Filtro presenze minime
    if 'presenze' in df.columns:
        df['presenze'] = pd.to_numeric(df['presenze'], errors='coerce').fillna(0)
        df = df[df['presenze'] >= min_presenze].copy()

    if df.empty:
        raise RuntimeError(f"Nessun giocatore con ≥{min_presenze} presenze nel dataset.")

    # 3. Mappa ruoli
    role_col = 'ruolo_fbref' if 'ruolo_fbref' in df.columns else None
    if role_col:
        df['ruolo'] = df[role_col].apply(_map_role)
    else:
        df['ruolo'] = 'C'

    # 4. Rinomina colonna nome
    df = df.rename(columns={'nome_fbref': 'nome'})

    # 5. Riempi valori mancanti
    _defaults: dict[str, dict] = {
        'gol_pg':         {'P': 0.00, 'D': 0.04, 'C': 0.12, 'A': 0.35},
        'assist_pg':      {'P': 0.00, 'D': 0.06, 'C': 0.15, 'A': 0.15},
        'ammonizioni_pg': {'P': 0.05, 'D': 0.18, 'C': 0.15, 'A': 0.10},
        'xg_pg':          {'P': 0.00, 'D': 0.03, 'C': 0.10, 'A': 0.30},
        'titolarita_pct': {'P': 0.70, 'D': 0.70, 'C': 0.65, 'A': 0.65},
    }
    for col, defaults in _defaults.items():
        if col not in df.columns:
            df[col] = df['ruolo'].map(defaults).fillna(0.1)
        else:
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = df.loc[mask, 'ruolo'].map(defaults).fillna(0.1)

    # 6. Calcola MV stimata, FM stimata, costo stimato
    df['mv_stimata'] = df.apply(
        lambda r: _estimate_mv(
            r['ruolo'], r['titolarita_pct'], int(r['presenze']),
            r['gol_pg'], r['assist_pg'], r['xg_pg'],
        ),
        axis=1,
    )

    df['fanta_media'] = df.apply(
        lambda r: _estimate_fm(
            r['ruolo'], r['mv_stimata'],
            r['gol_pg'], r['assist_pg'], r['ammonizioni_pg'], r['titolarita_pct'],
        ),
        axis=1,
    )

    df['costo_iniziale'] = df.apply(
        lambda r: _estimate_cost(r['ruolo'], r['fanta_media'], int(r['presenze'])),
        axis=1,
    )

    # 7. Selezione e ordinamento colonne finali
    final_cols = [
        'nome', 'ruolo', 'squadra', 'costo_iniziale', 'fanta_media', 'mv_stimata',
        'gol_pg', 'assist_pg', 'ammonizioni_pg', 'xg_pg', 'titolarita_pct', 'presenze',
    ]
    final_cols = [c for c in final_cols if c in df.columns]

    df = (
        df[final_cols]
        .drop_duplicates(subset=['nome'])
        .sort_values(by=['ruolo', 'fanta_media'], ascending=[True, False])
        .reset_index(drop=True)
    )

    return df


def generate_and_save(
    season: str = "current",
    output_name: str | None = None,
    min_presenze: int = 5,
) -> tuple[bool, str, pd.DataFrame | None]:
    """
    Genera e salva il dataset completo.
    Ritorna (successo, messaggio, dataframe).
    """
    try:
        df = generate_complete_dataset(season, min_presenze)

        if output_name is None:
            season_str = season.replace('-', '_')
            output_name = f"dataset_completo_{season_str}.csv"

        output_path = os.path.join(DATA_DIR, output_name)
        df.to_csv(output_path, index=False)

        counts = df['ruolo'].value_counts().to_dict()
        msg = (
            f"✅ {len(df)} giocatori generati e salvati in '{output_name}'\n"
            f"   P:{counts.get('P',0)}  D:{counts.get('D',0)}  "
            f"C:{counts.get('C',0)}  A:{counts.get('A',0)}"
        )
        return True, msg, df

    except Exception as e:
        return False, str(e), None


# ── Statistiche di distribuzione ──────────────────────────────────────────────
def dataset_summary(df: pd.DataFrame) -> str:
    lines = [f"Totale giocatori: {len(df)}"]
    for ruolo in ['P', 'D', 'C', 'A']:
        sub = df[df['ruolo'] == ruolo]
        if sub.empty:
            continue
        lines.append(
            f"  {ruolo}: {len(sub)} giocatori | "
            f"FM {sub['fanta_media'].min():.1f}–{sub['fanta_media'].max():.1f} "
            f"(media {sub['fanta_media'].mean():.2f}) | "
            f"Costo {sub['costo_iniziale'].min()}–{sub['costo_iniziale'].max()}"
        )
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    season = sys.argv[1] if len(sys.argv) > 1 else "current"
    print(f"\nGenerazione dataset Serie A — stagione: {season}")

    ok, msg, df = generate_and_save(season, min_presenze=5)
    print(msg)
    if df is not None:
        print("\nDistribuzione:")
        print(dataset_summary(df))
        print("\nTop 5 per FM stimata:")
        print(df.nlargest(5, 'fanta_media')[
            ['nome', 'ruolo', 'fanta_media', 'costo_iniziale', 'gol_pg', 'assist_pg']
        ].to_string(index=False))
