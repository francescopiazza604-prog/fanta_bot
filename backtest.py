"""
backtest.py — Engine di Backtest Causal-Temporale per FantaBot.

MODALITÀ BACKTEST (Zero Data Leakage / Out-of-Sample):
1. AI SQUAD (Previsione al Buio):
   - Riceve SOLO le informazioni pre-stagione (stats anno N-1, trasferimenti, quotazioni iniziali).
   - Genera le previsioni con il modello temporale/ML.
   - Sceglie la rosa (25 giocatori) e l'11 titolare usando lo STESSO ottimizzatore (es. Listone Master o MILP).
2. ORACLE SQUAD (Massimo Teorico / Senno di Poi):
   - Sceglie la rosa perfetta che avrebbe massimizzato i punti usando la FantaMedia REALE registrata a fine stagione.
3. CONFRONTO REALE:
   - Confronta la resa REALE della squadra scelta dall'IA con la squadra Oracle.
   - Calcola l'Efficienza dell'IA (%) = (Punti Reali Squadra AI / Punti Squadra Oracle) * 100.
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_TARGET = os.path.join(DATA_DIR, "serie_a_23_24_backtest.csv")


def _get_best_starter_score(team: pd.DataFrame, score_col: str) -> tuple[float, pd.DataFrame]:
    """
    Calcola l'11 titolare ottimale (11 giocatori: 1P, nD, nC, nA) sotto il miglior modulo
    (3-4-3, 4-3-3, 3-5-2, 4-4-2, 4-5-1, 5-3-2) che massimizza la somma di score_col.
    """
    if 'is_starter' in team.columns and team['is_starter'].sum() == 11:
        starters = team[team['is_starter']].copy()
        return round(starters[score_col].sum(), 2), starters

    formations = {
        '3-4-3': {'P': 1, 'D': 3, 'C': 4, 'A': 3},
        '4-3-3': {'P': 1, 'D': 4, 'C': 3, 'A': 3},
        '3-5-2': {'P': 1, 'D': 3, 'C': 5, 'A': 2},
        '4-4-2': {'P': 1, 'D': 4, 'C': 4, 'A': 2},
        '4-5-1': {'P': 1, 'D': 4, 'C': 5, 'A': 1},
    }

    best_score = -1.0
    best_starters = None

    for form_name, req in formations.items():
        curr_starters = []
        possible = True
        for r, n in req.items():
            sub = team[team['ruolo'] == r].nlargest(n, score_col)
            if len(sub) < n:
                possible = False
                break
            curr_starters.append(sub)

        if possible:
            starters_df = pd.concat(curr_starters)
            score = starters_df[score_col].sum()
            if score > best_score:
                best_score = score
                best_starters = starters_df

    if best_starters is None:
        # Fallback a 3-4-3
        starters_df = pd.concat([
            team[team['ruolo'] == 'P'].nlargest(1, score_col),
            team[team['ruolo'] == 'D'].nlargest(3, score_col),
            team[team['ruolo'] == 'C'].nlargest(4, score_col),
            team[team['ruolo'] == 'A'].nlargest(3, score_col),
        ])
        best_score = starters_df[score_col].sum()
        best_starters = starters_df

    return round(best_score, 2), best_starters


def _compute_accuracy_metrics(df_matched: pd.DataFrame) -> dict:
    """Calcola le metriche di accuratezza statistica sulle predizioni dell'IA."""
    if df_matched.empty:
        return {'rmse': 0.0, 'mae': 0.0, 'corr': 0.0}

    pred = df_matched['previsione_ia']
    actual = df_matched['fanta_media_y'] if 'fanta_media_y' in df_matched.columns else df_matched['fanta_media']

    rmse = float(np.sqrt(((pred - actual) ** 2).mean())) if len(df_matched) > 0 else 0.0
    mae = float((pred - actual).abs().mean()) if len(df_matched) > 0 else 0.0
    corr = float(pred.corr(actual)) if len(df_matched) > 2 else 0.0

    accuracies = {}
    for n in [5, 10, 15]:
        if len(df_matched) >= n:
            top_actual = set(df_matched.nlargest(n, actual.name)['nome'])
            top_pred = set(df_matched.nlargest(n, 'previsione_ia')['nome'])
            accuracies[f'top{n}_accuracy'] = round(len(top_actual & top_pred) / n, 2)

    return {
        'rmse': round(rmse, 3),
        'mae': round(mae, 3),
        'corr': round(corr, 3) if not np.isnan(corr) else 0.0,
        **accuracies
    }


def run_ia_backtest(
    budget: int = 500,
    strategy: str = 'listone',
    target_df: pd.DataFrame | None = None,
    stats_df: pd.DataFrame | None = None,
    target_path: str = DEFAULT_TARGET,
) -> dict:
    """
    Esegue il backtest temporale tra la scelta dell'IA (pre-stagione) e la realtà (fine stagione).

    Ritorna un dizionario con:
      - oracle_team: rosa ottimizzata col senno di poi (FM reale)
      - ai_team: rosa scelta dall'IA prima dell'inizio stagione
      - metrics: MAE, RMSE, Correlazione, Top-N accuracy
      - df_comparison: tabella comparativa per ogni giocatore
      - oracle_score: punti/FM 11 titolari squadra Oracle
      - ai_score_real: punti/FM REALE fatti dall'11 titolare della squadra AI
      - ai_score_pred: punti/FM PREVISTI dall'IA
      - efficiency: % di avvicinamento dell'IA al massimo teorico (Oracle)
    """
    # 1. Carica Ground Truth (Risultati Reali della stagione conclusa)
    if target_df is None:
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"File backtest non trovato: {target_path}")
        target_df = pd.read_csv(target_path)

    target_df = target_df.copy().reset_index(drop=True)

    # Verifica presenza FantaMedia reale
    fm_col = None
    for candidate in ('fanta_media', 'fm_reale', 'fantamedia', 'fm'):
        if candidate in target_df.columns and pd.to_numeric(target_df[candidate], errors='coerce').dropna().count() > 5:
            fm_col = candidate
            break

    if fm_col:
        target_df['fanta_media'] = pd.to_numeric(target_df[fm_col].astype(str).str.replace(',', '.'), errors='coerce').fillna(6.0)
        has_real_fm = True
    else:
        target_df['fanta_media'] = 6.0
        has_real_fm = False

    if 'costo_iniziale' not in target_df.columns:
        target_df['costo_iniziale'] = 1

    # 2. Genera Previsioni dell'IA (al buio, senza spiare la fanta_media reale)
    if stats_df is not None:
        ai_input = stats_df.copy()
        if 'fanta_media' not in ai_input.columns:
            ai_input['fanta_media'] = 6.0
        if 'costo_iniziale' not in ai_input.columns:
            ai_input['costo_iniziale'] = 1
    else:
        # Fallback se non c'è file stagione precedente: usa il dataset senza la colonna fanta_media reale
        ai_input = target_df.copy()

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, dir=DATA_DIR) as f:
        ai_input.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        from predict import train_prediction_model
        from optimizer import optimize_team

        # Addestra il modello AI al buio
        df_pred = train_prediction_model(temp_path, strategy=strategy)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    # 3. Costruisci SQUADRA AI (Scelta dall'IA al buio)
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, dir=DATA_DIR) as f:
        df_pred.to_csv(f.name, index=False)
        pred_temp_path = f.name

    try:
        ai_team = optimize_team(pred_temp_path, budget=budget, strategy=strategy)
    finally:
        if os.path.exists(pred_temp_path):
            os.unlink(pred_temp_path)

    # Merge con la FantaMedia reale realizzata a fine stagione
    if 'fanta_media' in ai_team.columns:
        ai_team = ai_team.rename(columns={'fanta_media': 'fanta_media_storica'})
        
    ai_team = ai_team.merge(
        target_df[['nome', 'fanta_media']],
        on='nome',
        how='left',
    )
    ai_team['fanta_media'] = ai_team['fanta_media'].fillna(5.50)

    # Calcola punteggio previsto vs REALE realizzato dall'11 titolare AI
    ai_score_pred, ai_starters_pred = _get_best_starter_score(ai_team, 'previsione_ia')
    ai_score_real, ai_starters_real = _get_best_starter_score(ai_team, 'fanta_media')

    # 4. Costruisci SQUADRA ORACLE (Senno di Poi — Massimizza FM REALE col solver sui COSTI REALI)
    oracle_input = df_pred.copy()
    
    # Sovrascrivi fanta_media con i voti REALI realizzati a fine stagione
    if 'fanta_media' in oracle_input.columns:
        oracle_input.drop(columns=['fanta_media'], inplace=True)
        
    oracle_input = oracle_input.merge(
        target_df[['nome', 'fanta_media']],
        on='nome',
        how='inner',
    )
    oracle_input['fanta_media'] = pd.to_numeric(oracle_input['fanta_media'], errors='coerce').fillna(5.50)
    oracle_input['score_selezione'] = oracle_input['fanta_media']
    oracle_input['previsione_ia'] = oracle_input['fanta_media']

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, dir=DATA_DIR) as f:
        oracle_input.to_csv(f.name, index=False)
        oracle_temp_path = f.name

    try:
        oracle_team = optimize_team(oracle_temp_path, budget=budget, strategy=strategy)
    finally:
        if os.path.exists(oracle_temp_path):
            os.unlink(oracle_temp_path)

    oracle_score, _ = _get_best_starter_score(oracle_team, 'fanta_media')

    # 5. Metriche di Accuratezza e Valutazione Giocatori
    df_comparison = oracle_input.copy()
    metrics = _compute_accuracy_metrics(df_comparison)

    # Status prestazioni giocatori scelti dall'IA
    ai_team['delta_fm'] = ai_team['fanta_media'] - ai_team['previsione_ia']
    ai_team['status'] = np.where(
        ai_team['delta_fm'] > 0.30, '🔥 SOPRA LE ATTESE',
        np.where(ai_team['delta_fm'] < -0.30, '❄️ SOTTO LE ATTESE', '✅ IN LINEA')
    )

    efficiency = round((ai_score_real / oracle_score * 100), 1) if oracle_score > 0 else 0.0

    return {
        'oracle_team': oracle_team,
        'ai_team': ai_team,
        'metrics': metrics,
        'df_comparison': df_comparison,
        'oracle_score': oracle_score,
        'ai_score_real': ai_score_real,
        'ai_score_pred': ai_score_pred,
        'efficiency': efficiency,
        'has_real_fm': has_real_fm,
    }


if __name__ == "__main__":
    print("=== TEST BACKTEST ENGINE ===")
    res = run_ia_backtest(budget=500, strategy='listone')
    print(f"Efficienza IA: {res['efficiency']}%")
    print(f"FM Oracle (Senno di poi): {res['oracle_score']} pti")
    print(f"FM AI Real (Realizzata):   {res['ai_score_real']} pti")
    print(f"FM AI Prev (Prevista):     {res['ai_score_pred']} pti")
    print("\nCampione 5 Giocatori AI:")
    print(res['ai_team'][['nome', 'ruolo', 'costo_iniziale', 'previsione_ia', 'fanta_media', 'status']].head(10))
