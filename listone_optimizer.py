"""
listone_optimizer.py — Ottimizzatore Fantacalcio per Campionato a Listone.

STRATEGIA DA VITTORIA A LISTONE:
1. Hyper-Concentrazione sull'11 Titolare:
   In un campionato a listone (prezzi fissi, nessun'asta) l'85-92% del budget
   viene allocato sugli 11 titolari (Super-Top Player).
2. 14 Riserve Economiche a Titolarità Garantita (Zero-Risk Bench):
   Il restante 8-15% del budget compra riserve da 1-4 crediti con titolarità >= 50-60%
   per garantire che la squadra non buchi mai la formazione.
3. Formulazione Double-Variable MILP:
   x_i in {0,1} -> Giocatore i nella rosa da 25 (3P + 8D + 8C + 6A)
   y_i in {0,1} -> Giocatore i nell'11 TITOLARE (y_i <= x_i)
4. Moduli Tattici Dinamici & Modificatore Difesa:
   Testa automaticamente i moduli (3-4-3, 4-3-3, 3-5-2, 4-4-2, 4-5-1, 5-3-2)
   calcolando l'impatto del Modificatore Difesa (+1, +3, +6 pti) se attivo.
5. Obiettivo: Massimizzare i Punti Totali Attesi sulle 38 giornate:
   E[Punti_38g] = 38 * titolarita_pct * FM_prevista + Bonus_Modificatore
"""

import os
import logging
import numpy as np
import pandas as pd
from pulp import (
    LpMaximize, LpProblem, LpVariable, lpSum,
    PULP_CBC_CMD, LpStatus,
)

logger = logging.getLogger(__name__)

# Composizione rosa standard 25 giocatori
ROSA_TOTALE = {'P': 3, 'D': 8, 'C': 8, 'A': 6}

# Formazioni tattiche valide per l'11 Titolare (P=1 obbligatorio)
FORMATIONS: dict[str, dict[str, int]] = {
    '3-4-3': {'P': 1, 'D': 3, 'C': 4, 'A': 3},
    '4-3-3': {'P': 1, 'D': 4, 'C': 3, 'A': 3},
    '3-5-2': {'P': 1, 'D': 3, 'C': 5, 'A': 2},
    '4-4-2': {'P': 1, 'D': 4, 'C': 4, 'A': 2},
    '4-5-1': {'P': 1, 'D': 4, 'C': 5, 'A': 1},
    '5-3-2': {'P': 1, 'D': 5, 'C': 3, 'A': 2},
}

def _calculate_expected_38g_points(df: pd.DataFrame, use_defense_mod: bool = True) -> pd.Series:
    """
    Calcola i punti totali attesi sulle 38 giornate per ciascun giocatore:
    E[Punti] = 38 * (titolarita_pct) * FM_prevista
    Se il modificatore difesa è attivo, aggiunge un boost ai difensori con media voto alta.
    """
    fm = df['previsione_ia'] if 'previsione_ia' in df.columns else df.get('fanta_media', df.get('costo_iniziale', 0))
    
    # Titolarità: se 0 presenze (pre-stagione inizio campionato), stima la titolarità dal costo di listone
    if 'titolarita_pct' in df.columns and df['titolarita_pct'].max() > 0.05:
        tit = df['titolarita_pct'].copy()
    else:
        costo = df['costo_iniziale'].astype(float)
        c_max = max(costo.max(), 1.0)
        tit = 0.35 + 0.60 * (costo / c_max)

    tit_clean = np.clip(tit.fillna(0.75), 0.20, 0.98)
    fm_clean = np.clip(fm.fillna(5.50), 3.0, 10.0)
    
    # Punti base attesi su 38 giornate
    pts_38 = 38.0 * tit_clean * fm_clean
    
    # Boost Modificatore Difesa: i difensori con media voto >= 6.15 o costo alto apportano punti extra attesi
    if use_defense_mod:
        if 'media_voto' in df.columns and df['media_voto'].std() > 0.1:
            mv = df['media_voto'].fillna(6.0)
            def_mod_boost = np.where(
                (df['ruolo'] == 'D') & (mv >= 6.15),
                (mv - 6.0) * 15.0,
                0.0
            )
        else:
            # Fallback pre-stagione basato su costo difensori
            def_mod_boost = np.where(
                (df['ruolo'] == 'D') & (df['costo_iniziale'] >= 15),
                (df['costo_iniziale'] - 12) * 1.5,
                0.0
            )
        pts_38 += def_mod_boost

    # Boost Super-Star Premium: premia i top player assoluti (FM >= 7.15 o costo >= 24)
    # per forzare l'acquisto dei trascinatori d'attacco e mediana nei campionati a listone
    star_boost = np.where(
        (fm_clean >= 7.15) | (df['costo_iniziale'] >= 24),
        (fm_clean - 6.50) * 10.0,
        0.0
    )
    pts_38 += star_boost

    return pd.Series(pts_38, index=df.index)


def optimize_listone_single_formation(
    df: pd.DataFrame,
    budget: int = 500,
    formation: str = '3-4-3',
    use_defense_modifier: bool = True,
    starter_budget_pct: float = 0.85,
    relax_level: int = 0,
) -> tuple[pd.DataFrame | None, float]:
    """
    Risolve il problema MILP per un singolo modulo tattico a Listone.
    """
    if formation not in FORMATIONS:
        raise ValueError(f"Modulo non valido: {formation}. Moduli disponibili: {list(FORMATIONS.keys())}")
    
    req_starters = FORMATIONS[formation]
    
    df = df.copy().reset_index(drop=True)
    df['pts_38g'] = _calculate_expected_38g_points(df, use_defense_modifier)
    
    costo = df['costo_iniziale'].astype(float).values
    pts_38g = df['pts_38g'].values
    titolarita = df.get('titolarita_pct', pd.Series(0.75, index=df.index)).fillna(0.75).values
    ruoli = df['ruolo'].values
    n_players = len(df)
    
    prob = LpProblem(f"Listone_Master_{formation}_r{relax_level}", LpMaximize)
    
    # x[i] = 1 se il giocatore i è nei 25 della rosa
    # y[i] = 1 se il giocatore i è nei 11 TITOLARI
    x = LpVariable.dicts("x", range(n_players), cat="Binary")
    y = LpVariable.dicts("y", range(n_players), cat="Binary")
    
    # ── Obiettivo: Massimizzare i punti degli 11 Titolari + Valore ed Utilità delle Riserve ──
    # Le riserve (x[i] - y[i]) contribuiscono in base ai loro punti attesi * titolarità * fattore di rotazione (0.35)
    bench_weight = 0.35
    prob += lpSum([
        y[i] * pts_38g[i] + (x[i] - y[i]) * (pts_38g[i] * bench_weight * titolarita[i])
        for i in range(n_players)
    ]), "Total_Expected_Season_Points"
    
    # ── Vincolo 1: y[i] <= x[i] (Puoi schierare titolare solo chi è in rosa) ──────
    for i in range(n_players):
        prob += y[i] <= x[i], f"Starter_In_Squad_{i}"
        
    # ── Vincolo 2: Composizione esatta rosa 25 giocatori ─────────────────────────
    for r, count in ROSA_TOTALE.items():
        prob += lpSum([x[i] for i in range(n_players) if ruoli[i] == r]) == count, f"Squad_Count_{r}"
        
    # ── Vincolo 3: Composizione esatta 11 Titolari per il modulo scelto ──────────
    for r, count in req_starters.items():
        prob += lpSum([y[i] for i in range(n_players) if ruoli[i] == r]) == count, f"Starter_Count_{r}"
        
    # ── Vincolo 4: Budget Totale <= budget ───────────────────────────────────────
    prob += lpSum([x[i] * costo[i] for i in range(n_players)]) <= budget, "Total_Budget_Limit"
    
    # ── Vincoli morbidi (dipendenti da relax_level) ──────────────────────────────
    if relax_level == 0:
        # Alloca fino all'80% per i titolari, lasciando ~20% per riserve di qualità
        min_starter_spend = starter_budget_pct * budget * 0.75
        max_starter_spend = budget * 0.88
        prob += lpSum([y[i] * costo[i] for i in range(n_players)]) >= min_starter_spend, "Min_Starter_Budget"
        prob += lpSum([y[i] * costo[i] for i in range(n_players)]) <= max_starter_spend, "Max_Starter_Budget"
        
        # Garanzia titolarità minima per i titolari
        for i in range(n_players):
            pres = df.loc[i, 'presenze'] if 'presenze' in df.columns else 20
            if titolarita[i] < 0.40 or pres < 10:
                prob += y[i] == 0, f"No_Unreliable_Starter_{i}"
                    
        # Copertura panchina con riserve utili e titolari (almeno 50% titolarità)
        high_tit_indices = [i for i in range(n_players) if titolarita[i] >= 0.50]
        if len(high_tit_indices) >= 15:
            prob += lpSum([(x[i] - y[i]) for i in high_tit_indices]) >= 7, "Min_Bench_Coverage"

        # Riserve di qualità con impatto in caso di infortunio del titolare:
        # Almeno 2 Difensori di riserva con costo >= 3 e titolarità >= 50%
        bench_d_quality = [i for i in range(n_players) if ruoli[i] == 'D' and costo[i] >= 3 and titolarita[i] >= 0.50]
        if len(bench_d_quality) >= 2:
            prob += lpSum([(x[i] - y[i]) for i in bench_d_quality]) >= 2, "Bench_D_Quality"

        # Almeno 2 Centrocampisti di riserva con costo >= 4 e titolarità >= 50%
        bench_c_quality = [i for i in range(n_players) if ruoli[i] == 'C' and costo[i] >= 4 and titolarita[i] >= 0.50]
        if len(bench_c_quality) >= 2:
            prob += lpSum([(x[i] - y[i]) for i in bench_c_quality]) >= 2, "Bench_C_Quality"

        # Almeno 1 Attaccante di riserva utile con costo >= 5 e titolarità >= 50%
        bench_a_quality = [i for i in range(n_players) if ruoli[i] == 'A' and costo[i] >= 5 and titolarita[i] >= 0.50]
        if len(bench_a_quality) >= 1:
            prob += lpSum([(x[i] - y[i]) for i in bench_a_quality]) >= 1, "Bench_A_Quality"

    elif relax_level == 1:
        min_starter_spend = starter_budget_pct * budget * 0.70
        prob += lpSum([y[i] * costo[i] for i in range(n_players)]) >= min_starter_spend, "Min_Starter_Budget"
    
    # level 2+: solo vincoli duri (composizione + budget totale)
    
    # Risoluzione SILENZIOSA
    solver = PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    
    if LpStatus[status] != 'Optimal':
        return None, 0.0

    # Estrazione risultati
    df_res = df.copy()
    df_res['in_squadra'] = [bool(x[i].varValue and x[i].varValue > 0.5) for i in range(n_players)]
    df_res['is_starter'] = [bool(y[i].varValue and y[i].varValue > 0.5) for i in range(n_players)]
    
    selected = df_res[df_res['in_squadra']].copy()
    selected['ruolo_titolare'] = selected['is_starter'].map({True: 'TITOLARE', False: 'PANCHINA'})
    
    # Calcolo totale punti attesi titolari
    starters_pts = selected[selected['is_starter']]['pts_38g'].sum()
    
    # Ordina: Prima i Titolari per ruolo (P, D, C, A), poi la Panchina per ruolo
    role_order = {'P': 1, 'D': 2, 'C': 3, 'A': 4}
    selected['role_rank'] = selected['ruolo'].map(role_order)
    selected = selected.sort_values(by=['is_starter', 'role_rank', 'costo_iniziale'], ascending=[False, True, False]).reset_index(drop=True)
    selected.drop(columns=['role_rank'], inplace=True)
    
    return selected, starters_pts


def optimize_listone_auto(
    data_path_or_df: str | pd.DataFrame,
    budget: int = 500,
    use_defense_modifier: bool = True,
    preferred_formation: str = 'AUTO',
) -> tuple[pd.DataFrame, str, float]:
    """
    Ottimizzatore Master per il Campionato a Listone.
    
    Se `preferred_formation` == 'AUTO', valuta TUTTI i moduli ufficiali (3-4-3, 4-3-3, 3-5-2, 4-4-2, 4-5-1, 5-3-2)
    e seleziona la combinazione (Modulo + Rosa) che garantisce il MAGGIORE PUNTEGGIO TOTALE STAGIONALE (38g).
    
    Ritorna: (DataFrame_Rosa_25, Modulo_Migliore, Punti_Stagionali_Attesi_11_Titolari)
    """
    if isinstance(data_path_or_df, str):
        if not os.path.exists(data_path_or_df):
            raise FileNotFoundError(f"File quotazioni non trovato: {data_path_or_df}")
        df = pd.read_csv(data_path_or_df)
    else:
        df = data_path_or_df.copy()
    
    # Pulizia basilare
    df = df[df['nome'].astype(str).str.strip() != ''].copy()
    df['ruolo'] = df['ruolo'].astype(str).str.upper().str.strip()
    df = df[df['ruolo'].isin({'P', 'D', 'C', 'A'})].copy().reset_index(drop=True)

    # Verifica colonne chiave
    if 'previsione_ia' not in df.columns and 'fanta_media' in df.columns:
        df['previsione_ia'] = df['fanta_media']
    elif 'previsione_ia' not in df.columns:
        df['previsione_ia'] = 6.0
        
    if 'titolarita_pct' not in df.columns:
        df['titolarita_pct'] = 0.75

    if preferred_formation != 'AUTO' and preferred_formation in FORMATIONS:
        candidates = [preferred_formation]
    else:
        candidates = list(FORMATIONS.keys())
        
    best_df = None
    best_formation = None
    best_score = -1.0
    
    for form in candidates:
        for relax in range(3):
            res_df, score = optimize_listone_single_formation(
                df,
                budget=budget,
                formation=form,
                use_defense_modifier=use_defense_modifier,
                starter_budget_pct=0.85,
                relax_level=relax,
            )
            if res_df is not None and score > best_score:
                best_score = score
                best_formation = form
                best_df = res_df
                break  # Trovata soluzione ottimale per questo modulo
            
    if best_df is None:
        raise ValueError(
            "Impossibile ottimizzare la rosa Listone con i dati correnti. "
            "Verifica che il dataset contenga abbastanza giocatori per ciascun ruolo."
        )
        
    logger.info(f"Miglior modulo per Listone: {best_formation} con {best_score:.1f} punti attesi.")
    return best_df, best_formation, best_score
