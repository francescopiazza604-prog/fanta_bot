"""
chaos_optimizer.py — Fantacalcio: Chaos & Upside Index Optimizer

Budget : variabile (default 500cr, standard Fantacalcio)
Rosa   : 3P + 8D + 8C + 6A = 25 giocatori
Modulo : fisso (il solver sceglie chi, non la tattica)

RIMOSSO: fattore 'Ownership' — inaffidabile prima dell'inizio del torneo.

PIPELINE:
  1. generate_mock_players(budget) → dataset sintetico con costi scalati al budget
  2. compute_chaos_upside_index()  → colonne 'cui' [0,1] e 'cui_label'
  3. build_chaos_score()           → 'score_selezione' = FM × (1 + w × CUI)
  4. optimize_chaos_team(budget)   → MILP con 4 livelli di vincoli:
       L1 — Composizione esatta (3P+8D+8C+6A)
       L2 — Budget totale ≤ budget_cr
       L3 — SOLID BASE: ≥50% budget su giocatori CUI ≤ mediana (affidabili)
       L4 — GAME CHANGERS: ≥2 MID + ≥2 ATT con CUI nel top 15% del DB

BILANCIAMENTO "Noioso ma sicuro" vs "Talento Imprevedibile":
  Il solver massimizza score_selezione = FM × (1 + 0.55 × CUI).
  Un giocatore FM=6.5 con CUI=1.0 vale score=10.08, uno con CUI=0.0 vale 6.50.
  I vincoli L3+L4 lo costringono a non abbandonare né la solidità né l'upside.
"""

import logging
import os
import numpy as np
import pandas as pd
from pulp import (
    LpMaximize, LpProblem, LpVariable, lpSum,
    PULP_CBC_CMD, LpStatus,
)

logger = logging.getLogger(__name__)

ROSA           = {'P': 3, 'D': 8, 'C': 8, 'A': 6}
BUDGET_DEFAULT = 500   # budget standard Fantacalcio; passa un valore diverso via CLI o API
_BUDGET_REF    = 500   # budget di riferimento per scalare i costi del mock data


# ── 1. GENERAZIONE DATI MOCK ──────────────────────────────────────────────────

def generate_mock_players(
    n_per_role: int = 80,
    seed: int = 42,
    budget: int = BUDGET_DEFAULT,
) -> pd.DataFrame:
    """
    Genera un dataset sintetico con metriche avanzate.

    I costi vengono scalati proporzionalmente al budget passato:
      scala = budget / _BUDGET_REF  (riferimento 500cr)
    Così a budget=250 i costi si dimezzano, a budget=1000 si raddoppiano.

    Colonne prodotte (OWNERSHIP RIMOSSA):
      nome, ruolo, costo_iniziale, fanta_media,
      storico_voti_std   → deviazione standard dei voti nelle ultime 15 gare
      peak_rate          → % partite con voto base ≥ 7.0
      dribbling_riusciti_p90, passaggi_chiave_p90
      gol_pg, assist_pg  → rendimento reale per 90 minuti
      xg_pg, xa_pg       → atteso statistico (per XG Delta)

    Il 25% dei giocatori è generato come profilo "caotico" (alta std, picchi
    frequenti): sono i candidati naturali al ruolo di Game Changer.
    """
    rng   = np.random.default_rng(seed)
    scala = budget / _BUDGET_REF   # fattore di scala rispetto al budget di riferimento

    # Costi di riferimento a budget=500; vengono moltiplicati per `scala`.
    # Le FM e le metriche P90 sono indipendenti dal budget.
    role_params = {
        'P': dict(cost=(1,  36),  fm=(4.5, 6.5), xg=(0.00, 0.00), xa=(0.00, 0.00),
                  drib=(0.0, 0.3),  pkey=(0.0, 0.5)),
        'D': dict(cost=(1,  60),  fm=(5.0, 7.2), xg=(0.01, 0.08), xa=(0.01, 0.12),
                  drib=(0.1, 1.8),  pkey=(0.1, 1.5)),
        'C': dict(cost=(2, 100),  fm=(5.5, 8.0), xg=(0.05, 0.22), xa=(0.05, 0.28),
                  drib=(0.3, 4.0),  pkey=(0.4, 4.0)),
        'A': dict(cost=(3, 160),  fm=(5.5, 9.5), xg=(0.10, 0.60), xa=(0.03, 0.28),
                  drib=(0.5, 5.5),  pkey=(0.2, 3.0)),
    }

    records = []
    idx = 0
    for ruolo, p in role_params.items():
        count = n_per_role if ruolo != 'P' else max(n_per_role // 3, 15)
        for _ in range(count):
            costo   = max(1, int(round(rng.integers(*p['cost']) * scala)))
            base_fm = float(rng.uniform(*p['fm']))
            xg_base = float(rng.uniform(*p['xg']))
            xa_base = float(rng.uniform(*p['xa']))
            drib    = float(rng.uniform(*p['drib']))
            pkey    = float(rng.uniform(*p['pkey']))

            # ── Profilo caotico vs regolare ───────────────────────────────────
            is_chaotic = rng.random() < 0.25
            if is_chaotic:
                std_voti = float(rng.uniform(0.8, 2.2))
            else:
                std_voti = float(rng.uniform(0.1, 0.55))

            voti_storici = np.clip(
                base_fm + rng.normal(0, std_voti, 15), 1.0, 10.0
            )
            storico_std = float(np.std(voti_storici))
            peak_rate   = float((voti_storici >= 7.0).mean())

            # ── XG Delta: alcuni giocatori "overperformano" ───────────────────
            clinical_bonus = (
                float(rng.uniform(-0.15, 0.55)) if ruolo in ('C', 'A') else 0.0
            )
            gol_pg    = float(np.clip(xg_base + clinical_bonus + rng.normal(0, 0.02), 0, 1.0))
            assist_pg = float(np.clip(xa_base + rng.normal(0, 0.02),               0, 0.6))

            records.append({
                'nome':                   f"{ruolo}_{idx:03d}",
                'ruolo':                  ruolo,
                'costo_iniziale':         costo,
                'fanta_media':            round(base_fm, 2),
                'storico_voti_std':       round(storico_std, 3),
                'peak_rate':              round(peak_rate, 3),
                'dribbling_riusciti_p90': round(drib, 2),
                'passaggi_chiave_p90':    round(pkey, 2),
                'gol_pg':                 round(gol_pg,    3),
                'assist_pg':              round(assist_pg, 3),
                'xg_pg':                  round(xg_base,   3),
                'xa_pg':                  round(xa_base,   3),
                # 'ownership_pct' → RIMOSSA: dato inaffidabile pre-torneo
            })
            idx += 1

    return pd.DataFrame(records).reset_index(drop=True)


# ── 2. CHAOS & UPSIDE INDEX ───────────────────────────────────────────────────

def compute_chaos_upside_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola il Chaos & Upside Index (CUI) per ogni giocatore su tre assi.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  CUI = 0.40 × Varianza_Positiva + 0.35 × Chaos_Makers + 0.25 × XG_Delta │
    └─────────────────────────────────────────────────────────────────────┘

    A) VARIANZA POSITIVA (40%)
       variance_score = storico_voti_std × (1 + peak_rate × 2)
       Misura non solo l'irregolarità, ma l'irregolarità verso l'ALTO:
       un giocatore con std=1.5 ma che picca spesso ≥7.0 è più prezioso
       di uno con std=1.5 che picca solo tra 4 e 6.

    B) CHAOS MAKERS (35%)
       chaos_maker = 0.55 × dribbling_p90 + 0.45 × passaggi_chiave_p90
       I dribbling rompono le linee difensive in modo imprevedibile.
       I passaggi chiave creano occasioni da azioni non catalogabili dall'xG.
       Peso leggermente maggiore al dribbling: più difficile da difendere.

    C) XG DELTA — Clinical Finishing (25%)
       xg_delta = clip((gol_pg - xg_pg) / xg_pg, 0, 3)
       xa_delta = clip((assist_pg - xa_pg) / xa_pg, 0, 3)
       clinical = xg_delta + xa_delta
       Cattura i giocatori che segnano/assistono su tiri/azioni "difficili"
       (bassa probabilità xG). Il clip a 0 ignora i giocatori che sottoperformano
       l'xG: non vogliamo penalizzarli, solo premiare i super-clinici.

    Portieri: il CUI è basato solo sulla Varianza Positiva (nessun senso
    assegnare Chaos Makers offensivi a un GK).

    Output aggiunto al DataFrame:
      'cui_var'   → componente varianza normalizzata [0,1]
      'cui_chaos' → componente chaos normalizzata [0,1]
      'cui_xg'    → componente xG delta normalizzata [0,1]
      'cui'       → indice finale normalizzato [0,1]
      'cui_label' → etichetta qualitativa
    """
    df = df.copy()

    # Fill missing columns with sensible defaults if not provided by scrapers
    if 'fanta_media' not in df.columns:
        df['fanta_media'] = df.get('previsione_ia', 6.0)
    if 'storico_voti_std' not in df.columns:
        df['storico_voti_std'] = 0.40
    if 'peak_rate' not in df.columns:
        df['peak_rate'] = 0.15
    if 'dribbling_riusciti_p90' not in df.columns:
        df['dribbling_riusciti_p90'] = 0.50
    if 'passaggi_chiave_p90' not in df.columns:
        df['passaggi_chiave_p90'] = 0.50
    if 'gol_pg' not in df.columns:
        df['gol_pg'] = 0.10
    if 'xg_pg' not in df.columns:
        df['xg_pg'] = 0.10
    if 'assist_pg' not in df.columns:
        df['assist_pg'] = 0.08
    if 'xa_pg' not in df.columns:
        df['xa_pg'] = 0.08

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / max(float(hi - lo), 1e-9)

    # ── A. Varianza Positiva ──────────────────────────────────────────────────
    variance_score = df['storico_voti_std'] * (1.0 + df['peak_rate'] * 2.0)
    norm_var = _norm(variance_score)

    # ── B. Chaos Makers ───────────────────────────────────────────────────────
    chaos_raw = 0.55 * df['dribbling_riusciti_p90'] + 0.45 * df['passaggi_chiave_p90']
    norm_chaos = _norm(chaos_raw)

    # ── C. XG Delta (Clinical Finishing) ─────────────────────────────────────
    xg_delta = (
        (df['gol_pg'] - df['xg_pg']) / df['xg_pg'].clip(lower=0.01)
    ).clip(lower=0.0, upper=3.0)

    xa_delta = (
        (df['assist_pg'] - df['xa_pg']) / df['xa_pg'].clip(lower=0.01)
    ).clip(lower=0.0, upper=3.0)

    clinical_score = (xg_delta + xa_delta).clip(upper=4.0)
    norm_clin = _norm(clinical_score)

    # ── CUI pesato ────────────────────────────────────────────────────────────
    cui_raw = 0.40 * norm_var + 0.35 * norm_chaos + 0.25 * norm_clin

    # Portieri: solo varianza (le metriche offensive non hanno senso)
    is_gk = df['ruolo'] == 'P'
    cui_raw[is_gk] = norm_var[is_gk]

    # ── Salva componenti e normalizza il CUI finale ───────────────────────────
    df['cui_var']   = norm_var.round(4)
    df['cui_chaos'] = norm_chaos.round(4)
    df['cui_xg']    = norm_clin.round(4)
    df['cui']       = _norm(cui_raw).round(4)

    df['cui_label'] = pd.cut(
        df['cui'],
        bins=[0.0, 0.33, 0.67, 1.01],
        labels=['Regolare', 'Variabile', 'Spacca-Partite'],
        include_lowest=True,
    ).astype(str)

    return df


# ── 3. SCORE COMPOSITO FM + CUI ───────────────────────────────────────────────

def build_chaos_score(df: pd.DataFrame, cui_weight: float = 0.55) -> pd.DataFrame:
    """
    Calcola 'score_selezione' per l'ottimizzatore.

    score = fanta_media × (1 + cui_weight × cui)

    Il CUI agisce come MOLTIPLICATORE della FM, non come addendo:
    questo previene che un giocatore con FM scarsa (5.0) ma CUI massimo
    batta uno con FM alta (8.0) e CUI medio.

    Esempi con cui_weight=0.55:
      FM=6.5, CUI=0.00 → score=6.50  ← titolare inamovibile noioso
      FM=6.5, CUI=0.50 → score=8.29  ← variabile ma con base solida
      FM=6.5, CUI=1.00 → score=10.08 ← spacca-partite con regolarità
      FM=5.0, CUI=1.00 → score=7.75  ← troppo irregolare, score penalizzato
      FM=8.0, CUI=0.30 → score=9.32  ← fenomeno + un po' di caos
    """
    df = df.copy()
    df['score_selezione'] = (
        df['fanta_media'] * (1.0 + cui_weight * df['cui'])
    ).round(4)
    return df


# ── 4. OTTIMIZZATORE MILP CON VINCOLI CHAOS ───────────────────────────────────

def optimize_chaos_team(
    df: pd.DataFrame,
    budget: int = BUDGET_DEFAULT,
) -> pd.DataFrame:
    """
    Risolve il problema Knapsack multi-vincolo (MILP via PuLP/CBC).

    VINCOLI:
    ┌─────┬──────────────────────────────────────────────────────────────────┐
    │ L1  │ Composizione esatta: 3P + 8D + 8C + 6A = 25                    │
    │ L2  │ Budget totale ≤ {budget} crediti                                 │
    │ L3  │ SOLID BASE: Σcosto[i∈solid] × x[i] ≥ 50% budget               │
    │     │   solid = giocatori con CUI ≤ mediana del DB (titolari 6/6.5)   │
    │ L4a │ GAME CHANGER MID: Σx[i∈top15%_C] ≥ 2                          │
    │ L4b │ GAME CHANGER ATT: Σx[i∈top15%_A] ≥ 2                          │
    └─────┴──────────────────────────────────────────────────────────────────┘

    COME IL SOLVER BILANCIA I DUE OBIETTIVI:
      Il vincolo L3 "blocca" almeno 125cr su giocatori prevedibili.
      Il vincolo L4 "forza" 2 MID + 2 ATT imprevedibili nel top 15%.
      La funzione obiettivo premia chi ha score = FM × (1 + 0.55 × CUI) alto.
      Il solver trova il portfolio che massimizza questo score rispettando
      sia il tetto di spesa che la struttura obbligata.

      In pratica: costruisce una fondazione solida (L3) e inserisce i
      fuoriclasse del caos (L4), ottimizzando i restanti slot liberamente.
    """
    # ── Soglie per i vincoli L3 e L4 ─────────────────────────────────────────
    cui_median    = float(df['cui'].median())
    cui_top15_th  = float(df['cui'].quantile(0.85))

    # Insiemi di indici per i vincoli
    solid_ids    = df[df['cui'] <= cui_median].index.tolist()
    gc_mid_ids   = df[(df['ruolo'] == 'C') & (df['cui'] >= cui_top15_th)].index.tolist()
    gc_att_ids   = df[(df['ruolo'] == 'A') & (df['cui'] >= cui_top15_th)].index.tolist()

    logger.info(f"CUI mediana DB: {cui_median:.3f}  |  Soglia top-15%: {cui_top15_th:.3f}")
    logger.info(f"Solid Base candidati: {len(solid_ids)}")
    logger.info(f"Game Changer MID (top-15%): {len(gc_mid_ids)}")
    logger.info(f"Game Changer ATT (top-15%): {len(gc_att_ids)}")

    # ── Fallback: se il top-15% ha troppo pochi C/A, allarga al top-25% del ruolo
    if len(gc_mid_ids) < 2:
        alt_th     = float(df[df['ruolo'] == 'C']['cui'].quantile(0.75))
        gc_mid_ids = df[(df['ruolo'] == 'C') & (df['cui'] >= alt_th)].index.tolist()
        logger.warning(f"Fallback Game Changer MID al top-25% ruolo: {len(gc_mid_ids)} candidati")

    if len(gc_att_ids) < 2:
        alt_th     = float(df[df['ruolo'] == 'A']['cui'].quantile(0.75))
        gc_att_ids = df[(df['ruolo'] == 'A') & (df['cui'] >= alt_th)].index.tolist()
        logger.warning(f"Fallback Game Changer ATT al top-25% ruolo: {len(gc_att_ids)} candidati")

    # ── Marca i flag nel DataFrame (utile per il report) ─────────────────────
    df = df.copy()
    df['is_solid']        = df['cui'] <= cui_median
    df['is_game_changer'] = (
        ((df['ruolo'] == 'C') & (df.index.isin(gc_mid_ids))) |
        ((df['ruolo'] == 'A') & (df.index.isin(gc_att_ids)))
    )

    # Tenta con vincolo SOLID BASE progressivamente rilassato se infeasibile.
    # Scala: 50% → 40% → 30% → 0% (solo L1+L2+L4).
    # Il vincolo L4 Game Changer rimane sempre attivo.
    solid_pct_levels = [0.50, 0.40, 0.30, 0.00]
    relax_used       = 0.50

    for solid_pct in solid_pct_levels:
        result = _solve_once(df, budget, solid_ids, gc_mid_ids, gc_att_ids, solid_pct)
        if result is not None:
            relax_used = solid_pct
            if solid_pct < 0.50:
                logger.warning(
                    f"SOLID BASE rilassato a {int(solid_pct*100)}% "
                    f"(pool troppo piccolo o budget insufficiente per il 50% standard)"
                )
            break
    else:
        raise ValueError(
            "Impossibile costruire la rosa nemmeno senza il vincolo SOLID BASE. "
            "Controlla che il dataset abbia almeno 3P + 8D + 8C + 6A entro il budget."
        )

    solid_budget_min = int(budget * relax_used)
    _print_report(result, budget, solid_budget_min, cui_top15_th)
    return result.reset_index(drop=True)


def _solve_once(
    df: pd.DataFrame,
    budget: int,
    solid_ids: list,
    gc_mid_ids: list,
    gc_att_ids: list,
    solid_pct: float,
) -> pd.DataFrame | None:
    """Costruisce e risolve il MILP con una specifica soglia SOLID BASE."""
    prob = LpProblem("FantaChaos_Optimizer", LpMaximize)
    x    = LpVariable.dicts("x", df.index, cat="Binary")

    # Obiettivo: massimizza score composito FM × (1 + CUI_weight)
    scores = df['score_selezione'].astype(float)
    if scores.isna().any() or np.isinf(scores).any():
        scores = scores.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    prob += lpSum(float(scores[i]) * x[i] for i in df.index)

    # ── L1: Composizione esatta ────────────────────────────────────────────────
    for ruolo, n in ROSA.items():
        ids = [i for i in df.index if df.loc[i, 'ruolo'] == ruolo]
        prob += lpSum(x[i] for i in ids) == n, f"Comp_{ruolo}"

    # ── L2: Budget totale ──────────────────────────────────────────────────────
    prob += (
        lpSum(float(df.loc[i, 'costo_iniziale']) * x[i] for i in df.index) <= budget,
        "Budget_totale",
    )

    # ── L3: SOLID BASE (disattivato quando solid_pct == 0) ────────────────────
    if solid_pct > 0.0 and solid_ids:
        solid_budget_min = int(budget * solid_pct)
        max_solid_spend  = int(
            df.loc[solid_ids, 'costo_iniziale']
              .nlargest(sum(ROSA.values()))
              .sum()
        )
        if solid_budget_min <= max_solid_spend:
            prob += (
                lpSum(float(df.loc[i, 'costo_iniziale']) * x[i] for i in solid_ids)
                >= solid_budget_min,
                "SolidBase_Budget",
            )

    # ── L4: GAME CHANGERS ─────────────────────────────────────────────────────
    if len(gc_mid_ids) >= 2:
        prob += lpSum(x[i] for i in gc_mid_ids) >= 2, "GameChanger_MID"
    if len(gc_att_ids) >= 2:
        prob += lpSum(x[i] for i in gc_att_ids) >= 2, "GameChanger_ATT"

    prob.solve(PULP_CBC_CMD(msg=0))
    if LpStatus[prob.status] != 'Optimal':
        return None

    selected = [i for i in df.index if (x[i].varValue or 0) > 0.5]
    if len(selected) != sum(ROSA.values()):
        return None

    return df.loc[selected].copy()


# ── 5. REPORT DIAGNOSTICO ─────────────────────────────────────────────────────

def _bar(value: float, width: int = 10) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def _print_report(
    team: pd.DataFrame,
    budget: int,
    solid_min: int,
    gc_threshold: float,
) -> None:
    total_cost  = int(team['costo_iniziale'].sum())
    total_score = round(team['score_selezione'].sum(), 2)

    solid_spend  = int(team[team['is_solid']]['costo_iniziale'].sum())
    chaos_spend  = total_cost - solid_spend
    n_gc_mid     = int(team[(team['ruolo'] == 'C') & team['is_game_changer']].shape[0])
    n_gc_att     = int(team[(team['ruolo'] == 'A') & team['is_game_changer']].shape[0])
    pct_solid    = round(solid_spend / budget * 100, 1)
    pct_chaos    = round(chaos_spend / budget * 100, 1)

    sep  = "═" * 72
    sep2 = "─" * 72

    print(f"\n{sep}")
    print(f"  FANTACALCIO — CHAOS & UPSIDE INDEX OPTIMIZER")
    print(f"  Budget usato: {total_cost}/{budget} cr  |  Score totale: {total_score}")
    print(f"{sep}")

    print(f"\n  BILANCIO SOLID BASE vs GAME CHANGERS")
    print(f"  {'Solidi (CUI ≤ mediana):':<30s} {solid_spend:>4d}cr ({pct_solid}%)  "
          f"[min {solid_min}cr richiesti — {'✓ OK' if solid_spend >= solid_min else '✗ VIOLATO'}]")
    print(f"  {'Variabili/Spacca-Partite:':<30s} {chaos_spend:>4d}cr ({pct_chaos}%)")
    print(f"  {'Game Changer MID (≥2):':<30s} {n_gc_mid}  "
          f"{'✓ OK' if n_gc_mid >= 2 else '✗ VIOLATO'}")
    print(f"  {'Game Changer ATT (≥2):':<30s} {n_gc_att}  "
          f"{'✓ OK' if n_gc_att >= 2 else '✗ VIOLATO'}")

    print(f"\n{sep2}")
    print(f"  {'GIOCATORE':<14s} {'R':>2s}  {'CR':>4s}  {'FM':>5s}  "
          f"{'CUI':>5s}  {'BREAKDOWN CUI (Var/Chs/XG)':>30s}  "
          f"{'SCORE':>6s}  PROFILO")
    print(f"{sep2}")

    for ruolo in ['P', 'D', 'C', 'A']:
        sub = team[team['ruolo'] == ruolo].sort_values('costo_iniziale', ascending=False)
        ruolo_names = {'P': 'PORTIERI', 'D': 'DIFENSORI', 'C': 'CENTROCAMPISTI', 'A': 'ATTACCANTI'}
        avg_cui = sub['cui'].mean()
        print(f"\n  ── {ruolo_names[ruolo]} ({len(sub)})  CUI medio: {avg_cui:.2f} ──")

        for _, row in sub.iterrows():
            gc_flag = " ★GC" if row['is_game_changer'] else ("  ·" if row['is_solid'] else "  ~")
            breakdown = (
                f"V={row['cui_var']:.2f} C={row['cui_chaos']:.2f} X={row['cui_xg']:.2f}"
            )
            cui_bar   = _bar(float(row['cui']))
            label_str = row['cui_label']
            print(
                f"  {row['nome']:<14s} {row['ruolo']:>2s}  "
                f"{int(row['costo_iniziale']):>4d}cr  "
                f"FM={row['fanta_media']:.1f}  "
                f"CUI={row['cui']:.2f}  "
                f"[{cui_bar}]  "
                f"{breakdown:>28s}  "
                f"sc={row['score_selezione']:.2f}  "
                f"[{label_str:>14s}]{gc_flag}"
            )

    print(f"\n{sep}")
    print(f"  Spesa per reparto:")
    for ruolo in ['P', 'D', 'C', 'A']:
        sub_t  = team[team['ruolo'] == ruolo]
        spesa  = int(sub_t['costo_iniziale'].sum())
        pct    = round(spesa / budget * 100, 1)
        avg_c  = round(sub_t['cui'].mean(), 2)
        n_gc   = int(sub_t['is_game_changer'].sum())
        n_sol  = int(sub_t['is_solid'].sum())
        print(f"    {ruolo}: {spesa:>3d}cr ({pct:>4.1f}%)  "
              f"CUI medio={avg_c:.2f}  "
              f"Solidi={n_sol}  GC={n_gc}")
    print(f"{sep}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Uso:
    #   python3 chaos_optimizer.py                        → mock, 500cr
    #   python3 chaos_optimizer.py 250                    → mock, 250cr
    #   python3 chaos_optimizer.py 500 quotazioni.xlsx    → dati reali, 500cr
    #   python3 chaos_optimizer.py 500 data/mio_file.csv  → dati reali, CSV
    budget          = int(sys.argv[1])    if len(sys.argv) > 1 else BUDGET_DEFAULT
    quotazioni_path = sys.argv[2]         if len(sys.argv) > 2 else None

    use_real = quotazioni_path is not None and os.path.exists(quotazioni_path)

    print(f"\n{'═'*72}")
    print(f"  FANTACALCIO — CHAOS & UPSIDE INDEX OPTIMIZER  (budget={budget}cr)")
    print(f"  Ownership rimossa. Rosa: 3P + 8D + 8C + 6A")
    if use_real:
        print(f"  Modalità: DATI REALI  ({quotazioni_path})")
    else:
        print(f"  Modalità: MOCK  (costi scalati {budget/_BUDGET_REF:.2f}x su ref {_BUDGET_REF}cr)")
        if quotazioni_path:
            print(f"  ATTENZIONE: file '{quotazioni_path}' non trovato — uso mock")
    print(f"{'═'*72}\n")

    # ── Step 1: dati reali o mock ─────────────────────────────────────────────
    if use_real:
        from chaos_data import prepare_chaos_data
        df_raw = prepare_chaos_data(quotazioni_path, budget=budget)
    else:
        df_raw = generate_mock_players(n_per_role=80, seed=42, budget=budget)
    print(f"Dataset mock: {len(df_raw)} giocatori  "
          f"(P={len(df_raw[df_raw.ruolo=='P'])}  D={len(df_raw[df_raw.ruolo=='D'])}  "
          f"C={len(df_raw[df_raw.ruolo=='C'])}  A={len(df_raw[df_raw.ruolo=='A'])})")

    # ── Step 2: Calcola Chaos & Upside Index ──────────────────────────────────
    df_cui = compute_chaos_upside_index(df_raw)

    print("\nDistribuzione CUI per ruolo:")
    print(f"  {'R':<3}  {'media':>6}  {'std':>6}  {'top-15% soglia':>15}  "
          f"{'Spacca-Partite':>15}  {'Regolari':>10}")
    for r in ['P', 'D', 'C', 'A']:
        sub = df_cui[df_cui['ruolo'] == r]
        n_sp = int((sub['cui_label'] == 'Spacca-Partite').sum())
        n_re = int((sub['cui_label'] == 'Regolare').sum())
        th85 = round(float(sub['cui'].quantile(0.85)), 3)
        print(f"  {r:<3}  {sub['cui'].mean():>6.3f}  {sub['cui'].std():>6.3f}  "
              f"{th85:>15.3f}  {n_sp:>15d}  {n_re:>10d}")

    # ── Step 3: Calcola score composito ───────────────────────────────────────
    df_scored = build_chaos_score(df_cui, cui_weight=0.55)

    print("\nTop 10 giocatori per score_selezione (pre-ottimizzazione):")
    top10 = df_scored.nlargest(10, 'score_selezione')[
        ['nome', 'ruolo', 'costo_iniziale', 'fanta_media', 'cui', 'cui_label', 'score_selezione']
    ]
    for _, r in top10.iterrows():
        print(f"  {r['nome']:<14s} {r['ruolo']:>2s}  {int(r['costo_iniziale']):>4d}cr  "
              f"FM={r['fanta_media']:.1f}  CUI={r['cui']:.2f}  "
              f"[{str(r['cui_label']):>14s}]  score={r['score_selezione']:.2f}")

    # ── Step 4: Ottimizzazione MILP ───────────────────────────────────────────
    print(f"\nOttimizzazione MILP in corso (CBC)...")
    team = optimize_chaos_team(df_scored, budget=budget)

    # ── Step 5: Game Changers non selezionati (watchlist) ────────────────────
    gc_th   = float(df_scored['cui'].quantile(0.85))
    gc_all  = df_scored[df_scored['cui'] >= gc_th].sort_values('score_selezione', ascending=False)
    gc_out  = gc_all[~gc_all['nome'].isin(team['nome'])]

    if not gc_out.empty:
        print("Top 5 Game Changers NON selezionati (watchlist acquisti futuri):")
        for _, r in gc_out.head(5).iterrows():
            print(f"  {r['nome']:<14s} {r['ruolo']:>2s}  {int(r['costo_iniziale']):>4d}cr  "
                  f"FM={r['fanta_media']:.1f}  CUI={r['cui']:.2f}  "
                  f"score={r['score_selezione']:.2f}  [{r['cui_label']}]")
