"""
optimizer.py — Ottimizzatore Fantacalcio con vincoli multi-tier.

Algoritmo: MILP (Mixed Integer Linear Programming) via PuLP + CBC.
Modello:   Knapsack multi-categoria con 6 livelli di vincoli gerarchici.

PERCHÉ L'APPROCCIO PRECEDENTE PRODUCEVA ROSE PIATTE:
  Il programma lineare puro massimizza lo score senza sapere che in Fantacalcio
  comprare 6 attaccanti mediocri a 5cr è peggio di comprare 2 top a 50cr + 4 a 1cr.
  Il solver non ha preferenze tra costi: se uno da 5cr ha score simile a uno da 50cr,
  sceglie quello da 5cr per risparmiare budget (e poi lo usa altrove male).

SOLUZIONE: 6 livelli di vincoli sovrapposti
  L1 — Composizione esatta: 3P + 8D + 8C + 6A = 25 giocatori
  L2 — Budget totale
  L3 — Band di budget per reparto [min%, max%] → impedisce la dispersione
  L4 — Slot "top tier" obbligatori per reparto → forza qualità in posizioni chiave
  L5 — Costo medio minimo per reparto → anti-piattume assoluto
  L6 — Qualità minima portieri → almeno 1 GK sopra la mediana di score
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

# ── L1: Composizione rosa standard Fantacalcio ───────────────────────────────
ROSA       = {'P': 3, 'D': 8, 'C': 8, 'A': 6}   # 25 giocatori totali
RUOLI_VALIDI = set(ROSA.keys())

# ── L3: Band di budget [min%, max%] per reparto per strategia ────────────────
#
# Logica:
#   AGGRESSIVA  → GK economici, difensori pochi ma buoni, centrocampisti qualità,
#                 MASSIMA concentrazione sull'attacco (2 top + scommesse).
#   CONSERVATIVA → distribuzione bilanciata, nessun reparto sacrificato.
#   SCOMMESSE   → GK/DEF economici, ATT con miglior rapporto FM/costo.
#
# La somma dei minimi è < 100% (lascia flessibilità al solver).
# La somma dei massimi è > 100% (non tutti i cap sono mai raggiungibili insieme).
#
# Con budget=500:
#   Aggressiva A: min=175cr, max=280cr per 6 attaccanti → avg 29-47cr/att
#   Aggressiva C: min=110cr, max=165cr per 8 medi → avg 14-21cr/medi
#   Aggressiva D: min= 60cr, max=100cr per 8 dif  → avg  8-13cr/dif
#   Aggressiva P: min= 10cr, max= 45cr per 3 GK   → avg  3-15cr/GK

BUDGET_BANDS: dict[str, dict[str, tuple[float, float]]] = {
    'master': {
        'P': (0.04, 0.11),
        'D': (0.17, 0.26),
        'C': (0.30, 0.40),
        'A': (0.26, 0.44),
    },
    'aggressiva': {
        'P': (0.02, 0.09),
        'D': (0.12, 0.20),
        'C': (0.30, 0.42),
        'A': (0.28, 0.48),
    },
    'moneyball': {
        'P': (0.02, 0.07),
        'D': (0.13, 0.21),
        'C': (0.28, 0.42),
        'A': (0.28, 0.48),
    },
    'sprint_calendario': {
        'P': (0.04, 0.11),
        'D': (0.15, 0.24),
        'C': (0.28, 0.40),
        'A': (0.28, 0.46),
    }
}

# ── L4a: Slot ATT (percentile) ───────────────────────────────────────────────
#
# Per gli attaccanti manteniamo l'approccio percentile: obbliga il solver
# a scegliere ≥ n_min dal top_pct% più costoso del pool ATT.
# GK: gestito da L6 (top squadra difensiva), non da L4.

ATT_TOP_SLOTS: dict[str, tuple[int, float]] = {
    'aggressiva':   (2, 0.20),  # 2 attaccanti nel top 20%
    'conservativa': (2, 0.38),  # 2 attaccanti solidi
    'scommesse':    (0, 0.25),  # nessun obbligo di top ATT — valore > nome
    'vip_pesante':  (2, 0.20),  # come aggressiva
    'difesa_forte': (1, 0.20),  # 1 top ATT + resto economici
}

# ── L4b: Soglie di costo assoluto per DEF e MID ──────────────────────────────
#
# Impone che il solver scelga ≥ n_min giocatori con costo > budget * cost_pct.
# Esempio a budget=500:
#   DEF: ≥1 difensore con costo > 500×4% = 20cr (wing-back come Dimarco, Theo)
#   MID: ≥2 centrocampisti con costo > 500×5% = 25cr (Barella, Calhanoglu, ...)
# Questo approccio scala con il budget ed è indipendente dalla distribuzione del pool.

TIER_COST_PCT: dict[str, dict[str, tuple[int, float]]] = {
    'aggressiva':   {'D': (1, 0.040), 'C': (2, 0.050)},
    'conservativa': {'D': (2, 0.038), 'C': (2, 0.048)},
    'scommesse':    {'D': (1, 0.030), 'C': (1, 0.040)},
    'vip_pesante':  {'D': (1, 0.040), 'C': (2, 0.050)},
    'difesa_forte': {'D': (2, 0.045), 'C': (1, 0.040)},  # 2 DEF premium
}

# ── Top squadre difensive per qualità portieri ───────────────────────────────
#
# Usato dal vincolo L6: almeno 1 GK titolare deve provenire da una squadra
# storicamente forte difensivamente (minor xGA). Se il dataset non ha la
# colonna 'squadra', il vincolo cade sul top-40% per score.

TOP_GK_SQUADS: frozenset[str] = frozenset({
    "Inter", "Juventus", "Milan", "Napoli", "Atalanta", "Roma", "Lazio",
    "inter", "juventus", "milan", "napoli", "atalanta", "roma", "lazio",
})

# ── L5: Costo minimo totale per reparto (anti-piattume assoluto) ─────────────
#
# Con n giocatori nel ruolo, impone: somma_costi >= n * min_avg_costo
# Evita che il solver scelga 8 difensori tutti a 1cr.
#
# Valori basati su prezzi reali Fantacalcio.it (budget=500):
#   Aggressiva:  GK ~2cr avg, DEF ~8cr avg, MID ~14cr avg, ATT ~30cr avg
#   Conservativa: tutti più alti, nessun reparto "sacrificato"
#   Scommesse:   GK/DEF economici, ATT valore/prezzo

MIN_AVG_COST: dict[str, dict[str, float]] = {
    'aggressiva':   {'P': 2.0,  'D':  7.0, 'C': 16.0, 'A': 20.0},
    'conservativa': {'P': 4.0,  'D': 11.0, 'C': 17.0, 'A': 15.0},
    'scommesse':    {'P': 1.5,  'D':  4.0, 'C':  9.0, 'A': 10.0},
    'vip_pesante':  {'P': 2.0,  'D':  8.0, 'C': 15.0, 'A': 20.0},
    'difesa_forte': {'P': 6.0,  'D': 13.0, 'C': 13.0, 'A': 12.0},  # GK+DEF premium
}


# ── Costo massimo per singolo giocatore (anti-star per scommesse) ────────────
#
# In scommesse il bot non deve comprare Vlahovic a 60cr: impedisce che il solver
# scelga 1 star + 5 ciabatte. Al massimo 1 giocatore "premium" tollerato.
MAX_PLAYER_COST_PCT: dict[str, float] = {
    'scommesse': 0.09,   # max 45cr a budget=500
}

# ── Funzione principale ───────────────────────────────────────────────────────

def optimize_team(
    data_path: str,
    budget: int = 500,
    strategy: str = 'conservativa',
) -> pd.DataFrame:
    """
    Costruisce la rosa ottimale da 25 giocatori (3P+8D+8C+6A).

    Garantisce:
    - Composizione esatta per ruolo (vincolo duro)
    - Budget rispettato (vincolo duro)
    - Band di budget per reparto (vincolo morbido con fallback)
    - Slot top-tier obbligatori per evitare rose piatte (vincolo morbido con fallback)
    - Costo medio minimo per reparto (vincolo morbido con fallback)
    - Almeno 1 GK di qualità (vincolo morbido)

    In caso di infeasibility, rilassa progressivamente i vincoli morbidi
    fino a trovare una soluzione. Se ancora impossibile lancia ValueError.
    """
    df = _load_and_validate(data_path)
    s  = strategy.lower().strip()

    if s in ('listone', 'listone_master', 'listone_auto'):
        from listone_optimizer import optimize_listone_auto
        result_df, best_form, best_score = optimize_listone_auto(df, budget=budget)
        return result_df

    for candidate in ('score_selezione', 'previsione_ia', 'fanta_media', 'costo_iniziale'):
        if candidate in df.columns:
            obj_col = candidate
            break
    else:
        raise ValueError("Nessuna colonna di score disponibile nel dataset.")

    # Tenta prima con tutti i vincoli, poi con rilassamenti progressivi
    for relax_level in range(4):
        result = _build_and_solve(df, budget, s, obj_col, relax_level)
        if result is not None and not result.empty:
            if relax_level > 0:
                logger.warning(
                    f"Soluzione trovata con rilassamento livello {relax_level} "
                    f"(pool troppo piccolo o budget insufficiente per i vincoli completi)"
                )
            return result

    raise ValueError(
        "Impossibile costruire la rosa con i vincoli dati. "
        "Prova ad aumentare il budget o caricare un file quotazioni con più giocatori."
    )


# ── Caricamento e validazione dati ────────────────────────────────────────────

def _load_and_validate(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df[df['nome'].astype(str).str.strip() != ''].copy()
    df['ruolo'] = df['ruolo'].astype(str).str.upper().str.strip()
    df = df[df['ruolo'].isin(RUOLI_VALIDI)].copy().reset_index(drop=True)

    # Sanitize numeric columns to prevent PuLP NaN/inf errors
    for num_col in ['costo_iniziale', 'score_selezione', 'previsione_ia', 'fanta_media', 'vip_total']:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce').fillna(0.0)

    counts = df['ruolo'].value_counts()
    for r, n in ROSA.items():
        if counts.get(r, 0) < n:
            raise ValueError(
                f"Giocatori insufficienti per ruolo {r}: "
                f"servono {n}, trovati {counts.get(r, 0)}. "
                "Carica un file con più giocatori."
            )

    # Filtro titolarità: attivo solo se i dati sono reali (non tutti zero)
    if 'titolarita_pct' in df.columns and df['titolarita_pct'].median() > 0.10:
        df_tit = df[df['titolarita_pct'] >= 0.20].copy().reset_index(drop=True)
        if all(df_tit['ruolo'].value_counts().get(r, 0) >= n for r, n in ROSA.items()):
            logger.info(
                f"Filtro titolarità attivo: {len(df)} → {len(df_tit)} giocatori"
            )
            df = df_tit

    # Filtro presenze: esclude giocatori con storia di infortuni frequenti
    if 'presenze' in df.columns and df['presenze'].median() > 8:
        df_pres = df[df['presenze'] >= 8].copy().reset_index(drop=True)
        if all(df_pres['ruolo'].value_counts().get(r, 0) >= n for r, n in ROSA.items()):
            logger.info(
                f"Filtro presenze: {len(df)} → {len(df_pres)} giocatori (min 8 presenze)"
            )
            df = df_pres

    return df


# ── Calcolo tier ──────────────────────────────────────────────────────────────

def _top_tier_indices(df: pd.DataFrame, ruolo: str, top_pct: float, n_min: int) -> list[int]:
    """
    Restituisce gli indici dei giocatori nel top_pct% più costoso del ruolo.
    Garantisce sempre almeno n_min candidati (espande il pool se necessario).
    """
    sub = df[df['ruolo'] == ruolo]
    if sub.empty:
        return []

    # Quanti giocatori nel top_pct%? Almeno n_min, al massimo tutti.
    n_target = max(n_min, int(np.ceil(len(sub) * top_pct)))
    n_target = min(n_target, len(sub))

    return sub.nlargest(n_target, 'costo_iniziale').index.tolist()


# ── Build + Solve ─────────────────────────────────────────────────────────────

def _build_and_solve(
    df: pd.DataFrame,
    budget: int,
    strategy: str,
    obj_col: str,
    relax_level: int,
) -> pd.DataFrame | None:
    """
    Costruisce e risolve il modello MILP.

    relax_level=0: tutti i vincoli attivi
    relax_level=1: band allargate 25%, tier slot ridotti di 1
    relax_level=2: band allargate 50%, tier slot ridotti di 2
    relax_level=3: solo composizione + budget totale + GK quality
    """
    bands   = BUDGET_BANDS.get(strategy, BUDGET_BANDS['master'])
    min_avg = MIN_AVG_COST.get(strategy, MIN_AVG_COST.get('master', {}))

    # Fattori di rilassamento
    band_relax  = 1.0 + 0.25 * relax_level   # espande le band
    slot_reduce = relax_level                  # riduce n_min dei top slot
    use_soft    = (relax_level < 3)            # disattiva L3/L4/L5 al livello 3

    prob = LpProblem("Fantacalcio_MultiTier", LpMaximize)
    x    = LpVariable.dicts("x", df.index, cat="Binary")

    # ── Obiettivo: score strategico con penalità giocatori mediocri ───────────
    #
    # Un giocatore è "mediocre sopravvalutato" se:
    #   - costa più di 5 crediti (non è una scommessa economica)
    #   - ha VIP_total sotto la media del suo ruolo (non porta valore nascosto)
    # Penalità: riduce il coefficient obiettivo del 15-20%, proporzionale al costo.
    # Effetto: il solver preferisce giocatori con valore intrinseco, non solo costo.
    obj_coeffs: dict = {}
    if 'vip_total' in df.columns:
        role_mean_vip = df.groupby('ruolo')['vip_total'].transform('mean')
        is_mediocre   = (df['costo_iniziale'] > 5) & (df['vip_total'] < role_mean_vip)
        max_cost      = max(float(df['costo_iniziale'].max()), 1.0)
        for i in df.index:
            base = float(df.loc[i, obj_col])
            if bool(is_mediocre.loc[i]):
                # Penalità cresce con il costo: chi paga 30cr mediocre è più "sbagliato"
                pen = (float(df.loc[i, 'costo_iniziale']) - 5.0) / max_cost * 0.20
                obj_coeffs[i] = base * (1.0 - pen)
            else:
                obj_coeffs[i] = base
    else:
        for i in df.index:
            obj_coeffs[i] = float(df.loc[i, obj_col])

    prob += lpSum(obj_coeffs[i] * x[i] for i in df.index)

    # ── L1: Composizione esatta per ruolo ─────────────────────────────────────
    for ruolo, n in ROSA.items():
        role_ids = [i for i in df.index if df.loc[i, 'ruolo'] == ruolo]
        prob += lpSum(x[i] for i in role_ids) == n, f"Comp_{ruolo}"

    # ── L2: Budget totale ─────────────────────────────────────────────────────
    prob += (
        lpSum(df.loc[i, 'costo_iniziale'] * x[i] for i in df.index) <= budget,
        "Budget_totale",
    )

    if use_soft:
        for ruolo in RUOLI_VALIDI:
            role_ids = [i for i in df.index if df.loc[i, 'ruolo'] == ruolo]
            if not role_ids:
                continue

            spend = lpSum(df.loc[i, 'costo_iniziale'] * x[i] for i in role_ids)
            lo_pct, hi_pct = bands[ruolo]

            max_possible = df.loc[role_ids, 'costo_iniziale'].nlargest(ROSA[ruolo]).sum()
            min_possible = df.loc[role_ids, 'costo_iniziale'].nsmallest(ROSA[ruolo]).sum()

            # ── L3: Band di budget per reparto ────────────────────────────────
            lo_val = int(budget * lo_pct / band_relax)
            hi_val = int(budget * hi_pct * band_relax)

            if lo_val >= 1 and lo_val <= max_possible:
                prob += spend >= lo_val, f"BandMin_{ruolo}"

            if hi_val >= min_possible:
                prob += spend <= hi_val, f"BandMax_{ruolo}"

            # ── L4a: Slot top-tier ATT (percentile) ───────────────────────────
            if ruolo == 'A':
                n_min_orig, top_pct = ATT_TOP_SLOTS.get(strategy, (1, 0.25))
                n_min = max(0, n_min_orig - slot_reduce)
                if n_min > 0:
                    top_ids = _top_tier_indices(df, ruolo, top_pct, n_min)
                    valid   = [i for i in top_ids if i in df.index]
                    if len(valid) >= n_min:
                        prob += lpSum(x[i] for i in valid) >= n_min, f"TopSlot_A"

            # ── L4b: Tier DEF e MID — soglia costo assoluta ───────────────────
            #
            # Vincola il solver a scegliere ≥ n_min giocatori con costo ≥ soglia.
            # Esempio a budget=500: DEF ≥1 con costo≥20cr, MID ≥2 con costo≥25cr.
            tier_cfg = TIER_COST_PCT.get(strategy, {})
            if ruolo in tier_cfg:
                n_min_orig, cost_pct = tier_cfg[ruolo]
                n_min     = max(0, n_min_orig - slot_reduce)
                threshold = budget * cost_pct
                top_cost  = [i for i in role_ids if df.loc[i, 'costo_iniziale'] >= threshold]
                # Fallback: se pochi player sopra soglia, usa i top-N per costo
                if len(top_cost) < n_min:
                    top_cost = sorted(role_ids,
                                      key=lambda i: df.loc[i, 'costo_iniziale'],
                                      reverse=True)[:max(n_min + 2, 5)]
                if n_min > 0 and len(top_cost) >= n_min:
                    prob += lpSum(x[i] for i in top_cost) >= n_min, f"TierCost_{ruolo}"

            # ── L5: Costo medio minimo per reparto ────────────────────────────
            min_total = int(min_avg.get(ruolo, 1.0) * ROSA[ruolo] / band_relax)
            if min_total >= 1 and min_total <= max_possible:
                prob += spend >= min_total, f"MinAvg_{ruolo}"

    # ── L7: Costo max per singolo giocatore (solo scommesse) ─────────────────
    #
    # Impedisce di comprare un unico star da 60cr + 5 riserve economiche.
    # Al massimo 1 giocatore "premium" tollerato.
    max_pct = MAX_PLAYER_COST_PCT.get(strategy)
    if max_pct and use_soft:
        max_cost_single = int(budget * max_pct)
        expensive_ids = [i for i in df.index if df.loc[i, 'costo_iniziale'] > max_cost_single]
        if expensive_ids:
            prob += lpSum(x[i] for i in expensive_ids) <= 1, "Scommesse_MaxPlayerCost"

    # ── L6: GK qualità — preferisce top squadre difensive ────────────────────
    #
    # Se il dataset ha la colonna 'squadra', impone 1 GK da top difesa.
    # Altrimenti cade sul top-40% di score (logica precedente, più robusta).
    gk_ids = [i for i in df.index if df.loc[i, 'ruolo'] == 'P']
    if gk_ids:
        if 'squadra' in df.columns:
            gk_top_squad = [
                i for i in gk_ids
                if str(df.loc[i, 'squadra']).strip() in TOP_GK_SQUADS
            ]
        else:
            gk_top_squad = []

        if len(gk_top_squad) >= 1:
            prob += lpSum(x[i] for i in gk_top_squad) >= 1, "GK_top_squad"
        else:
            # Fallback: almeno 1 GK nel top 40% per score
            gk_scores  = df.loc[gk_ids, obj_col]
            gk_thresh  = gk_scores.quantile(0.60)
            gk_quality = [i for i in gk_ids if df.loc[i, obj_col] >= gk_thresh]
            if gk_quality:
                prob += lpSum(x[i] for i in gk_quality) >= 1, "GK_quality"

    # ── Risolvi ───────────────────────────────────────────────────────────────
    prob.solve(PULP_CBC_CMD(msg=0))

    if LpStatus[prob.status] != 'Optimal':
        logger.debug(f"Relax={relax_level}: {LpStatus[prob.status]}")
        return None

    selected = [i for i in df.index if (x[i].varValue or 0) > 0.5]
    if len(selected) != sum(ROSA.values()):
        logger.debug(f"Relax={relax_level}: selezionati {len(selected)} invece di 25")
        return None

    result = df.loc[selected].reset_index(drop=True)
    _log_summary(result, budget, strategy, obj_col)
    return result


# ── Log diagnostico ───────────────────────────────────────────────────────────

def _log_summary(team: pd.DataFrame, budget: int, strategy: str, obj_col: str) -> None:
    total_cost  = int(team['costo_iniziale'].sum())
    total_score = round(team[obj_col].sum(), 2)
    logger.info(f"\n{'='*55}")
    logger.info(f"  ROSA OTTIMIZZATA — {strategy.upper()} — Budget {budget}")
    logger.info(f"  Costo totale: {total_cost}/{budget} cr.  Score: {total_score}")
    logger.info(f"{'='*55}")
    for ruolo in ['P', 'D', 'C', 'A']:
        sub = team[team['ruolo'] == ruolo].sort_values('costo_iniziale', ascending=False)
        spesa = int(sub['costo_iniziale'].sum())
        pct   = round(spesa / budget * 100, 1)
        nomi  = ", ".join(
            f"{r['nome']} ({int(r['costo_iniziale'])}cr)" for _, r in sub.iterrows()
        )
        logger.info(f"  {ruolo}: {spesa}cr ({pct}%) — {nomi}")
    logger.info(f"{'='*55}\n")


# ── Confronto 5 squadre ───────────────────────────────────────────────────────

_FIVE_STRATEGIES: list[tuple[str, str, str]] = [
    ("listone",      "🏆 Listone Master", "Iper-concentrazione 11 Top + 14 Coperture 1cr"),
    ("conservativa", "Conservativa",      "Rendimento sicuro e costante"),
    ("aggressiva",   "Aggressiva",        "Massimo attacco e bonus"),
    ("scommesse",    "Scommesse",         "Miglior rapporto FM/costo"),
    ("vip_pesante",  "VIP Premium",       "Giocatori tatticamente sottovalutati"),
    ("difesa_forte", "Difesa Forte",      "Clean sheet + top attaccante"),
]


def _team_titolari(team: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        team[team['ruolo'] == 'P'].head(1),
        team[team['ruolo'] == 'D'].head(4),
        team[team['ruolo'] == 'C'].head(4),
        team[team['ruolo'] == 'A'].head(3),
    ])


def _team_pros_cons(
    team: pd.DataFrame,
    strategy: str,
    punti: float,
    spesa: dict[str, int],
    budget: int,
) -> tuple[list[str], list[str]]:
    """Genera commenti pro/contro calibrati sulla composizione reale della squadra."""
    pro:    list[str] = []
    contro: list[str] = []
    s = strategy.lower()

    titolari = _team_titolari(team)
    score_col = next(
        (c for c in ('score_selezione', 'previsione_ia') if c in titolari.columns),
        None,
    )
    varianza = round(titolari[score_col].std(), 2) if score_col else 0.0

    pct_att = round(spesa.get('A', 0) / budget * 100)
    pct_def = round(spesa.get('D', 0) / budget * 100)
    pct_mid = round(spesa.get('C', 0) / budget * 100)
    pct_gk  = round(spesa.get('P', 0) / budget * 100)
    budget_rimasto = budget - sum(spesa.values())

    top_att = team[team['ruolo'] == 'A'].nlargest(2, 'costo_iniziale')
    gk_row  = team[team['ruolo'] == 'P'].nlargest(1, 'costo_iniziale')

    if 'conserv' in s:
        pro.append(f"Varianza bassa ({varianza:.1f}): risultati prevedibili ogni giornata")
        pro.append("FM storica come ancora: scelti solo chi ha già dimostrato di rendere")
        if varianza < 0.9:
            pro.append("Ideale per leghe in cui conta l'accumulo: minimize le giornate a zero")
        contro.append("Basso upside: difficile dominare con punteggi da top 5%")
        if pct_att < 30:
            contro.append(f"Attacco contenuto ({pct_att}% budget): pochi bonus se gli attaccanti non segnano")

    elif 'agg' in s:
        nomi_top = " + ".join(top_att['nome'].head(2).tolist()) if not top_att.empty else ""
        pro.append(f"Attacco forte ({pct_att}% budget): massima produzione bonus prevista")
        if nomi_top:
            pro.append(f"Coppia ATT premium: {nomi_top}")
        contro.append(f"Varianza alta ({varianza:.1f}): rischio settimane a zero se gli attaccanti non segnano")
        if pct_def < 15:
            contro.append(f"Difesa sacrificata ({pct_def}% budget): più esposizione ai malus")

    elif 'sco' in s or 'hype' in s:
        pro.append("Massimo valore per credito: ogni punto FM costa meno che nelle altre strategie")
        pro.append("Libera budget per il mercato: crediti in più per prendere la prossima rivelazione")
        contro.append("I 'sottovalutati' possono deludere se il mercato aveva ragione")
        contre_fm = round(punti, 1)
        if contre_fm < 65:
            contro.append(f"FM prevista titolari ({contre_fm}) inferiore alle strategie premium")

    elif 'vip' in s:
        pro.append("Cattura valore tattico non prezzato: quinte d'attacco, trequartisti listati C/D")
        pro.append("Ideale se i tuoi avversari usano solo statistiche aggregate")
        contro.append("Richiede vip_config.json aggiornato con i profili tattici corretti")
        contro.append("Meno affidabile con dati statistici scarsi (prime giornate di stagione)")

    elif 'difesa' in s:
        gk_nome = gk_row.iloc[0]['nome'] if not gk_row.empty else "?"
        gk_costo = int(gk_row.iloc[0]['costo_iniziale']) if not gk_row.empty else 0
        pro.append(f"GK+DEF premium ({pct_gk + pct_def}% budget): clean sheet frequenti → +3 ogni settimana")
        pro.append(f"Portiere di qualità: {gk_nome} ({gk_costo} cr.)")
        contro.append(f"Attacco ridotto ({pct_att}% budget): dipende molto da 1-2 attaccanti top")
        contro.append("Se la difesa non fa clean sheet la settimana è persa in partenza")

    # Pro/contro universali
    if budget_rimasto >= 15:
        pro.append(f"Avanzano {budget_rimasto} cr.: flessibilità per gli svincoli di mercato")
    if punti >= 70:
        pro.append(f"FM titolari prevista alta ({punti}): sopra la media delle leghe competitive")
    elif punti < 63:
        contro.append(f"FM titolari prevista ({punti}) sotto la soglia competitiva")

    return pro[:4], contro[:3]


def optimize_team_multi(
    data_path: str,
    budget: int = 500,
) -> list[dict]:
    """
    Genera 5 rose ottimali con strategie diverse e ritorna una lista di dict con:
      team, strategy, label, tagline, costo_totale, punti_previsti,
      spesa_per_ruolo, pro, contro.

    Le squadre vengono generate con train_prediction_model (con cache) per ogni
    strategia, quindi sono realmente diverse nella scelta dei giocatori.
    """
    results: list[dict] = []

    for strategy, label, tagline in _FIVE_STRATEGIES:
        try:
            from predict import train_prediction_model
            import tempfile

            df = train_prediction_model(data_path, strategy=strategy)

            with tempfile.NamedTemporaryFile(
                suffix='.csv', delete=False,
                dir=os.path.dirname(data_path),
            ) as f:
                df.to_csv(f.name, index=False)
                temp_path = f.name

            try:
                team = optimize_team(temp_path, budget=budget, strategy=strategy)
            finally:
                os.unlink(temp_path)

        except Exception as e:
            logger.warning(f"Strategia '{strategy}' fallita: {e}")
            continue

        titolari = _team_titolari(team)
        score_col = next(
            (c for c in ('previsione_ia',) if c in titolari.columns),
            None,
        )
        punti = round(titolari[score_col].sum(), 1) if score_col else 0.0
        costo = int(team['costo_iniziale'].sum())
        spesa = {
            r: int(team[team['ruolo'] == r]['costo_iniziale'].sum())
            for r in ['P', 'D', 'C', 'A']
        }
        pro, contro = _team_pros_cons(team, strategy, punti, spesa, budget)

        results.append({
            'team':            team,
            'strategy':        strategy,
            'label':           label,
            'tagline':         tagline,
            'budget':          budget,
            'costo_totale':    costo,
            'punti_previsti':  punti,
            'spesa_per_ruolo': spesa,
            'pro':             pro,
            'contro':          contro,
        })

    return results


# ── CLI di test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    base     = os.path.dirname(os.path.abspath(__file__))
    test_csv = os.path.join(base, "data", "serie_a_23_24_backtest.csv")
    strat    = sys.argv[1] if len(sys.argv) > 1 else "aggressiva"
    bdg      = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    print(f"\nTest: strategia={strat}, budget={bdg}")
    print(f"Dataset: {test_csv}\n")

    team = optimize_team(test_csv, budget=bdg, strategy=strat)

    for ruolo in ['P', 'D', 'C', 'A']:
        sub = team[team['ruolo'] == ruolo].sort_values('costo_iniziale', ascending=False)
        print(f"\n── {ruolo} ({len(sub)}) ──")
        for _, row in sub.iterrows():
            fm  = f"FM={row['fanta_media']:.1f}" if 'fanta_media' in row and row['fanta_media'] > 0 else ""
            sc  = f"score={row.get('score_selezione', row.get('previsione_ia', 0)):.2f}"
            print(f"  {row['nome']:22s}  {int(row['costo_iniziale']):3d}cr  {sc}  {fm}")

    print(f"\nCosto totale: {int(team['costo_iniziale'].sum())}/{bdg} cr.")
    print(f"Spesa per reparto:")
    for ruolo in ['P', 'D', 'C', 'A']:
        spesa = int(team[team['ruolo'] == ruolo]['costo_iniziale'].sum())
        print(f"  {ruolo}: {spesa}cr ({round(spesa/bdg*100,1)}%)")
