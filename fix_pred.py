import re
with open("predict.py", "r") as f:
    content = f.read()

target = """    for ruolo in df['ruolo'].unique():
        mask = df['ruolo'] == ruolo
        sub  = df[mask]
        prev = sub['previsione_ia']
        p_min, p_max = prev.min(), prev.max()
        p_range = (p_max - p_min) if p_max > p_min else 1.0

        def _rescale(series: pd.Series) -> pd.Series:
            mn, mx = series.min(), series.max()
            if mx == mn:
                return pd.Series(p_min + p_range / 2, index=series.index)
            return p_min + (series - mn) / (mx - mn) * p_range

        fm_clean = sub['fanta_media'].fillna(prev) if 'fanta_media' in sub.columns else prev

        if 'conserv' in s:
            if 'fanta_media' not in df.columns or sub['fanta_media'].dropna().empty or sub['fanta_media'].dropna().std() < 0.2:
                continue
            df.loc[mask, 'score_selezione'] = 0.75 * fm_clean + 0.25 * prev

        elif 'agg' in s and 'difesa' not in s:
            if ruolo == 'P':
                continue
            gb  = gol_bonus_ruolo.get(ruolo, 3.0)
            off = (sub['gol_pg'] * gb + sub['assist_pg'] + sub['xg_pg'] * 0.5
                   if sub['gol_pg'].std() >= 0.01
                   else fm_clean)
            df.loc[mask, 'score_selezione'] = 0.4 * prev + 0.6 * _rescale(off)

        elif 'sco' in s or 'hype' in s:
            fm = (fm_clean.clip(lower=4.0)
                  if 'fanta_media' in df.columns and sub['fanta_media'].dropna().size > 5 and sub['fanta_media'].dropna().std() > 0.3
                  else prev)
            costo_clip = sub['costo_iniziale'].clip(lower=1)
            # Rapporto lineare (non sqrt): penalizza molto i giocatori cari
            ratio = fm / costo_clip
            # Bonus esplicito per i giocatori economici (vera "scommessa")
            costo_norm = (costo_clip / costo_clip.max()).clip(0.01, 1.0)
            ratio = ratio + (1.0 - costo_norm) * 0.3
            if 'upgrade_squadra' in df.columns:
                ratio = ratio + sub['upgrade_squadra'].clip(0, 1) * (1.0 - costo_norm) * 0.5
            # Peso prev basso (0.25): la ratio FM/costo guida la scelta
            df.loc[mask, 'score_selezione'] = 0.25 * prev + 0.75 * _rescale(ratio)

        elif 'vip' in s:
            df.loc[mask, 'score_selezione'] = prev

        elif 'difesa' in s:
            if ruolo in ('P', 'D'):
                score_base = (0.80 * fm_clean + 0.20 * prev
                              if 'fanta_media' in df.columns and sub['fanta_media'].dropna().size > 5 and sub['fanta_media'].dropna().std() > 0.2
                              else prev)
                if ruolo == 'P' and 'gk_defense_score' in df.columns:
                    score_base = (score_base + sub['gk_defense_score'].clip(0, 3) * 0.4).clip(3.0, 14.0)
                df.loc[mask, 'score_selezione'] = score_base
            else:
                if 'fanta_media' in df.columns and sub['fanta_media'].dropna().size > 5 and sub['fanta_media'].dropna().std() > 0.2:
                    df.loc[mask, 'score_selezione'] = 0.75 * fm_clean + 0.25 * prev"""

replacement = """    for ruolo in df['ruolo'].unique():
        mask = df['ruolo'] == ruolo
        sub  = df[mask]
        prev = sub['previsione_ia']
        p_min, p_max = prev.min(), prev.max()
        p_range = (p_max - p_min) if p_max > p_min else 1.0

        def _rescale(series: pd.Series) -> pd.Series:
            mn, mx = series.min(), series.max()
            if mx == mn:
                return pd.Series(p_min + p_range / 2, index=series.index)
            return p_min + (series - mn) / (mx - mn) * p_range

        fm_clean = sub['fanta_media'].fillna(prev) if 'fanta_media' in sub.columns else prev

        if 'master' in s:
            df.loc[mask, 'score_selezione'] = prev
            
        elif 'agg' in s:
            if ruolo == 'P':
                continue
            gb  = gol_bonus_ruolo.get(ruolo, 3.0)
            off = (sub['gol_pg'] * gb + sub['assist_pg'] + sub['xg_pg'] * 0.5
                   if sub['gol_pg'].std() >= 0.01
                   else fm_clean)
            df.loc[mask, 'score_selezione'] = 0.4 * prev + 0.6 * _rescale(off)

        elif 'money' in s:
            fm = (fm_clean.clip(lower=4.0)
                  if 'fanta_media' in df.columns and sub['fanta_media'].dropna().size > 5 and sub['fanta_media'].dropna().std() > 0.3
                  else prev)
            costo_clip = sub['costo_iniziale'].clip(lower=1)
            ratio = fm / costo_clip
            costo_norm = (costo_clip / costo_clip.max()).clip(0.01, 1.0)
            ratio = ratio + (1.0 - costo_norm) * 0.3
            if 'upgrade_squadra' in df.columns:
                ratio = ratio + sub['upgrade_squadra'].clip(0, 1) * (1.0 - costo_norm) * 0.5
            df.loc[mask, 'score_selezione'] = 0.25 * prev + 0.75 * _rescale(ratio)
            
        elif 'sprint' in s:
            # Il calendario è già stato applicato in apply_calendar_modifiers con peso x2
            # Qui possiamo aggiungere ulteriore boost sulla forma recente (avvio bruciante)
            if 'forma_recente_score' in df.columns:
                forma_boost = sub['forma_recente_score'] * 0.50
                df.loc[mask, 'score_selezione'] = prev + forma_boost
            else:
                df.loc[mask, 'score_selezione'] = prev"""
content = content.replace(target, replacement)

target2 = """    # ── Penalità presenze: penalizza i giocatori con storia di infortuni ────────
    # Soglia per strategia: scommesse tollera più rischio, conservativa/difesa meno.
    if 'presenze' in df.columns and df['presenze'].median() > 5:
        _soglie = {'scommesse': 10, 'hype': 10, 'agg': 14, 'vip': 14}
        soglia = next((v for k, v in _soglie.items() if k in s), 16)"""

replacement2 = """    # ── Penalità presenze: penalizza i giocatori con storia di infortuni ────────
    if 'presenze' in df.columns and df['presenze'].median() > 5:
        _soglie = {'moneyball': 10, 'agg': 12, 'sprint': 12, 'master': 16}
        soglia = next((v for k, v in _soglie.items() if k in s), 16)"""
content = content.replace(target2, replacement2)

target3 = """    # ── Boost forma recente ────────────────────────────────────────────────────
    if 'forma_recente_score' in df.columns and df['forma_recente_score'].abs().sum() > 0:
        forma_weight = 0.25 if ('agg' in s or 'vip' in s) else 0.10 if 'difesa' in s else 0.15
        df['score_selezione'] = (
            df['score_selezione'] + df['forma_recente_score'] * forma_weight
        ).clip(3.0, 14.0)

    # ── Boost VIP ─────────────────────────────────────────────────────────────
    if 'vip_total' in df.columns:
        _vip_w = {'conserv': 0.20, 'agg': 0.40, 'sco': 0.55,
                  'hype': 0.55, 'vip': 0.70, 'difesa': 0.30}
        key        = next((k for k in _vip_w if k in s), 'conserv')
        vip_boost  = df['vip_total'].clip(MIN_VIP_PENALTY, MAX_VIP_BOOST) * _vip_w[key]
        df['score_selezione'] = (df['score_selezione'] * (1 + vip_boost)).clip(3.0, 14.0)"""

replacement3 = """    # ── Boost forma recente ────────────────────────────────────────────────────
    if 'forma_recente_score' in df.columns and df['forma_recente_score'].abs().sum() > 0 and 'sprint' not in s:
        forma_weight = 0.25 if ('agg' in s or 'money' in s) else 0.15
        df['score_selezione'] = (
            df['score_selezione'] + df['forma_recente_score'] * forma_weight
        ).clip(3.0, 14.0)

    # ── Boost VIP ─────────────────────────────────────────────────────────────
    if 'vip_total' in df.columns:
        _vip_w = {'master': 0.30, 'agg': 0.40, 'sprint': 0.30, 'money': 0.80}
        key        = next((k for k in _vip_w if k in s), 'master')
        vip_boost  = df['vip_total'].clip(MIN_VIP_PENALTY, MAX_VIP_BOOST) * _vip_w[key]
        df['score_selezione'] = (df['score_selezione'] * (1 + vip_boost)).clip(3.0, 14.0)"""
content = content.replace(target3, replacement3)

with open("predict.py", "w") as f:
    f.write(content)
