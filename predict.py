import os
import json
import math
import hashlib
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False

logger = logging.getLogger(__name__)

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "data")
VIP_CONFIG_PATH  = os.path.join(DATA_DIR, "vip_config.json")
MODEL_CACHE_PATH = os.path.join(DATA_DIR, "model_cache.pkl")

CURRENT_YEAR    = 2026
MAX_VIP_BOOST   =  0.90
MIN_VIP_PENALTY = -0.20

# ── Defaults per ruolo ────────────────────────────────────────────────────────

_ROLE_DEFAULTS = {
    'gol_pg':          {'P': 0.00, 'D': 0.04, 'C': 0.12, 'A': 0.35},
    'assist_pg':       {'P': 0.00, 'D': 0.06, 'C': 0.15, 'A': 0.15},
    'ammonizioni_pg':  {'P': 0.05, 'D': 0.18, 'C': 0.15, 'A': 0.10},
    'xg_pg':           {'P': 0.00, 'D': 0.03, 'C': 0.10, 'A': 0.30},
    'titolarita_pct':  {'P': 0.75, 'D': 0.70, 'C': 0.70, 'A': 0.65},
    'presenze':        {'P': 25.0, 'D': 22.0, 'C': 22.0, 'A': 20.0},
    'gol_subiti_pg':   {'P': 1.20, 'D': 0.0,  'C': 0.0,  'A': 0.0},
    'clean_sheet_pg':  {'P': 0.30, 'D': 0.0,  'C': 0.0,  'A': 0.0},
    'cambio_squadra':  {'P': 0,    'D': 0,     'C': 0,    'A': 0},
    'upgrade_squadra': {'P': 0,    'D': 0,     'C': 0,    'A': 0},
    'tipo_prestito':   {'P': 0,    'D': 0,     'C': 0,    'A': 0},
    'coach_bonus':     {'P': 0.0,  'D': 0.0,   'C': 0.0,  'A': 0.0},
}

STAT_COLS = [
    'gol_pg', 'assist_pg', 'ammonizioni_pg', 'xg_pg', 'titolarita_pct',
    'presenze', 'gol_subiti_pg', 'clean_sheet_pg',
]
TRANSFER_COLS = ['cambio_squadra', 'upgrade_squadra', 'tipo_prestito', 'coach_bonus']


# ── Model cache ──────────────────────────────────────────────────────────────

def _data_hash(data_path: str) -> str:
    h = hashlib.md5()
    for path in [
        data_path,
        os.path.join(DATA_DIR, "fantacalcio_stats_cache.csv"),
        os.path.join(DATA_DIR, "injuries_cache.json"),
        os.path.join(DATA_DIR, "transfers_cache.csv"),
        VIP_CONFIG_PATH,
    ]:
        if os.path.exists(path):
            h.update(str(os.path.getmtime(path)).encode())
    return h.hexdigest()


def _load_model_cache(data_path: str):
    if not _JOBLIB_OK or not os.path.exists(MODEL_CACHE_PATH):
        return None
    try:
        cache = joblib.load(MODEL_CACHE_PATH)
        if cache.get('hash') == _data_hash(data_path):
            logger.info("Modello caricato dalla cache.")
            return cache.get('models')
    except Exception:
        pass
    return None


def _save_model_cache(models: dict, data_path: str) -> None:
    if not _JOBLIB_OK:
        return
    try:
        joblib.dump({'models': models, 'hash': _data_hash(data_path)}, MODEL_CACHE_PATH)
        logger.info("Modello salvato in cache.")
    except Exception as e:
        logger.warning(f"Impossibile salvare cache modello: {e}")


def invalidate_model_cache() -> None:
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            os.unlink(MODEL_CACHE_PATH)
        except Exception:
            pass


# ── Età e curva di carriera ──────────────────────────────────────────────────

def _load_age_data() -> tuple[dict, dict]:
    """Carica youth_players e age_overrides da vip_config.json."""
    try:
        if os.path.exists(VIP_CONFIG_PATH):
            with open(VIP_CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
            youth = {k: v for k, v in cfg.get("youth_players", {}).items()
                     if not k.startswith("_")}
            overrides = {k: v for k, v in cfg.get("age_overrides", {}).items()
                         if not k.startswith("_")}
            return youth, overrides
    except Exception:
        pass
    return {}, {}


def _match_name(nome: str, key: str) -> bool:
    """Match per nome completo o per cognome (ultima parola)."""
    n, k = nome.lower().strip(), key.lower().strip()
    if n == k:
        return True
    cognome_n = n.split()[-1] if n else ""
    cognome_k = k.split()[-1] if k else ""
    return bool(cognome_n) and cognome_n == cognome_k


def _estimate_age(nome: str, ruolo: str, costo: float,
                  youth: dict, overrides: dict) -> float:
    """
    Stima l'età del giocatore.
    Priorità: age_overrides > youth_players > euristica per prezzo.
    """
    nome_clean = str(nome).strip()

    # 1. Override manuale esplicito
    for key, age_val in overrides.items():
        if _match_name(nome_clean, key):
            return float(age_val)

    # 2. Da youth_players (birth_year noto)
    for key, data in youth.items():
        if _match_name(nome_clean, key):
            return float(CURRENT_YEAR - int(data.get("birth_year", CURRENT_YEAR - 25)))

    # 3. Euristica: portieri più vecchi in media, campo per prezzo
    if ruolo == 'P':
        return 30.0  # i GK titolari in Serie A hanno mediamente 28-32 anni

    # Per i giocatori di campo: costo alto = spesso giocatore nel pieno (25-29)
    # costo basso = potrebbe essere giovane non ancora prezzato o veterano economico
    # Usiamo 26 come default (quasi neutro sulla curva di carriera)
    return 26.0


def _prime_factor(age: float) -> float:
    """
    Fattore di carriera: curva a campana con picco a 27 anni.
    - età 22: 0.73 (ancora in crescita)
    - età 27: 1.00 (picco)
    - età 30: 0.87 (leggero calo)
    - età 33: 0.64 (calo significativo)
    - età 36: 0.41 (fine carriera)
    """
    return math.exp(-0.5 * ((age - 27.0) / 4.5) ** 2)


# ── Forma recente ────────────────────────────────────────────────────────────

def _load_forma_recente() -> dict:
    try:
        if os.path.exists(VIP_CONFIG_PATH):
            with open(VIP_CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
            return {k: v for k, v in cfg.get("forma_recente", {}).items()
                    if not k.startswith("_")}
    except Exception:
        pass
    return {}


def _compute_forma_score(nome: str, fanta_media_stagionale: float,
                          forma_recente: dict) -> float:
    nome_clean = str(nome).strip()
    voti = forma_recente.get(nome_clean)
    if voti is None:
        cognome = nome_clean.split()[-1].lower()
        for k, v in forma_recente.items():
            if cognome and cognome == k.split()[-1].lower():
                voti = v
                break
    if not voti:
        return 0.0
    media = sum(float(v) for v in voti) / len(voti)
    base  = float(fanta_media_stagionale) if fanta_media_stagionale and float(fanta_media_stagionale) > 0 else 6.0
    return round(media - base, 3)


def save_forma_recente(nome: str, voti: list[float]) -> bool:
    try:
        cfg = {}
        if os.path.exists(VIP_CONFIG_PATH):
            with open(VIP_CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
        cfg.setdefault("forma_recente", {})[nome] = [round(float(v), 2) for v in voti]
        with open(VIP_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        invalidate_model_cache()
        return True
    except Exception as e:
        logger.error(f"Errore salvataggio forma_recente: {e}")
        return False


# ── Feature engineering ──────────────────────────────────────────────────────

def _fill_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    for col in STAT_COLS + TRANSFER_COLS:
        default_map = _ROLE_DEFAULTS.get(col, {})
        if col not in df.columns:
            df[col] = df['ruolo'].map(default_map).fillna(0)
        else:
            if col in STAT_COLS and col not in ('gol_subiti_pg', 'clean_sheet_pg', 'presenze'):
                mask = df[col].isna() | (df[col] == 0)
            else:
                mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = df.loc[mask, 'ruolo'].map(default_map).fillna(0)
    return df


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    costo_max = df['costo_iniziale'].max() or 1
    df['qualita_mercato'] = df['costo_iniziale'] / costo_max

    # ── Età e curva di carriera ───────────────────────────────────────────────
    youth, overrides = _load_age_data()
    df['eta'] = df.apply(
        lambda r: _estimate_age(r.get('nome', ''), r.get('ruolo', 'C'),
                                float(r.get('costo_iniziale', 5)), youth, overrides),
        axis=1,
    )
    df['prime_factor'] = df['eta'].apply(_prime_factor)

    # ── Continuità (presenze normalizzate su 38 giornate) ────────────────────
    df['presenze_pct'] = (df['presenze'] / 38.0).clip(0, 1)

    # ── GK: score difensivo basato su clean sheet ─────────────────────────────
    gk_mask = df['ruolo'] == 'P'
    df['gk_defense_score'] = 0.0
    if gk_mask.any():
        df.loc[gk_mask, 'gk_defense_score'] = (
            df.loc[gk_mask, 'clean_sheet_pg'] * 3.0
            - df.loc[gk_mask, 'gol_subiti_pg'].clip(0, 4) * 0.5
        )

    # ── Bonus offensivo outfield ──────────────────────────────────────────────
    df['bonus_potential'] = (
        df['gol_pg'] * 3.0
        + df['assist_pg'] * 1.5
        + df['xg_pg'] * 1.0
        - df['ammonizioni_pg'] * 0.5
    )
    df.loc[gk_mask, 'bonus_potential'] = df.loc[gk_mask, 'gk_defense_score']

    coach_bonus = df['coach_bonus'] if 'coach_bonus' in df.columns else pd.Series(0.0, index=df.index)
    df['transfer_effect'] = (
        df['upgrade_squadra'] * 0.60          # peso raddoppiato: club migliore = più FM
        + coach_bonus * 0.40                   # bonus tattico allenatore/sistema di gioco
        - df['cambio_squadra'] * 0.10          # penalità adattamento (primo anno in nuova squadra)
        - df['tipo_prestito'] * 0.20           # prestito = incertezza sulla continuità
    )
    df['disponibilita'] = df['titolarita_pct'].clip(0, 1)

    # ── Forma recente ─────────────────────────────────────────────────────────
    forma = _load_forma_recente()
    df['forma_recente_score'] = df.apply(
        lambda r: _compute_forma_score(r.get('nome', ''), r.get('fanta_media', 6.0), forma),
        axis=1,
    )

    return df


# ── Modelli per ruolo ─────────────────────────────────────────────────────────
#
# Ogni ruolo usa le feature più rilevanti per la sua meccanica di scoring.
# I portieri non usano feature offensive; gli attaccanti non usano gk_defense_score.
# L'età e la continuità sono feature universali perché predicono il futuro,
# non ricostruiscono il passato.

ROLE_FEATURE_COLS: dict[str, list[str]] = {
    'P': [
        'costo_iniziale', 'qualita_mercato',
        'gk_defense_score', 'clean_sheet_pg', 'gol_subiti_pg',
        'ammonizioni_pg', 'titolarita_pct', 'disponibilita',
        'eta', 'prime_factor', 'presenze_pct',
        'transfer_effect', 'forma_recente_score',
    ],
    'D': [
        'costo_iniziale', 'qualita_mercato',
        'gol_pg', 'assist_pg', 'xg_pg', 'ammonizioni_pg',
        'titolarita_pct', 'bonus_potential', 'disponibilita',
        'eta', 'prime_factor', 'presenze_pct',
        'cambio_squadra', 'upgrade_squadra', 'tipo_prestito', 'coach_bonus', 'transfer_effect',
        'forma_recente_score',
    ],
    'C': [
        'costo_iniziale', 'qualita_mercato',
        'gol_pg', 'assist_pg', 'xg_pg', 'ammonizioni_pg',
        'titolarita_pct', 'bonus_potential', 'disponibilita',
        'eta', 'prime_factor', 'presenze_pct',
        'cambio_squadra', 'upgrade_squadra', 'tipo_prestito', 'coach_bonus', 'transfer_effect',
        'forma_recente_score',
    ],
    'A': [
        'costo_iniziale', 'qualita_mercato',
        'gol_pg', 'assist_pg', 'xg_pg', 'ammonizioni_pg',
        'titolarita_pct', 'bonus_potential', 'disponibilita',
        'eta', 'prime_factor', 'presenze_pct',
        'cambio_squadra', 'upgrade_squadra', 'tipo_prestito', 'coach_bonus', 'transfer_effect',
        'forma_recente_score',
    ],
}

# Feature cols per compatibilità con codice che usa FEATURE_COLS (backtest, ecc.)
FEATURE_COLS = sorted({col for cols in ROLE_FEATURE_COLS.values() for col in cols})


def _train_role_model(X: pd.DataFrame, y: pd.Series, ruolo: str):
    """
    Modello adattivo:
    - < 8 campioni:  DummyRegressor (media di ruolo)
    - 8–30 campioni: Ridge regression (lineare regolarizzata, non overfittando)
    - > 30 campioni: GradientBoosting (non-lineare, sfrutta correlazioni complesse)

    Separare i ruoli impedisce al modello di confondere
    'gol_pg alto per un difensore' con 'gol_pg alto per un attaccante'.
    """
    n = len(X)

    if n < 8:
        from sklearn.dummy import DummyRegressor
        m = DummyRegressor(strategy='mean')
        m.fit(X, y)
        return m

    if n < 30:
        # Dataset piccolo: modello lineare regolarizzato
        # Ridge con alpha alto = shrinkage forte = meno overfitting
        m = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=3.0)),
        ])
    else:
        # Dataset sufficiente: GBR con iperparametri conservativi per ruolo
        n_est = min(200, max(50, n * 2))
        depth = 3 if ruolo == 'P' else 4
        min_leaf = max(2, n // 25)
        m = GradientBoostingRegressor(
            n_estimators=n_est,
            learning_rate=0.05,
            max_depth=depth,
            min_samples_leaf=min_leaf,
            subsample=0.8,
            random_state=42,
        )

    m.fit(X, y)
    return m


def _train_all_models(df: pd.DataFrame, y: pd.Series) -> dict:
    """Addestra un modello per ognuno dei 4 ruoli."""
    models = {}
    for ruolo in ['P', 'D', 'C', 'A']:
        mask = df['ruolo'] == ruolo
        if mask.sum() < 3:
            logger.warning(f"Troppo pochi campioni per ruolo {ruolo} ({mask.sum()}), skip")
            continue
        feat_cols = [c for c in ROLE_FEATURE_COLS[ruolo] if c in df.columns]
        X_r = df.loc[mask, feat_cols].fillna(0)
        y_r = y[mask] if isinstance(y, pd.Series) else y.loc[mask]
        models[ruolo] = (feat_cols, _train_role_model(X_r, y_r, ruolo))
        logger.info(f"Modello {ruolo}: {type(models[ruolo][1]).__name__} su {mask.sum()} campioni")
    return models


def _predict_all_roles(df: pd.DataFrame, models: dict) -> pd.Series:
    """Applica i 4 modelli e restituisce la serie previsione_ia."""
    pred = pd.Series(6.0, index=df.index)
    for ruolo, (feat_cols, model) in models.items():
        mask = df['ruolo'] == ruolo
        if not mask.any():
            continue
        X_r = df.loc[mask, feat_cols].fillna(0)
        pred.loc[mask] = model.predict(X_r).clip(3.0, 10.0)
    return pred


# ── Regressione alla media ────────────────────────────────────────────────────

def _apply_regression_to_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bayesian shrinkage: sposta la previsione verso la media del ruolo in modo
    proporzionale all'incertezza (poche presenze = più shrinkage).

    Principio: chi ha giocato 5 partite e ha segnato tanto probabilmente non
    ha una FM di 9.5 — potrebbe essere un picco di forma. Chi ha giocato 35
    partite con FM 9.5 è più credibile.

    Formula: pred_corr = (1-α) × pred + α × media_ruolo
    dove α = 0.30 × (1 - presenze_pct)
      → 5 presenze  → α ≈ 0.27  (27% shrinkage verso media)
      → 20 presenze → α ≈ 0.14  (14% shrinkage)
      → 35 presenze → α ≈ 0.03  (3% shrinkage, quasi nessuno)
    """
    df = df.copy()
    role_mean = df.groupby('ruolo')['previsione_ia'].transform('mean')

    presenze_pct = df.get('presenze_pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    alpha = (0.30 * (1.0 - presenze_pct)).clip(0.03, 0.30)

    df['previsione_ia'] = (
        (1.0 - alpha) * df['previsione_ia'] + alpha * role_mean
    ).clip(3.0, 10.0).round(3)

    return df


# ── Strategia ────────────────────────────────────────────────────────────────

def _apply_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Calcola 'score_selezione' usato dall'ottimizzatore.
    NON modifica 'previsione_ia' (stima FM realistica con regressione alla media).
    """
    s = strategy.lower()
    df = df.copy()
    df['score_selezione'] = df['previsione_ia'].copy()

    gol_bonus_ruolo = {'P': 0.0, 'D': 4.0, 'C': 3.0, 'A': 3.0}

    for ruolo in df['ruolo'].unique():
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
                df.loc[mask, 'score_selezione'] = prev

    # Garantisce che score_selezione non contenga mai NaN
    df['score_selezione'] = df['score_selezione'].fillna(df['previsione_ia']).fillna(0.0)

    # ── Penalità presenze: penalizza i giocatori con storia di infortuni ────────
    if 'presenze' in df.columns and df['presenze'].median() > 5:
        _soglie = {'moneyball': 10, 'agg': 12, 'sprint': 12, 'master': 16}
        soglia = next((v for k, v in _soglie.items() if k in s), 16)
        mask_low = df['presenze'] < soglia
        if mask_low.any():
            # fattore lineare: 0 presenze → ×0.40, a soglia → ×1.00
            factor = ((df.loc[mask_low, 'presenze'] / soglia) * 0.60 + 0.40).clip(0.40, 1.0)
            df.loc[mask_low, 'score_selezione'] = (
                df.loc[mask_low, 'score_selezione'] * factor
            ).clip(3.0, 14.0)

    # ── Boost forma recente ────────────────────────────────────────────────────
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
        df['score_selezione'] = (df['score_selezione'] * (1 + vip_boost)).clip(3.0, 14.0)

    df['score_selezione'] = df['score_selezione'].fillna(df['previsione_ia']).fillna(0.0)
    return df


# ── Pipeline principale ───────────────────────────────────────────────────────

def train_prediction_model(
    data_path: str,
    stats_season: str = "current",
    strategy: str = "conservativa",
    use_temporal: bool = True,
) -> pd.DataFrame:
    """
    Pipeline completo:
      1. Carica quotazioni
      2. Arricchisce con stats (Fantacalcio.it), trasferimenti, infortuni
      3. Feature engineering (età, continuità, GK defense score, ecc.)
      4. Predizione (temporale se disponibile, altrimenti same-season)
      5. Applica regressione alla media (Bayesian shrinkage)
      6. VIP + Strategia + Infortuni

    Ordine di priorità per la predizione:
      - Modello temporale (features stagione N → FM stagione N+1): se use_temporal=True
        e i modelli sono stati addestrati con /allena, produce previsioni causalmente corrette.
      - Same-season fallback: addestra su fanta_media della stessa stagione delle features
        (circolare, ma utile prima che il modello temporale esista).
    """
    df = pd.read_csv(data_path)
    
    # Salva le stats "reali" caricate dall'utente prima che gli scraper le sovrascrivano.
    # Questo è FONDAMENTALE se il campionato è già iniziato (es. prime 3 giornate).
    original_fm = df['fanta_media'].copy() if 'fanta_media' in df.columns else None
    original_pv = df['presenze'].copy() if 'presenze' in df.columns else None

    # Fantacalcio.it: solo per la stagione corrente (non ha dati storici per stagione)
    if stats_season == "current":
        try:
            from scraper_fantacalcio import merge_stats_with_quotazioni
            df = merge_stats_with_quotazioni(df)
        except Exception:
            pass

    # FBref: season-aware, funziona sia per current che per stagioni storiche (backtest)
    try:
        from scraper_stats import merge_stats_for_season as _fbref_merge
        df = _fbref_merge(df, stats_season)
    except Exception:
        pass

    try:
        from scraper_transfermarkt import merge_transfers_with_quotazioni
        df = merge_transfers_with_quotazioni(df)
    except Exception:
        pass

    df = _fill_missing_stats(df)
    df = _build_features(df)

    # ── Predizione ─────────────────────────────────────────────────────────────
    temporal_used = False

    if use_temporal:
        try:
            from temporal_model import temporal_models_exist, load_temporal_models, predict_with_temporal
            if temporal_models_exist():
                t_models, _, _ = load_temporal_models()
                if t_models:
                    df['previsione_ia'] = predict_with_temporal(df, t_models)
                    temporal_used = True
                    logger.info("✓ Modello temporale attivo (previsione causale stagione N+1).")
        except Exception as e:
            logger.warning(f"Modello temporale non disponibile: {e}")

    if not temporal_used:
        # Fallback same-season: circolare ma utile in assenza del temporale
        if 'fanta_media' in df.columns and df['fanta_media'].std() > 0.5:
            y = df['fanta_media']
        else:
            y = (5.0
                 + df['qualita_mercato'] * 3.0
                 + df['bonus_potential'].clip(-1, 3) * 0.5
                 + df['transfer_effect'])

        models = _load_model_cache(data_path)
        if models is None:
            logger.info("Addestramento modelli same-season in corso...")
            models = _train_all_models(df, y)
            _save_model_cache(models, data_path)

        df['previsione_ia'] = _predict_all_roles(df, models)
        if not temporal_used:
            logger.info("ℹ️  Usando modello same-season. Esegui /allena per il modello temporale.")

    df['_temporal_model'] = temporal_used

    # Regressione alla media: riduce le previsioni estreme per chi ha poche presenze
    df = _apply_regression_to_mean(df)

    try:
        from vip import compute_vip
        df = compute_vip(df)
    except Exception as e:
        logger.warning(f"VIP non applicato: {e}")

    # Arricchimento Nuovi Arrivi Esteri + Rigoristi e Punizionisti
    try:
        from foreign_league import enrich_dataset_with_foreign_arrivals
        df = enrich_dataset_with_foreign_arrivals(df)
    except Exception as e:
        logger.warning(f"Foreign league enrichment non applicato: {e}")

    try:
        from set_pieces import apply_set_pieces_boost
        df = apply_set_pieces_boost(df)
    except Exception as e:
        logger.warning(f"Set pieces boost non applicato: {e}")

    # ── BLENDING FORMA INIZIO CAMPIONATO (Se a Campionato Iniziato) ─────────────
    # Se l'utente ha caricato le "Statistiche" con voti veri delle prime N giornate
    # incrociamo la pre-season prediction con la forma reale attuale.
    if original_fm is not None and original_pv is not None:
        mask_played = original_pv > 0
        if mask_played.sum() > 30 and original_fm.std() > 0.5:
            logger.info(f"⚽ Campionato iniziato rilevato ({mask_played.sum()} giocatori a voto). Applico blending forma attuale.")
            
            # Peso della forma reale rispetto all'IA basato sulle partite giocate.
            # Es. 1 partita = 15% forma, 85% IA. 3 partite = 45% forma, 55% IA.
            alpha = (original_pv * 0.15).clip(0.0, 0.60) 
            
            df.loc[mask_played, 'previsione_ia'] = (
                (1.0 - alpha[mask_played]) * df.loc[mask_played, 'previsione_ia'] +
                alpha[mask_played] * original_fm[mask_played]
            ).round(3)
            
            # Penalizza leggermente (es. -0.25) i giocatori teoricamente titolari che non hanno giocato
            mask_benched = (original_pv == 0) & (df.get('titolarita_pct', 1.0) > 0.6)
            df.loc[mask_benched, 'previsione_ia'] = (df.loc[mask_benched, 'previsione_ia'] - 0.25).clip(lower=3.0)

    # ── STRATEGIA, CALENDARIO E NEWS ───────────────────────────────────────────
    df = _apply_strategy(df, strategy)
    try:
        from scraper_injuries import apply_injury_modifiers
        df = apply_injury_modifiers(df)
    except Exception:
        pass

    try:
        news_file = os.path.join(os.path.dirname(data_path), "news_latest.txt")
        if os.path.exists(news_file):
            with open(news_file, "r", encoding="utf-8") as f:
                news_txt = f.read()
            if news_txt.strip():
                from patch_app import apply_news_modifiers
                df = apply_news_modifiers(df, news_txt)
                logger.info("✓ Notizie automatiche applicate con successo.")
    except Exception as e:
        logger.warning(f"Applicazione news automatiche fallita: {e}")

    try:
        from calendar_analyzer import apply_calendar_modifiers
        df = apply_calendar_modifiers(df, strategy=strategy)
    except Exception as e:
        logger.warning(f"Calendario non applicato: {e}")

    # Garantire che fanta_media contenga la stima se priva di storico nel listone di inizio stagione
    if 'fanta_media' not in df.columns or df['fanta_media'].isna().all():
        df['fanta_media'] = df['previsione_ia']
    else:
        df['fanta_media'] = df['fanta_media'].fillna(df['previsione_ia'])
        mask_zero = df['fanta_media'] == 0
        df.loc[mask_zero, 'fanta_media'] = df.loc[mask_zero, 'previsione_ia']

    return df

# ── Spiegazione per giocatore ─────────────────────────────────────────────────

_COACH_STYLE_DESC = {
    'pressing':     'pressing alto (favorisce esterni, mezzali e difensori offensivi)',
    'possesso':     'gioco di possesso (favorisce tecnici e centrocampisti creativi)',
    'contropiede':  'contropiede (favorisce attaccanti veloci e trequartisti)',
    'equilibrato':  'sistema equilibrato',
}

_INJURY_TYPE_DESC = {
    'infortunio_grave': 'infortunio grave (stop oltre 2 mesi)',
    'infortunio':       'infortunio in corso',
    'squalifica':       'squalificato',
    'panchina':         'fuori rosa / non convocato',
    'recupero':         'in recupero / rientro imminente',
    'titolare':         'confermato titolare',
    'rigorista':        'designato rigorista',
    'cambio_allenatore':'cambio allenatore in corso',
}


def build_compact_reason(row: pd.Series) -> str:
    """Una riga che riassume il motivo principale della scelta per quel giocatore."""
    parts: list[str] = []
    ruolo = str(row.get('ruolo', 'C'))

    if ruolo == 'P':
        cs = float(row.get('clean_sheet_pg', 0) or 0)
        if cs > 0.30:
            parts.append(f"CS {cs:.0%}")
    else:
        gol = float(row.get('gol_pg', 0) or 0)
        ast = float(row.get('assist_pg', 0) or 0)
        if gol > 0.15:
            parts.append(f"gol/g {gol:.2f}")
        if ast > 0.12:
            parts.append(f"ass/g {ast:.2f}")

    if int(row.get('cambio_squadra', 0) or 0) == 1:
        upgrade = float(row.get('upgrade_squadra', 0) or 0)
        style = str(row.get('coach_style', '') or '')
        if upgrade > 0:
            parts.append("upgrade club")
        elif upgrade < 0:
            parts.append("downgrade club")
        if style and style not in ('equilibrato', ''):
            parts.append(f"tattica {style}")

    eta = float(row.get('eta', 26) or 26)
    if eta <= 22:
        parts.append(f"prospetto {eta:.0f}a")
    elif eta >= 33:
        parts.append(f"veterano {eta:.0f}a")

    forma = float(row.get('forma_recente_score', 0) or 0)
    if forma >= 0.5:
        parts.append("in forma")
    elif forma <= -0.5:
        parts.append("fuori forma")

    inj = float(row.get('injury_modifier', 1.0) or 1.0)
    if inj < 0.90:
        parts.append(f"infort. ×{inj:.2f}")
    elif inj > 1.05:
        parts.append(f"boost ×{inj:.2f}")

    if 'calendar_diff' in row:
        cal_diff = float(row.get('calendar_diff', 3.0))
        if cal_diff < 2.7:
            parts.append("calendario facile")
        elif cal_diff > 3.3:
            parts.append("calendario ostico")

    return " · ".join(parts)


def build_player_explanation(row: pd.Series) -> str:
    """
    Spiegazione passo-passo di come l'IA ha stimato la FM per un giocatore.
    Genera un testo fluido tipo "Scout Report" basato sulle feature del modello.
    """
    ruolo = str(row.get('ruolo', 'C'))
    nome = str(row.get('nome', ''))
    costo = int(row.get('costo_iniziale', 0) or 0)
    prev = float(row.get('previsione_ia', 0) or 0)
    fm_st = row.get('fanta_media')
    
    lines = [f"🔍 **Scout Report IA: {nome}** [{ruolo}] — Quotazione: {costo} cr.\n"]
    
    # 1. Base e Anagrafica
    eta = float(row.get('eta', 26) or 26)
    if eta <= 22:
        stage = "è un prospetto in forte rampa di lancio"
    elif eta <= 26:
        stage = "è in fase di ascesa verso il prime fisico"
    elif eta <= 29:
        stage = "si trova nel pieno della maturità calcistica"
    elif eta <= 32:
        stage = "è un giocatore esperto ma con un lieve calo fisiologico atteso"
    else:
        stage = "è nella fase finale della carriera, con un minutaggio potenzialmente ridotto"
        
    intro = f"Il modello predittivo ha analizzato il profilo di {nome}. A {eta:.0f} anni, il giocatore {stage}."
    if fm_st and float(fm_st) > 0:
        intro += f" Parte da una fantamedia storica di {float(fm_st):.2f}."
    lines.append(intro)

    # 2. Analisi Statistica
    if ruolo == 'P':
        cs = float(row.get('clean_sheet_pg', 0) or 0)
        gs = float(row.get('gol_subiti_pg', 0) or 0)
        tit = float(row.get('titolarita_pct', 0.7) or 0.7)
        stat_text = f"Tra i pali garantisce una titolarità del {tit:.0%}, con una media di {gs:.2f} gol subiti a partita e una probabilità di clean sheet del {cs:.0%}."
        lines.append(stat_text)
    else:
        gol = float(row.get('gol_pg', 0) or 0)
        ast = float(row.get('assist_pg', 0) or 0)
        tit = float(row.get('titolarita_pct', 0.7) or 0.7)
        if gol > 0.2 or ast > 0.15:
            stat_text = f"Offensivamente i suoi numeri sono ottimi: produce {gol:.2f} gol e {ast:.2f} assist ogni 90 minuti."
        elif gol > 0.05 or ast > 0.05:
            stat_text = f"Il suo contributo offensivo è discreto ({gol:.2f} gol e {ast:.2f} assist a partita)."
        else:
            stat_text = f"Statisticamente non è un portatore di grandi bonus ({gol:.2f} gol a partita)."
        lines.append(stat_text + f" La sua affidabilità in termini di presenze è stimata al {tit:.0%}.")

    # 3. Fattori Esterni (Mercato, Calendario, Infortuni, VIP)
    factors = []
    
    if int(row.get('cambio_squadra', 0) or 0) == 1:
        upgrade = float(row.get('upgrade_squadra', 0) or 0)
        coach_b = float(row.get('coach_bonus', 0) or 0)
        if upgrade > 0:
            factors.append("il trasferimento in una squadra di fascia superiore ne aumenta l'appetibilità")
        elif upgrade < 0:
            factors.append("il passaggio a una squadra di caratura inferiore potrebbe limitarne il potenziale")
        if coach_b > 0:
            factors.append("il sistema tattico del nuovo allenatore esalta particolarmente le sue caratteristiche")
            
    forma = float(row.get('forma_recente_score', 0) or 0)
    if forma > 0.25:
        factors.append("il trend di forma recente è decisamente positivo")
    elif forma < -0.25:
        factors.append("attualmente sta attraversando un periodo di scarsa forma")
        
    vip = float(row.get('vip_total', 0) or 0)
    if vip > 0.05:
        factors.append("i parametri tattici avanzati (VIP) indicano un potenziale hidden bonus")
        
    if 'calendar_diff' in row:
        cal_diff = float(row.get('calendar_diff', 3.0))
        if cal_diff < 2.7:
            factors.append("il calendario iniziale favorevole della sua squadra rappresenta un vantaggio")
        elif cal_diff > 3.3:
            factors.append("dovrà affrontare un calendario iniziale ostico che ne abbassa la proiezione")
            
    inj_mod = float(row.get('injury_modifier', 1.0) or 1.0)
    if inj_mod < 0.95:
        inj_type = str(row.get('injury_type', 'un problema fisico'))
        tipo_label = _INJURY_TYPE_DESC.get(inj_type, inj_type)
        factors.append(f"pesa negativamente la situazione medica attuale ({tipo_label})")
        
    if factors:
        lines.append("A livello di contesto, l'algoritmo rileva che " + ", ".join(factors[:-1]) + (" e " if len(factors)>1 else "") + factors[-1] + ".")

    # 4. Conclusione
    temporal = bool(row.get('_temporal_model', False))
    model_tag = "Modello Causale Temporale" if temporal else "Modello Standard (Same-Season)"
    
    lines.append(f"\n🤖 **Sintesi e Previsione Finale**")
    lines.append(f"Integrando tutti questi parametri matematici, il _{model_tag}_ ha ricalibrato il valore del giocatore, fissando la fantamedia attesa a **{prev:.2f}**.")
    
    return "\n".join(lines)


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, 'data', 'serie_a_23_24_backtest.csv')

    print("\n--- ANALISI IA (modelli separati per ruolo) ---")
    results = train_prediction_model(path)

    print("\nTop 10 Qualità/Prezzo:")
    results['convenienza'] = results['previsione_ia'] / (results['costo_iniziale'] + 1)
    for _, row in results.nlargest(10, 'convenienza').iterrows():
        eta = row.get('eta', 0)
        pf  = row.get('prime_factor', 1.0)
        print(
            f"  {row['nome']:22s} {row['ruolo']} "
            f"| FM: {row['previsione_ia']:.2f} "
            f"| età: {eta:.0f} (pf={pf:.2f}) "
            f"| costo: {int(row['costo_iniziale'])}"
        )

    print("\nPortieri (con CS probability):")
    for _, row in results[results['ruolo'] == 'P'].sort_values('previsione_ia', ascending=False).iterrows():
        print(
            f"  {row['nome']:22s} "
            f"| FM: {row['previsione_ia']:.2f} "
            f"| CS: {row.get('clean_sheet_pg', 0):.0%} "
            f"| gol_sub: {row.get('gol_subiti_pg', 0):.2f}/g "
            f"| età: {row.get('eta', 0):.0f}"
        )
