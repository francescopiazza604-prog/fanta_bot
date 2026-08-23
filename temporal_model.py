"""
Modello temporale: impara da stats(stagione N) → FM(stagione N+1).

Perché è meglio del modello same-season:
  - Il modello attuale usa gli stessi dati sia come feature che come target (circolare).
  - Questo modello usa la stagione PRECEDENTE come features, quella SUCCESSIVA come target.
  - Test reale di predizione: il modello non ha mai visto la stagione da predire.

Come funziona:
  TRAINING (eseguito una volta all'inizio stagione):
    features = FBref stats stagione N   (es. 2022-23)
    target   = FM stagione N+1          (es. 2023-24, dalla colonna fanta_media del file quotazioni)
    → 4 modelli separati per ruolo (P / D / C / A)

  PREDIZIONE (a ogni /ottimizza o /top):
    features = FBref stats stagione N+1 (es. 2023-24, in cache come fbref_stats_cache.csv)
    → applica i modelli → FM stagione N+2 prevista (es. 2024-25)

Matchin cross-stagione: rapidfuzz con score_cutoff=70 per i nomi internazionali FBref.
"""

import os
import logging
import numpy as np
import pandas as pd

try:
    import joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False

logger = logging.getLogger(__name__)

BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
DATA_DIR              = os.path.join(BASE_DIR, "data")
TEMPORAL_MODELS_PATH  = os.path.join(DATA_DIR, "temporal_models.pkl")
TRAINING_DATASET_PATH = os.path.join(DATA_DIR, "temporal_training_dataset.csv")

# Stagione di default per i features di training
# Cambia questa costante se vuoi usare una stagione diversa
DEFAULT_FEATURE_SEASON = "2022-2023"

# Feature usate dal modello temporale per ruolo
# Nota: fanta_media è ESCLUSA — è il target, non può essere una feature
TEMPORAL_FEATURES: dict[str, list[str]] = {
    'P': [
        'costo_iniziale', 'qualita_mercato',
        'gk_defense_score', 'clean_sheet_pg', 'gol_subiti_pg',
        'ammonizioni_pg', 'titolarita_pct',
        'eta', 'prime_factor', 'presenze_pct',
        'transfer_effect',
    ],
    'D': [
        'costo_iniziale', 'qualita_mercato',
        'gol_pg', 'assist_pg', 'xg_pg', 'ammonizioni_pg',
        'bonus_potential', 'titolarita_pct',
        'eta', 'prime_factor', 'presenze_pct',
        'cambio_squadra', 'upgrade_squadra', 'tipo_prestito', 'transfer_effect',
    ],
    'C': [
        'costo_iniziale', 'qualita_mercato',
        'gol_pg', 'assist_pg', 'xg_pg', 'ammonizioni_pg',
        'bonus_potential', 'titolarita_pct',
        'eta', 'prime_factor', 'presenze_pct',
        'cambio_squadra', 'upgrade_squadra', 'tipo_prestito', 'transfer_effect',
    ],
    'A': [
        'costo_iniziale', 'qualita_mercato',
        'gol_pg', 'assist_pg', 'xg_pg', 'ammonizioni_pg',
        'bonus_potential', 'titolarita_pct',
        'eta', 'prime_factor', 'presenze_pct',
        'cambio_squadra', 'upgrade_squadra', 'tipo_prestito', 'transfer_effect',
    ],
}


# ── Match cross-stagione ──────────────────────────────────────────────────────

def _fuzzy_match(nome: str, candidates: list[str], cutoff: int = 70) -> str | None:
    """
    Cerca il miglior match di `nome` nella lista `candidates` usando rapidfuzz.
    Fallback a match per cognome se rapidfuzz non è disponibile.
    """
    try:
        from rapidfuzz import process, fuzz
        result = process.extractOne(
            str(nome).lower(),
            [c.lower() for c in candidates],
            scorer=fuzz.token_set_ratio,
            score_cutoff=cutoff,
        )
        if result:
            idx = [c.lower() for c in candidates].index(result[0])
            return candidates[idx]
    except ImportError:
        pass

    # Fallback: match per cognome (ultima parola)
    cognome = str(nome).lower().split()[-1]
    for cand in candidates:
        if cognome in cand.lower() or cand.lower().split()[-1] == cognome:
            return cand
    return None


def _link_by_name(
    df_features: pd.DataFrame,   # FBref stats (nome_fbref o nome)
    df_target: pd.DataFrame,     # quotazioni con fanta_media
    min_score: int = 70,
) -> pd.DataFrame:
    """
    Collega i due DataFrame per nome giocatore.
    Ritorna un DataFrame con feature + target per ogni giocatore matchato.
    """
    feat_col   = 'nome_fbref' if 'nome_fbref' in df_features.columns else 'nome'
    target_col = 'nome'

    feat_names   = df_features[feat_col].tolist()
    target_names = df_target[target_col].tolist()

    rows = []
    matched = 0

    for _, feat_row in df_features.iterrows():
        feat_nome = str(feat_row[feat_col])
        match = _fuzzy_match(feat_nome, target_names, min_score)
        if match is None:
            continue

        target_row = df_target[df_target[target_col] == match]
        if target_row.empty:
            continue

        target_row = target_row.iloc[0]
        combined   = feat_row.to_dict()
        combined['nome'] = match

        # Target: FM reale dalla stagione successiva
        if 'fanta_media' in target_row:
            combined['fanta_media_next'] = float(target_row['fanta_media'])
        # Aggiungi costo e ruolo dalla quotazioni (più affidabili dell'FBref)
        if 'costo_iniziale' in target_row:
            combined['costo_iniziale'] = float(target_row['costo_iniziale'])
        if 'ruolo' in target_row:
            combined['ruolo'] = str(target_row['ruolo'])

        rows.append(combined)
        matched += 1

    logger.info(
        f"Link cross-stagione: {matched}/{len(df_features)} giocatori matchati "
        f"su {len(df_target)} in quotazioni"
    )
    return pd.DataFrame(rows).reset_index(drop=True)


# ── Feature engineering (replica di predict.py per il dataset temporale) ─────

def _enrich_with_age_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge eta, prime_factor, presenze_pct, bonus_potential, gk_defense_score."""
    from predict import (
        _fill_missing_stats, _build_features,
        _load_age_data, _estimate_age, _prime_factor,
    )
    df = _fill_missing_stats(df)
    df = _build_features(df)
    return df


# ── Training ─────────────────────────────────────────────────────────────────

def _train_one_role(X: pd.DataFrame, y: pd.Series, ruolo: str):
    """Stesso modello adattivo di predict.py."""
    from predict import _train_role_model
    return _train_role_model(X, y, ruolo)


def train_temporal_models(training_df: pd.DataFrame) -> dict:
    """
    Addestra 4 modelli separati per ruolo sul dataset temporale.
    Ogni modello impara: features(N-1) → FM(N).
    """
    models   = {}
    metrics  = {}

    for ruolo in ['P', 'D', 'C', 'A']:
        mask = training_df['ruolo'] == ruolo
        sub  = training_df[mask]

        if mask.sum() < 5:
            logger.warning(f"Ruolo {ruolo}: solo {mask.sum()} campioni, skip")
            continue

        feat_cols = [c for c in TEMPORAL_FEATURES[ruolo] if c in sub.columns]
        X = sub[feat_cols].fillna(0)
        y = sub['fanta_media_next']

        # Cross-validation per stimare l'errore reale
        from sklearn.model_selection import KFold, cross_val_score
        import numpy as np

        n = len(X)
        cv = min(5, n)  # max 5 fold, non più del numero di campioni

        model = _train_one_role(X, y, ruolo)

        if n >= 10:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y,
                                     cv=cv, scoring='neg_mean_absolute_error')
            mae_cv = round(-scores.mean(), 3)
        else:
            model.fit(X, y)
            preds  = model.predict(X)
            mae_cv = round(float(np.abs(preds - y.values).mean()), 3)

        # Riaddestra su tutti i dati (dopo la CV)
        model.fit(X, y)
        models[ruolo]  = (feat_cols, model)
        metrics[ruolo] = {
            'n': int(mask.sum()),
            'mae_cv': mae_cv,
            'model_type': type(model).__name__,
        }
        logger.info(
            f"Temporale {ruolo}: {type(model).__name__} "
            f"su {mask.sum()} campioni — MAE cross-val: {mae_cv}"
        )

    return models, metrics


# ── Persistenza ───────────────────────────────────────────────────────────────

def save_temporal_models(models: dict, metrics: dict,
                          feature_season: str = DEFAULT_FEATURE_SEASON) -> None:
    if not _JOBLIB_OK:
        logger.warning("joblib non disponibile — modelli temporali non salvati")
        return
    joblib.dump({
        'models': models,
        'metrics': metrics,
        'feature_season': feature_season,
    }, TEMPORAL_MODELS_PATH)
    logger.info(f"Modelli temporali salvati in {TEMPORAL_MODELS_PATH}")


def load_temporal_models() -> tuple[dict | None, dict | None, str | None]:
    """Ritorna (models, metrics, feature_season) oppure (None, None, None)."""
    if not _JOBLIB_OK or not os.path.exists(TEMPORAL_MODELS_PATH):
        return None, None, None
    try:
        cache = joblib.load(TEMPORAL_MODELS_PATH)
        return cache['models'], cache.get('metrics', {}), cache.get('feature_season')
    except Exception as e:
        logger.warning(f"Errore caricamento modelli temporali: {e}")
        return None, None, None


def temporal_models_exist() -> bool:
    return os.path.exists(TEMPORAL_MODELS_PATH)


# ── Predizione con modelli temporali ─────────────────────────────────────────

def predict_with_temporal(df: pd.DataFrame, models: dict) -> pd.Series:
    """
    Applica i modelli temporali per predire la FM della stagione successiva.
    Ritorna una Series con previsione_ia per ogni giocatore.
    """
    pred = pd.Series(6.0, index=df.index)
    for ruolo, (feat_cols, model) in models.items():
        mask = df['ruolo'] == ruolo
        if not mask.any():
            continue
        available = [c for c in feat_cols if c in df.columns]
        missing   = [c for c in feat_cols if c not in df.columns]
        if missing:
            logger.debug(f"Feature mancanti per {ruolo}: {missing}")
        X = df.loc[mask, available].fillna(0)
        pred.loc[mask] = model.predict(X).clip(3.0, 10.0)
    return pred


# ── Pipeline principale ───────────────────────────────────────────────────────

def run_temporal_training(
    quotazioni_path: str,
    feature_season: str = DEFAULT_FEATURE_SEASON,
) -> tuple[bool, str, dict]:
    """
    Pipeline completo:
    1. Scarica FBref stats della stagione feature_season
    2. Carica le quotazioni (contiene fanta_media della stagione corrente)
    3. Collega i giocatori per nome
    4. Costruisce il dataset temporale con feature engineering
    5. Addestra 4 modelli per ruolo
    6. Salva e ritorna le metriche

    Ritorna (successo, messaggio, metriche).
    """
    # 1. Scarica/carica FBref stats stagione precedente
    from scraper_stats import fetch_and_save_stats, load_stats_for_season
    logger.info(f"Caricamento FBref stats {feature_season}...")

    fbref_stats = load_stats_for_season(feature_season)
    if fbref_stats is None or fbref_stats.empty:
        logger.info(f"Cache FBref {feature_season} assente, scaricamento in corso...")
        ok = fetch_and_save_stats(feature_season)
        if not ok:
            return False, (
                f"❌ Impossibile scaricare FBref stats {feature_season}.\n"
                "Prova a incollare il CSV manualmente (sezione 'FBref CSV' nella sidebar)."
            ), {}
        fbref_stats = load_stats_for_season(feature_season)

    if fbref_stats is None or fbref_stats.empty:
        return False, "❌ Dati FBref vuoti.", {}

    # 2. Carica quotazioni
    try:
        quotazioni = pd.read_csv(quotazioni_path)
    except Exception as e:
        return False, f"❌ Impossibile leggere quotazioni: {e}", {}

    if 'fanta_media' not in quotazioni.columns:
        return False, (
            "❌ Il file quotazioni non ha la colonna 'fanta_media'.\n"
            "Carica le quotazioni ufficiali di Fantacalcio.it (Excel quotazioni/statistiche)."
        ), {}

    # Filtra giocatori senza FM (riserve o errori)
    quotazioni = quotazioni.dropna(subset=['fanta_media'])
    quotazioni = quotazioni[quotazioni['fanta_media'] > 4.0]

    if len(quotazioni) < 30:
        return False, (
            f"❌ Solo {len(quotazioni)} giocatori con FM valida nelle quotazioni. "
            "Servono almeno 30."
        ), {}

    # 3. Collega i giocatori per nome
    linked = _link_by_name(fbref_stats, quotazioni, min_score=68)

    if len(linked) < 20:
        return False, (
            f"❌ Solo {len(linked)} giocatori matchati tra FBref e quotazioni. "
            "Prova ad abbassare il cutoff di matching o verifica i nomi nel file."
        ), {}

    # 4. Feature engineering
    try:
        linked = _enrich_with_age_and_features(linked)
    except Exception as e:
        logger.warning(f"Feature engineering parziale: {e}")

    # Salva il dataset di training per debug/analisi
    linked.to_csv(TRAINING_DATASET_PATH, index=False)
    logger.info(f"Dataset temporale: {len(linked)} giocatori → {TRAINING_DATASET_PATH}")

    # 5. Addestramento modelli
    models, metrics = train_temporal_models(linked)

    if not models:
        return False, "❌ Nessun ruolo aveva abbastanza dati per l'addestramento.", {}

    # 6. Salva
    save_temporal_models(models, metrics, feature_season)

    # 7. Messaggio di riepilogo
    lines = [
        f"✅ Modello temporale addestrato ({feature_season} → stagione corrente)\n",
        f"Giocatori nel training set: {len(linked)}",
        "",
        "Accuratezza per ruolo (MAE cross-val — FM prevista vs FM reale):",
    ]
    for ruolo, m in metrics.items():
        quality = "🟢 ottima" if m['mae_cv'] < 0.5 else "🟡 discreta" if m['mae_cv'] < 0.8 else "🔴 bassa"
        lines.append(
            f"  {ruolo}: MAE {m['mae_cv']:.2f} FM | {m['n']} giocatori | "
            f"{m['model_type']} | {quality}"
        )

    lines += [
        "",
        "Da ora /ottimizza e /top useranno questo modello temporale.",
        "Riaddestra ogni stagione con le nuove quotazioni.",
    ]
    return True, "\n".join(lines), metrics


# ── Metriche su dataset esterno ───────────────────────────────────────────────

def evaluate_on_dataset(models: dict, test_df: pd.DataFrame) -> dict:
    """
    Valuta i modelli temporali su un dataset con fanta_media reale.
    Utile per il backtest: confronta previsioni con risultati noti.
    """
    import numpy as np
    results = {}

    for ruolo, (feat_cols, model) in models.items():
        mask = test_df['ruolo'] == ruolo
        sub  = test_df[mask]
        if sub.empty or 'fanta_media' not in sub.columns:
            continue

        available = [c for c in feat_cols if c in sub.columns]
        X    = sub[available].fillna(0)
        y    = sub['fanta_media']
        pred = pd.Series(model.predict(X).clip(3.0, 10.0), index=sub.index)

        rmse = float(np.sqrt(((pred - y) ** 2).mean()))
        mae  = float((pred - y).abs().mean())
        corr = float(pred.corr(y)) if len(sub) > 2 else 0.0

        # Top-N accuracy
        acc = {}
        for n in [5, 10]:
            if len(sub) >= n:
                real_top = set(y.nlargest(n).index)
                pred_top = set(pred.nlargest(n).index)
                acc[f'top{n}'] = round(len(real_top & pred_top) / n, 2)

        results[ruolo] = {
            'rmse': round(rmse, 3), 'mae': round(mae, 3), 'corr': round(corr, 3),
            **acc,
        }

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    quotazioni_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "quotazioni_correnti.csv")
    feature_season  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_FEATURE_SEASON

    if not os.path.exists(quotazioni_path):
        # Usa il backtest come proxy (ha fanta_media 2023-24)
        quotazioni_path = os.path.join(DATA_DIR, "serie_a_23_24_backtest.csv")
        print(f"Quotazioni non trovate, uso backtest: {quotazioni_path}")

    ok, msg, metrics = run_temporal_training(quotazioni_path, feature_season)
    print(msg)

    if ok:
        print("\nTest predizione su backtest...")
        test = pd.read_csv(quotazioni_path)
        from predict import _fill_missing_stats, _build_features
        test = _fill_missing_stats(test)
        test = _build_features(test)
        models, _, _ = load_temporal_models()
        if models:
            pred = predict_with_temporal(test, models)
            test['previsione_temporale'] = pred
            print("\nTop 10 per FM prevista (modello temporale):")
            for _, r in test.nlargest(10, 'previsione_temporale').iterrows():
                fm_real = r.get('fanta_media', '?')
                fm_str  = f"{float(fm_real):.1f}" if fm_real != '?' else '?'
                print(
                    f"  {str(r['nome']):22s} {r['ruolo']} "
                    f"| Prev: {r['previsione_temporale']:.2f} "
                    f"| FM reale: {fm_str}"
                )
