"""
Parser strutturato di infortuni/squalifiche da feed RSS di AgenteFantacalcio.
Estrae eventi per giocatore e calcola un modificatore da applicare alle previsioni.
"""
import re
import json
import os
import time
import logging
import warnings
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INJURIES_CACHE = os.path.join(DATA_DIR, "injuries_cache.json")

HEADERS = {'User-Agent': 'Mozilla/5.0'}

# (pattern_regex, tipo_evento, modificatore_base)
_EVENT_RULES: list[tuple[str, str, float]] = [
    (r'stagione\s+finit|lungo\s+stop|mesi\s+di\s+stop|rottura',    'infortunio_grave',  0.30),
    (r'infortun\w+|stop\s+di|problema\s+fisic|si\s+ferma|lesion',  'infortunio',         0.70),
    (r'squalific\w+',                                               'squalifica',         0.80),
    (r'panch\w+|esclus\w+|non\s+convocato',                        'panchina',           0.82),
    (r'ritorn\w+\s+in\s+gruppo|recup\w+|disponibil\w+|rientr\w+',  'recupero',           1.05),
    (r'\britolare\b|conferm\w+\s+tit',                             'titolare',           1.10),
    (r'rigorista',                                                  'rigorista',          1.15),
    (r'nuovo\s+allenator\w+|cambio\s+(di\s+)?panch',               'cambio_allenatore',  1.00),
]

_DURATION_RULES: list[tuple[str, int]] = [
    (r'(\d+)\s*settiman\w+', 7),
    (r'(\d+)\s*mes\w+',      30),
    (r'(\d+)\s*giorn\w+',    1),
]

STALE_DAYS = 30


def _duration_days(text: str) -> int | None:
    for pattern, mult in _DURATION_RULES:
        m = re.search(pattern, text.lower())
        if m:
            try:
                return int(m.group(1)) * mult
            except Exception:
                pass
    return None


def _modifier_from_injury(duration_days: int | None, base: float) -> float:
    if base > 1.0 or duration_days is None:
        return base
    if duration_days > 60:
        return 0.25
    if duration_days > 21:
        return 0.45
    if duration_days > 7:
        return 0.65
    return base


def _parse_item(title: str, description: str, player_names: list[str]) -> list[dict]:
    full = f"{title} {description}".lower()
    results = []

    for player in player_names:
        parts = player.lower().split()
        # Match on surname (last word, min 3 chars)
        surname = next((p for p in reversed(parts) if len(p) >= 3), None)
        if not surname or surname not in full:
            continue

        for pattern, event_type, base_mod in _EVENT_RULES:
            if re.search(pattern, full):
                dur = _duration_days(full)
                modifier = _modifier_from_injury(dur, base_mod)
                results.append({
                    'nome': player,
                    'tipo': event_type,
                    'modifier': round(modifier, 3),
                    'duration_days': dur,
                    'testo': title[:120],
                    'timestamp': datetime.now().isoformat(),
                })
                break

    return results


def fetch_injury_updates(player_names: list[str]) -> dict[str, dict]:
    """
    Scarica il feed RSS di AgenteFantacalcio e restituisce
    un dict {nome_giocatore: {tipo, modifier, duration_days, testo, timestamp}}.
    Mantiene l'evento più grave per ogni giocatore.
    """
    url = "https://news.google.com/rss/search?q=fantacalcio+infortunio+OR+squalifica&hl=it&gl=IT&ceid=IT:it"
    events: dict[str, dict] = {}

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'xml')
        items = soup.find_all('item')

        for item in items:
            title = item.find('title').get_text() if item.find('title') else ""
            desc = item.find('description').get_text() if item.find('description') else ""
            clean_desc = BeautifulSoup(desc, "html.parser").get_text().strip()

            for event in _parse_item(title, clean_desc, player_names):
                nome = event['nome']
                if nome not in events or event['modifier'] < events[nome]['modifier']:
                    events[nome] = event

    except Exception as e:
        logger.warning(f"Feed infortuni non disponibile: {e}")

    return events


def fetch_and_save_injuries(player_names: list[str]) -> tuple[bool, int]:
    """Scarica gli aggiornamenti e li salva in cache. Restituisce (ok, n_trovati)."""
    events = fetch_injury_updates(player_names)
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(INJURIES_CACHE, 'w', encoding='utf-8') as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        return True, len(events)
    except Exception as e:
        logger.error(f"Errore salvataggio cache infortuni: {e}")
        return False, 0


def load_cached_injuries() -> dict[str, dict]:
    if os.path.exists(INJURIES_CACHE):
        try:
            with open(INJURIES_CACHE, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_injury_modifiers(player_names: list[str]) -> dict[str, float]:
    """
    Restituisce {nome: modifier} per i giocatori con eventi recenti in cache.
    Gli eventi più vecchi di STALE_DAYS vengono ignorati.
    """
    cache = load_cached_injuries()
    modifiers: dict[str, float] = {}
    cutoff = datetime.now() - timedelta(days=STALE_DAYS)

    for nome in player_names:
        surname = nome.lower().split()[-1] if nome.lower().split() else nome.lower()
        match_key = None

        if nome in cache:
            match_key = nome
        else:
            for key in cache:
                if surname in key.lower():
                    match_key = key
                    break

        if match_key:
            event = cache[match_key]
            try:
                ts = datetime.fromisoformat(event['timestamp'])
                if ts >= cutoff:
                    modifiers[nome] = event['modifier']
            except Exception:
                modifiers[nome] = event['modifier']

    return modifiers


def apply_injury_modifiers(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Applica i modificatori infortuni/recupero a previsione_ia e score_selezione.
    Salva anche injury_type per le spiegazioni per giocatore.
    """
    import pandas as pd
    cache = load_cached_injuries()
    cutoff = datetime.now() - timedelta(days=STALE_DAYS)

    modifiers_map: dict[str, float] = {}
    types_map: dict[str, str] = {}

    for nome in df['nome'].tolist():
        surname = nome.lower().split()[-1] if nome.lower().split() else nome.lower()
        match_key = nome if nome in cache else next(
            (k for k in cache if surname in k.lower()), None
        )
        if match_key:
            event = cache[match_key]
            try:
                ts = datetime.fromisoformat(event['timestamp'])
                valid = ts >= cutoff
            except Exception:
                valid = True
            if valid:
                modifiers_map[nome] = event['modifier']
                types_map[nome] = event.get('tipo', '')

    if not modifiers_map:
        return df

    df = df.copy()
    df['injury_modifier'] = df['nome'].map(modifiers_map).fillna(1.0)
    df['injury_type'] = df['nome'].map(types_map).fillna('')
    df['previsione_ia'] = (df['previsione_ia'] * df['injury_modifier']).clip(2.0, 10.0)

    # Propaga anche a score_selezione usato dall'ottimizzatore MILP
    if 'score_selezione' in df.columns:
        df['score_selezione'] = (df['score_selezione'] * df['injury_modifier']).clip(2.0, 14.0)

    return df


def get_injury_summary(player_names: list[str]) -> list[dict]:
    """Restituisce lista di eventi recenti ordinati per gravità."""
    cache = load_cached_injuries()
    cutoff = datetime.now() - timedelta(days=STALE_DAYS)
    results = []

    for nome in player_names:
        surname = nome.lower().split()[-1] if nome.lower().split() else nome.lower()
        match_key = None

        if nome in cache:
            match_key = nome
        else:
            for key in cache:
                if surname in key.lower():
                    match_key = key
                    break

        if match_key:
            event = cache[match_key]
            try:
                ts = datetime.fromisoformat(event['timestamp'])
                if ts >= cutoff:
                    results.append({'nome': nome, **event})
            except Exception:
                results.append({'nome': nome, **event})

    return sorted(results, key=lambda x: x['modifier'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import pandas as pd
    test_players = ['Lautaro Martinez', 'Vlahovic', 'Theo Hernandez', 'Barella', 'Sommer']
    ok, n = fetch_and_save_injuries(test_players)
    print(f"Infortuni trovati: {n}")
    modifiers = get_injury_modifiers(test_players)
    for nome, mod in modifiers.items():
        print(f"  {nome}: modifier={mod}")
