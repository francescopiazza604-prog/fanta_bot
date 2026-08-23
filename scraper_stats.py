import requests
import pandas as pd
from io import StringIO
import os
import time
import logging

logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9,it;q=0.8',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Referer': 'https://fbref.com/',
}
FBREF_SERIE_A_URL = "https://fbref.com/en/comps/11/stats/Serie-A-Stats"
FBREF_MANUAL_GUIDE = (
    "FBref è protetto da Cloudflare. Per scaricare i dati manualmente:\n"
    "1. Apri https://fbref.com/en/comps/11/stats/Serie-A-Stats nel browser\n"
    "2. Scorri fino alla tabella 'Standard Stats'\n"
    "3. Clicca 'Share & Export' → 'Get table as CSV'\n"
    "4. Copia il testo CSV e incollalo nella casella 'Incolla CSV FBref' nella sidebar\n"
    "   oppure caricalo come file .csv"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATS_CACHE_PATH = os.path.join(DATA_DIR, "fbref_stats_cache.csv")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    seen: dict[str, int] = {}
    new_cols = []
    for col in df.columns:
        name = str(col[-1])
        if name in seen:
            seen[name] += 1
            new_cols.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            new_cols.append(name)
    df.columns = new_cols
    return df


def _fbref_url(season: str) -> str:
    if season == "current":
        return FBREF_SERIE_A_URL
    return f"https://fbref.com/en/comps/11/{season}/stats/{season}-Serie-A-Stats"


def _fetch_html(url: str) -> str | None:
    """
    Scarica HTML con bypass Cloudflare (cloudscraper) e fallback requests.
    Ritorna il testo HTML oppure None se entrambi falliscono.
    """
    # Tentativo 1: cloudscraper (gestisce JS challenge di Cloudflare)
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "darwin", "mobile": False}
        )
        time.sleep(2)
        resp = scraper.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.text) > 5000:
            logger.info("FBref: scaricato via cloudscraper")
            return resp.text
        logger.warning(f"cloudscraper: status {resp.status_code}, testo corto ({len(resp.text)} char)")
    except Exception as e:
        logger.warning(f"cloudscraper fallito: {e}")

    # Tentativo 2: requests standard con ritardo
    try:
        time.sleep(5)
        resp = requests.get(url, headers=HEADERS, timeout=25)
        resp.raise_for_status()
        if len(resp.text) > 5000:
            logger.info("FBref: scaricato via requests standard")
            return resp.text
    except requests.RequestException as e:
        logger.error(f"requests fallito: {e}")

    return None


def parse_fbref_csv_text(csv_text: str) -> pd.DataFrame | None:
    """
    Parsifica testo CSV incollato dalla pagina FBref ('Share & Export → Get table as CSV').
    Ritorna DataFrame con le stesse colonne di fetch_fbref_stats, oppure None.
    """
    try:
        df = pd.read_csv(StringIO(csv_text.strip()))
        return _normalize_fbref_df(df)
    except Exception as e:
        logger.error(f"Errore parsing CSV FBref manuale: {e}")
        return None


def _uncomment_fbref_tables(html: str) -> str:
    """
    FBref nasconde le tabelle dentro commenti HTML per bloccare lo scraping:
        <!-- <table id="stats_standard">...</table> -->
    Questa funzione rimuove i commenti che contengono tag <table>, rendendo
    le tabelle visibili a pd.read_html.
    """
    import re
    return re.sub(
        r'<!--\s*(<table[\s\S]*?</table>)\s*-->',
        r'\1',
        html,
        flags=re.IGNORECASE,
    )


def parse_fbref_html(html_content: str) -> pd.DataFrame | None:
    """
    Parsifica la pagina HTML di FBref salvata con Ctrl+S dal browser.
    FBref nasconde le tabelle in commenti HTML — le estraiamo prima di parsificare.
    """
    # Rimuove i commenti HTML che avvolgono le tabelle (anti-scraping FBref)
    html_clean = _uncomment_fbref_tables(html_content)

    # Prova prima con l'id esatto della tabella Standard Stats
    for attrs in [{'id': 'stats_standard'}, {'id': 'stats_standard_sq'}]:
        try:
            tables = pd.read_html(StringIO(html_clean), attrs=attrs)
            if tables:
                t = _flatten_columns(tables[0])
                result = _normalize_fbref_df(t)
                if result is not None and not result.empty:
                    logger.info(f"FBref HTML (id={attrs}): {len(result)} giocatori")
                    return result
        except Exception:
            pass

    # Fallback: scansiona tutte le tabelle cercando quella con 'Player'
    try:
        all_tables = pd.read_html(StringIO(html_clean))
    except Exception as e:
        logger.error(f"Errore lettura HTML FBref: {e}")
        return None

    best: pd.DataFrame | None = None
    for t in all_tables:
        t = _flatten_columns(t)
        cols = [str(c) for c in t.columns]
        has_player = any('player' in c.lower() for c in cols)
        if not has_player:
            continue
        # Rinomina varianti di 'Player' in 'Player' standard
        for c in cols:
            if 'player' in c.lower() and c != 'Player':
                t = t.rename(columns={c: 'Player'})
                break
        result = _normalize_fbref_df(t)
        if result is not None and not result.empty:
            if best is None or len(result) > len(best):
                best = result  # tieni la tabella con più giocatori

    if best is not None:
        logger.info(f"FBref HTML (fallback scan): {len(best)} giocatori")
        return best

    logger.warning(
        "Nessuna tabella Standard Stats trovata. "
        "Prova a salvare la pagina con Ctrl+S → 'Pagina web, solo HTML'."
    )
    return None


def _normalize_fbref_df(df: pd.DataFrame) -> pd.DataFrame | None:
    """Normalizza un DataFrame FBref (da HTML o CSV) in formato standard."""
    df = _flatten_columns(df)

    col_map = {
        'Player': 'nome_fbref', 'Squad': 'squadra', 'Pos': 'ruolo_fbref',
        'MP': 'presenze', 'Starts': 'titolarita', 'Min': 'minuti',
        'Gls': 'gol', 'Ast': 'assist', 'CrdY': 'ammonizioni',
        'CrdR': 'espulsioni', 'xG': 'xg', 'xAG': 'xag',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if 'nome_fbref' not in df.columns:
        logger.warning("Colonna 'Player' non trovata nel DataFrame FBref")
        return None

    df = df[df['nome_fbref'] != 'Player'].dropna(subset=['nome_fbref'])

    num_cols = ['presenze', 'titolarita', 'minuti', 'gol', 'assist',
                'ammonizioni', 'espulsioni', 'xg', 'xag']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    safe_mp = df['presenze'].replace(0, 1) if 'presenze' in df.columns else pd.Series(1, index=df.index)
    df['gol_pg'] = (df.get('gol', pd.Series(0, index=df.index)) / safe_mp).round(3)
    df['assist_pg'] = (df.get('assist', pd.Series(0, index=df.index)) / safe_mp).round(3)
    df['ammonizioni_pg'] = (df.get('ammonizioni', pd.Series(0, index=df.index)) / safe_mp).round(3)
    df['xg_pg'] = (df.get('xg', pd.Series(0, index=df.index)) / safe_mp).round(3)
    df['titolarita_pct'] = (df.get('titolarita', safe_mp) / safe_mp).round(3)

    keep = ['nome_fbref', 'squadra', 'ruolo_fbref', 'presenze',
            'gol_pg', 'assist_pg', 'ammonizioni_pg', 'xg_pg', 'titolarita_pct']
    keep = [c for c in keep if c in df.columns]
    return df[keep].reset_index(drop=True)


def fetch_fbref_stats(season: str = "current") -> pd.DataFrame | None:
    """
    Scarica le stats Serie A da FBref. Usa cloudscraper per bypassare Cloudflare.
    season: "current" oppure "YYYY-YYYY" (es. "2022-2023")
    """
    url = _fbref_url(season)
    html = _fetch_html(url)
    if html is None:
        logger.error(f"FBref non raggiungibile per stagione '{season}'. {FBREF_MANUAL_GUIDE}")
        return None

    try:
        all_tables = pd.read_html(StringIO(html))

        # Cerca la tabella Standard Stats (Player in colonne, >50 righe)
        for t in all_tables:
            t = _flatten_columns(t)
            if 'Player' in t.columns and len(t) > 50:
                return _normalize_fbref_df(t)

        logger.warning("Tabella standard stats non trovata su FBref — struttura pagina cambiata?")
        return None

    except Exception as e:
        logger.error(f"Errore parsing FBref: {e}")
        return None


def fetch_and_save_stats(season: str = "current") -> bool:
    """Fetch FBref stats and cache them. Returns True on success."""
    df = fetch_fbref_stats(season)
    if df is not None:
        os.makedirs(DATA_DIR, exist_ok=True)
        if season == "current":
            cache_path = STATS_CACHE_PATH
        else:
            cache_path = os.path.join(DATA_DIR, f"fbref_stats_{season.replace('-', '_')}.csv")
        df.to_csv(cache_path, index=False)
        logger.info(f"Stats FBref [{season}] salvate: {len(df)} giocatori")
        return True
    return False


def load_stats_for_season(season: str = "current") -> pd.DataFrame | None:
    """Load cached stats for a specific season."""
    if season == "current":
        cache_path = STATS_CACHE_PATH
    else:
        cache_path = os.path.join(DATA_DIR, f"fbref_stats_{season.replace('-', '_')}.csv")
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path)
        except Exception:
            return None
    return None


def merge_stats_for_season(df: pd.DataFrame, season: str = "current") -> pd.DataFrame:
    """Enrich a quotazioni DataFrame with FBref stats from a specific season."""
    stats = load_stats_for_season(season)
    if stats is None or stats.empty:
        return df

    available = [c for c in STAT_COLS if c in stats.columns]
    if not available:
        return df

    df = df.copy()
    fbref_names = stats['nome_fbref'].tolist()

    try:
        from rapidfuzz import process, fuzz
        def find_match(nome: str) -> str | None:
            result = process.extractOne(
                str(nome).lower(),
                [n.lower() for n in fbref_names],
                scorer=fuzz.token_set_ratio,
                score_cutoff=65,
            )
            if result:
                return fbref_names[[n.lower() for n in fbref_names].index(result[0])]
            return None
    except ImportError:
        def find_match(nome: str) -> str | None:
            return _match_name(nome, fbref_names)

    rows = []
    for nome in df['nome']:
        match = find_match(str(nome))
        if match:
            row_data = stats.loc[stats['nome_fbref'] == match, available].iloc[0].to_dict()
        else:
            row_data = {c: None for c in available}
        rows.append(row_data)

    enriched = pd.DataFrame(rows, index=df.index)
    for col in available:
        df[col] = enriched[col]

    for col in available:
        if col in _ROLE_DEFAULTS:
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = df.loc[mask, 'ruolo'].map(
                    _ROLE_DEFAULTS[col]
                ).fillna(0.1)

    return df


def load_cached_stats() -> pd.DataFrame | None:
    if os.path.exists(STATS_CACHE_PATH):
        try:
            return pd.read_csv(STATS_CACHE_PATH)
        except Exception:
            return None
    return None


_ROLE_DEFAULTS = {
    'gol_pg':         {'P': 0.00, 'D': 0.04, 'C': 0.12, 'A': 0.35},
    'assist_pg':      {'P': 0.00, 'D': 0.06, 'C': 0.15, 'A': 0.15},
    'ammonizioni_pg': {'P': 0.05, 'D': 0.18, 'C': 0.15, 'A': 0.10},
    'xg_pg':          {'P': 0.00, 'D': 0.03, 'C': 0.10, 'A': 0.30},
    'titolarita_pct': {'P': 0.75, 'D': 0.70, 'C': 0.70, 'A': 0.65},
}

STAT_COLS = ['gol_pg', 'assist_pg', 'ammonizioni_pg', 'xg_pg', 'titolarita_pct']


def _match_name(nome: str, fbref_names: list[str]) -> str | None:
    """Match an Italian last-name against FBref full names."""
    cognome = str(nome).lower().split()[-1] if nome else ""
    best_score, best = 0.0, None
    for cand in fbref_names:
        for part in cand.lower().split():
            if len(part) < 3:
                continue
            score = min(len(cognome), len(part)) / max(len(cognome), len(part), 1)
            if (cognome in part or part in cognome) and score > best_score:
                best_score, best = score, cand
    return best if best_score >= 0.65 else None


def merge_stats_with_quotazioni(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a quotazioni DataFrame with FBref per-game stats."""
    stats = load_cached_stats()
    if stats is None or stats.empty:
        return df

    available = [c for c in STAT_COLS if c in stats.columns]
    if not available:
        return df

    df = df.copy()
    fbref_names = stats['nome_fbref'].tolist()

    try:
        from rapidfuzz import process, fuzz
        def find_match(nome: str) -> str | None:
            result = process.extractOne(
                str(nome).lower(),
                [n.lower() for n in fbref_names],
                scorer=fuzz.token_set_ratio,
                score_cutoff=65,
            )
            if result:
                return fbref_names[[n.lower() for n in fbref_names].index(result[0])]
            return None
    except ImportError:
        def find_match(nome: str) -> str | None:
            return _match_name(nome, fbref_names)

    rows = []
    for nome in df['nome']:
        match = find_match(str(nome))
        if match:
            row_data = stats.loc[stats['nome_fbref'] == match, available].iloc[0].to_dict()
        else:
            row_data = {c: None for c in available}
        rows.append(row_data)

    enriched = pd.DataFrame(rows, index=df.index)
    for col in available:
        df[col] = enriched[col]

    for col in available:
        if col in _ROLE_DEFAULTS:
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = df.loc[mask, 'ruolo'].map(
                    _ROLE_DEFAULTS[col]
                ).fillna(0.1)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Scaricamento stats Serie A da FBref...")
    ok = fetch_and_save_stats()
    if ok:
        df = load_cached_stats()
        print(df.head(10).to_string())
    else:
        print("Errore nel download.")
