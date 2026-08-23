"""
Scraper per statistiche Fantacalcio.it — sostituisce scraper_stats.py (FBref).

Il sito è una SPA (Single Page Application): i dati NON sono nel HTML statico.
L'unico modo affidabile è autenticarsi con le credenziali e scaricare l'Excel ufficiale.

Meccanismo di login: Laravel SPA → cookie XSRF-TOKEN → header X-XSRF-TOKEN.

Fallback manuale: l'utente scarica l'Excel dal proprio browser e lo carica nell'app.
"""

import os
import re
import time
import logging
import urllib.parse
from io import BytesIO

import requests
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL  = "https://www.fantacalcio.it"
STATS_API = f"{BASE_URL}/api/v1/Excel/stats/20/1"   # 20 = Serie A, 1 = stagione corrente
LOGIN_URL = f"{BASE_URL}/login"

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
CACHE_PATH = os.path.join(DATA_DIR, "fantacalcio_stats_cache.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
    "Accept": "application/json, text/html, */*",
    "Referer": BASE_URL,
    "Origin": BASE_URL,
}

STAT_COLS = ["gol_pg", "assist_pg", "ammonizioni_pg", "xg_pg", "titolarita_pct"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_number(val) -> float:
    try:
        return float(str(val).replace(",", ".").strip())
    except Exception:
        return 0.0


def _parse_rig(val) -> tuple[int, int]:
    try:
        parts = re.split(r"[/\\]", str(val))
        return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        return 0, 0


COL_MAP = {
    # Nome giocatore
    "Calciatore": "nome", "Nome": "nome", "Player": "nome", "name": "nome",
    # Ruolo
    "R": "ruolo", "Ruolo": "ruolo", "Pos": "ruolo", "Role": "ruolo",
    # Squadra
    "Sq": "squadra", "Squadra": "squadra", "Squad": "squadra", "Team": "squadra",
    # Presenze
    "PV": "presenze", "Pv": "presenze", "Presenze": "presenze", "MP": "presenze",
    "Pg": "presenze", "PG": "presenze", "G": "presenze",
    # Media Voto
    "MV": "media_voto", "Mv": "media_voto", "Media Voto": "media_voto", "MV ": "media_voto",
    # Fanta Media
    "FM": "fanta_media_fc", "Fm": "fanta_media_fc", "Fantamedia": "fanta_media_fc",
    "FantaMedia": "fanta_media_fc", "Fanta Media": "fanta_media_fc",
    # Gol
    "Gol": "gol", "Goals": "gol", "G.": "gol",
    # Gol subiti
    "GS": "gol_subiti", "Gs": "gol_subiti", "Gol Subiti": "gol_subiti",
    # Rigori
    "Rig": "rigori_raw", "Rigori": "rigori_raw",
    # Rigori parati
    "RP": "rigori_parati", "Rp": "rigori_parati", "Rig. Parati": "rigori_parati",
    # Assist
    "Ass": "assist", "Assist": "assist",
    # Ammonizioni
    "Amm": "ammonizioni", "Ammonizioni": "ammonizioni",
    # Espulsioni
    "Esp": "espulsioni", "Espulsioni": "espulsioni",
}

_KNOWN_JUNK = {
    "nan", "none", "calciatore", "nome", "player", "r", "ruolo",
    "squadra", "sq", "team", "#", ""
}


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Converte la tabella Fantacalcio.it (da Excel o CSV) nel formato interno.
    Gestisce diversi formati di intestazione e nomi colonna.
    """
    df = df.copy()

    # Normalizza nomi colonne: strip + mappa
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})

    # Se "nome" non c'è, cerca la colonna con testo più lungo (≈ nomi di giocatori)
    if "nome" not in df.columns:
        str_cols = [c for c in df.columns if df[c].dtype == object]
        for col in str_cols:
            sample = df[col].dropna().astype(str)
            if len(sample) > 5 and sample.str.len().median() > 5:
                df = df.rename(columns={col: "nome"})
                logger.info(f"Colonna 'nome' auto-rilevata da '{col}'")
                break

    if "nome" not in df.columns:
        logger.warning(f"Colonna 'nome' non trovata. Colonne disponibili: {list(df.columns)}")
        return None

    # Pulisci nomi
    df["nome"] = df["nome"].astype(str).str.strip()
    df = df[df["nome"].str.len() > 2].copy()
    df = df[~df["nome"].str.lower().isin(_KNOWN_JUNK)].copy()

    if df.empty:
        logger.warning("DataFrame vuoto dopo pulizia nomi")
        return None

    # Numerici in formato italiano (virgola → punto)
    for col in ["presenze", "media_voto", "fanta_media_fc", "gol", "gol_subiti",
                "rigori_parati", "assist", "ammonizioni", "espulsioni"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_number)

    # Rigori "N / M" o solo numero
    if "rigori_raw" in df.columns:
        parsed = df["rigori_raw"].apply(lambda x: _parse_rig(str(x)))
        df["rig_segnati"] = [r[0] for r in parsed]
        df["rig_tirati"]  = [r[1] for r in parsed]
        df.drop(columns=["rigori_raw"], inplace=True)
    else:
        df["rig_segnati"] = 0
        df["rig_tirati"]  = 0

    # Calcola metriche per partita
    if "presenze" not in df.columns:
        df["presenze"] = 0
    safe_pv = df["presenze"].apply(_parse_number).replace(0, 1)
    max_pv  = max(df["presenze"].apply(_parse_number).max(), 1)

    def _col(name: str) -> pd.Series:
        return df[name].apply(_parse_number) if name in df.columns else pd.Series(0.0, index=df.index)

    df["gol_pg"]         = (_col("gol")          / safe_pv).round(3)
    df["assist_pg"]      = (_col("assist")        / safe_pv).round(3)
    df["ammonizioni_pg"] = (_col("ammonizioni")   / safe_pv).round(3)
    df["titolarita_pct"] = (df["presenze"].apply(_parse_number) / max_pv).clip(0, 1).round(3)

    gol_open    = (_col("gol") - _col("rig_segnati")).clip(lower=0)
    df["xg_pg"] = (gol_open / safe_pv).round(3)

    # Portieri: gol subiti per partita + probabilità clean sheet (formula Poisson)
    # P(gol=0 | lambda) = exp(-lambda), dove lambda = gol_subiti_pg
    import math as _math
    df["gol_subiti_pg"] = (_col("gol_subiti") / safe_pv).round(3)
    df["clean_sheet_pg"] = df["gol_subiti_pg"].apply(
        lambda gs: round(_math.exp(-max(float(gs), 0.0)), 3)
    )

    logger.info(f"_normalize_df: {len(df)} giocatori, colonne: {list(df.columns)}")
    return df.reset_index(drop=True)


def _try_parse_excel(file_bytes: bytes) -> tuple[pd.DataFrame | None, str]:
    """
    Tenta di parsificare un Excel con diversi offset di intestazione.
    Ritorna (DataFrame, messaggio_errore).
    """
    for header_row in [0, 1, 2]:
        try:
            df = pd.read_excel(BytesIO(file_bytes), header=header_row)
            result = _normalize_df(df)
            if result is not None and len(result) > 5:
                logger.info(f"Excel parsificato con header_row={header_row}, {len(result)} righe")
                return result, ""
        except Exception as e:
            logger.debug(f"header_row={header_row} fallito: {e}")

    # Ultimo tentativo: leggi senza header e usa prima riga come header
    try:
        df = pd.read_excel(BytesIO(file_bytes), header=None)
        # Trova la riga che assomiglia a un'intestazione (contiene "Calciatore" o "Nome" o "FM")
        for i, row in df.iterrows():
            row_vals = [str(v).strip() for v in row.values]
            if any(v in COL_MAP for v in row_vals):
                df.columns = row_vals
                df = df.iloc[i+1:].reset_index(drop=True)
                result = _normalize_df(df)
                if result is not None and len(result) > 5:
                    logger.info(f"Excel parsificato con intestazione alla riga {i}")
                    return result, ""
                break
    except Exception as e:
        logger.debug(f"Tentativo header manuale fallito: {e}")

    # Raccogli info debug
    try:
        df_raw = pd.read_excel(BytesIO(file_bytes), header=0)
        cols = list(df_raw.columns[:10])
        first_rows = df_raw.head(3).to_string()
        msg = (
            f"Formato non riconosciuto. "
            f"Colonne trovate: {cols}. "
            f"Prime righe:\n{first_rows}"
        )
    except Exception as e:
        msg = f"Impossibile leggere il file: {e}"

    return None, msg


# ── Autenticazione (Laravel SPA con XSRF-TOKEN) ───────────────────────────────

def _login(email: str, password: str) -> requests.Session | None:
    """
    Login su Fantacalcio.it via pattern Laravel SPA:
    1. GET /login  → il server imposta il cookie XSRF-TOKEN
    2. POST /login con header X-XSRF-TOKEN + body JSON {email, password}
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        # Step 1: GET login page per ricevere XSRF-TOKEN cookie
        resp = session.get(LOGIN_URL, timeout=15)
        resp.raise_for_status()

        # Estrai XSRF-TOKEN dal cookie (Laravel lo URL-codifica)
        xsrf = session.cookies.get("XSRF-TOKEN") or session.cookies.get("xsrf-token")
        if xsrf:
            xsrf = urllib.parse.unquote(xsrf)

        # Step 2: POST credenziali
        post_headers = {
            **HEADERS,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Requested-With": "XMLHttpRequest",
        }
        if xsrf:
            post_headers["X-XSRF-TOKEN"] = xsrf

        resp = session.post(
            LOGIN_URL,
            json={"email": email, "password": password},
            headers=post_headers,
            allow_redirects=True,
            timeout=15,
        )

        # Verifica successo: status 200 o redirect fuori da /login
        if resp.status_code in (200, 302) and "/login" not in resp.url:
            logger.info("Login Fantacalcio.it: riuscito")
            return session

        # Alcuni siti rispondono 200 anche a POST /login con HTML
        # — considera successo se abbiamo almeno un cookie di sessione
        session_cookies = [c for c in session.cookies if "session" in c.name.lower()
                           or "laravel" in c.name.lower() or "remember" in c.name.lower()]
        if session_cookies:
            logger.info("Login Fantacalcio.it: riuscito (via cookie sessione)")
            return session

        logger.warning(f"Login fallito: status={resp.status_code}, url={resp.url}")
        logger.debug(f"Risposta login: {resp.text[:300]}")
        return None

    except Exception as e:
        logger.error(f"Errore login: {e}")
        return None


def _download_excel(session: requests.Session) -> pd.DataFrame | None:
    """Scarica il file Excel statistiche dall'API autenticata."""
    try:
        resp = session.get(STATS_API, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "")
            if "excel" in content_type or "spreadsheet" in content_type or len(resp.content) > 1000:
                df, err = _try_parse_excel(resp.content)
                if df is not None:
                    return df
                logger.error(f"Parsing Excel API fallito: {err}")
                return None
            else:
                logger.warning(f"Risposta inattesa dall'API Excel: {content_type}")
                return None
        logger.error(f"Excel API: HTTP {resp.status_code}")
        return None
    except Exception as e:
        logger.error(f"Errore download Excel: {e}")
        return None


# ── Import manuale (fallback senza login) ─────────────────────────────────────

def import_excel_manual(file_bytes: bytes) -> tuple[pd.DataFrame | None, str]:
    """
    Parsifica un file Excel scaricato manualmente da Fantacalcio.it.
    Ritorna (DataFrame, messaggio_errore).
    """
    return _try_parse_excel(file_bytes)


# ── API pubblica ──────────────────────────────────────────────────────────────

def fetch_and_save_stats(email: str = "", password: str = "") -> tuple[bool, str]:
    """
    Scarica le statistiche da Fantacalcio.it e le salva in cache.

    Richiede credenziali: il sito è una SPA e i dati non sono nel HTML statico.
    Senza credenziali, usa solo la cache esistente o il file caricato manualmente.
    """
    if not email or not password:
        return False, (
            "Inserisci email e password Fantacalcio.it nell'expander '🔑 Account' "
            "oppure carica il file Excel manualmente."
        )

    auth_session = _login(email, password)
    if auth_session is None:
        return False, "Login fallito — verifica email e password"

    df = _download_excel(auth_session)
    if df is None or df.empty:
        return False, "Download Excel fallito dopo il login"

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    logger.info(f"Stats salvate: {len(df)} giocatori → {CACHE_PATH}")
    return True, f"✅ {len(df)} giocatori scaricati da Fantacalcio.it"


def save_manual_excel(file_bytes: bytes) -> tuple[bool, str]:
    """Salva in cache un Excel caricato manualmente dall'utente."""
    df, err = import_excel_manual(file_bytes)
    if df is None or df.empty:
        return False, (
            f"File non riconosciuto. {err}\n\n"
            "Assicurati di caricare il file Excel della pagina Statistiche di Fantacalcio.it, "
            "non la pagina Quotazioni."
        )
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return True, f"✅ {len(df)} giocatori importati dall'Excel Fantacalcio.it"


def load_cached_stats() -> pd.DataFrame | None:
    if os.path.exists(CACHE_PATH):
        try:
            return pd.read_csv(CACHE_PATH)
        except Exception:
            return None
    return None


# ── Merge con DataFrame quotazioni ────────────────────────────────────────────

def merge_stats_with_quotazioni(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arricchisce il DataFrame quotazioni con le stats Fantacalcio.it.
    Match diretto per nome (stessa fonte) + fallback sul cognome.
    """
    stats = load_cached_stats()
    if stats is None or stats.empty:
        return df

    available = [c for c in STAT_COLS if c in stats.columns]
    if not available:
        return df

    df = df.copy()
    stats_clean = stats.dropna(subset=["nome"])
    lookup_exact = stats_clean.set_index("nome")[available].to_dict(orient="index")
    lookup_lower = {k.lower().strip(): v for k, v in lookup_exact.items()}

    def _get(nome: str, col: str):
        n = str(nome).lower().strip()
        row = lookup_lower.get(n)
        if row:
            return row.get(col)
        # Fallback: match sul cognome (ultima parola)
        cognome = n.split()[-1]
        for key, row in lookup_lower.items():
            if cognome and cognome == key.split()[-1]:
                return row.get(col)
        return None

    for col in available:
        df[col] = df["nome"].apply(lambda n, c=col: _get(n, c))

    return df


# ── Alias backward-compatibili ────────────────────────────────────────────────

def merge_stats_for_season(df: pd.DataFrame, season: str = "current") -> pd.DataFrame:
    return merge_stats_with_quotazioni(df)


def load_stats_for_season(season: str = "current") -> pd.DataFrame | None:
    return load_cached_stats()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    email    = os.environ.get("FANTACALCIO_EMAIL", "")
    password = os.environ.get("FANTACALCIO_PASSWORD", "")
    if not email:
        print("Uso: FANTACALCIO_EMAIL=... FANTACALCIO_PASSWORD=... python scraper_fantacalcio.py")
        sys.exit(1)
    ok, msg = fetch_and_save_stats(email, password)
    print(msg)
    if ok:
        df = load_cached_stats()
        print(df[["nome", "gol_pg", "assist_pg", "titolarita_pct"]].head(15).to_string())
