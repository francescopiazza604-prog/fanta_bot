import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
import time
import logging

logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'it-IT,it;q=0.9,en-US;q=0.8',
    'Referer': 'https://www.google.it/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRANSFERS_CACHE = os.path.join(DATA_DIR, "transfers_cache.csv")

# Tier 1 = top club, Tier 3 = piccolo club
SERIE_A_TIERS: dict[int, list[str]] = {
    1: ['inter', 'juventus', 'milan', 'napoli', 'roma', 'lazio'],
    2: ['atalanta', 'fiorentina', 'bologna', 'torino', 'hellas verona', 'udinese', 'sampdoria', 'genoa'],
    3: ['cagliari', 'monza', 'lecce', 'empoli', 'salernitana', 'frosinone',
        'spezia', 'cremonese', 'venezia', 'parma', 'como', 'brescia', 'ascoli'],
}

# (stile_gioco, bonus_FM_per_ruolo)
# Il bonus è il delta FM atteso per un giocatore di quel ruolo in questo sistema tattico.
# Stili: 'pressing' | 'possesso' | 'contropiede' | 'equilibrato'
COACH_TACTICAL_PROFILES: dict[str, tuple[str, dict[str, float]]] = {
    'inter':         ('pressing',     {'P': 0.05, 'D': 0.15, 'C': 0.10, 'A': 0.10}),
    'juventus':      ('equilibrato',  {'P': 0.05, 'D': 0.10, 'C': 0.05, 'A': 0.05}),
    'milan':         ('pressing',     {'P': 0.05, 'D': 0.10, 'C': 0.15, 'A': 0.10}),
    'napoli':        ('possesso',     {'P': 0.00, 'D': 0.05, 'C': 0.20, 'A': 0.15}),
    'atalanta':      ('pressing',     {'P': 0.05, 'D': 0.20, 'C': 0.15, 'A': 0.15}),
    'roma':          ('equilibrato',  {'P': 0.00, 'D': 0.05, 'C': 0.10, 'A': 0.10}),
    'lazio':         ('contropiede',  {'P': 0.00, 'D': 0.05, 'C': 0.05, 'A': 0.20}),
    'fiorentina':    ('possesso',     {'P': 0.00, 'D': 0.05, 'C': 0.15, 'A': 0.10}),
    'bologna':       ('pressing',     {'P': 0.00, 'D': 0.10, 'C': 0.10, 'A': 0.10}),
    'torino':        ('equilibrato',  {'P': 0.00, 'D': 0.10, 'C': 0.05, 'A': 0.05}),
    'hellas verona': ('pressing',     {'P': 0.00, 'D': 0.10, 'C': 0.05, 'A': 0.05}),
    'udinese':       ('contropiede',  {'P': 0.00, 'D': 0.05, 'C': 0.00, 'A': 0.10}),
    'genoa':         ('equilibrato',  {'P': 0.00, 'D': 0.05, 'C': 0.00, 'A': 0.05}),
    'cagliari':      ('contropiede',  {'P': 0.00, 'D': 0.05, 'C': 0.00, 'A': 0.10}),
    'monza':         ('equilibrato',  {'P': 0.00, 'D': 0.05, 'C': 0.05, 'A': 0.05}),
    'lecce':         ('contropiede',  {'P': 0.00, 'D': 0.05, 'C': 0.00, 'A': 0.05}),
    'empoli':        ('possesso',     {'P': 0.00, 'D': 0.05, 'C': 0.10, 'A': 0.05}),
    'venezia':       ('equilibrato',  {'P': 0.00, 'D': 0.00, 'C': 0.00, 'A': 0.05}),
    'como':          ('possesso',     {'P': 0.00, 'D': 0.05, 'C': 0.10, 'A': 0.10}),
    'parma':         ('equilibrato',  {'P': 0.00, 'D': 0.00, 'C': 0.05, 'A': 0.05}),
    'salernitana':   ('contropiede',  {'P': 0.00, 'D': 0.00, 'C': 0.00, 'A': 0.05}),
    'frosinone':     ('pressing',     {'P': 0.00, 'D': 0.05, 'C': 0.05, 'A': 0.05}),
}


def get_coach_tactical_bonus(club: str, player_role: str) -> float:
    """Bonus FM atteso per il ruolo in base al sistema tattico del club di destinazione."""
    club_key = str(club).lower().strip()
    for key, (_, bonuses) in COACH_TACTICAL_PROFILES.items():
        if key in club_key or club_key in key:
            return bonuses.get(player_role, 0.0)
    return 0.0


def get_coach_style(club: str) -> str:
    """Stile di gioco del club (pressing / possesso / contropiede / equilibrato)."""
    club_key = str(club).lower().strip()
    for key, (style, _) in COACH_TACTICAL_PROFILES.items():
        if key in club_key or club_key in key:
            return style
    return 'equilibrato'


_TRANSFER_DEFAULTS = {
    'cambio_squadra': 0,
    'upgrade_squadra': 0,
    'tipo_prestito': 0,
    'coach_bonus': 0.0,
    'coach_style': 'equilibrato',
}


def get_club_tier(club: str) -> int:
    name = str(club).lower().strip()
    for tier, clubs in SERIE_A_TIERS.items():
        for c in clubs:
            if c in name or name in c:
                return tier
    return 2


def _calc_impact(from_club: str, to_club: str, tipo: str, player_role: str = 'C') -> dict:
    from_tier = get_club_tier(from_club)
    to_tier = get_club_tier(to_club)
    upgrade = max(-1, min(1, from_tier - to_tier))
    return {
        'cambio_squadra': 1,
        'upgrade_squadra': upgrade,
        'tipo_prestito': 1 if 'prestito' in str(tipo).lower() else 0,
        'coach_bonus': get_coach_tactical_bonus(to_club, player_role),
        'coach_style': get_coach_style(to_club),
    }


def fetch_transfermarkt_transfers(season_year: int = 2026) -> pd.DataFrame | None:
    """
    Prova a scrapare i trasferimenti Serie A da Transfermarkt.
    Restituisce None se il sito non risponde o blocca la richiesta.
    """
    url = (f"https://www.transfermarkt.it/serie-a/transfers/wettbewerb/IT1"
           f"/saison_id/{season_year}")
    try:
        time.sleep(2)
        try:
            import cloudscraper
            scraper = cloudscraper.create_scraper()
            resp = scraper.get(url, timeout=20)
        except Exception:
            resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'lxml')
        transfers = []
        seen_names = set()

        for club_box in soup.select('.box'):
            headline = club_box.select_one('.content-box-headline a, .content-box-headline, h2, h3')
            club_name = headline.get_text(strip=True) if headline else ''

            for row in club_box.select('tr'):
                try:
                    p_link = row.find('a', href=lambda h: h and '/profil/spieler/' in h)
                    if not p_link:
                        continue
                    nome = p_link.get_text(strip=True)
                    if not nome or len(nome) < 3 or nome in seen_names:
                        continue

                    # Estrarre squadre di origine e destinazione
                    c_links = row.find_all('a', href=lambda h: h and ('/verein/' in h or '/startseite/verein/' in h))
                    from_c = c_links[0].get_text(strip=True) if len(c_links) > 0 and c_links[0].get_text(strip=True) else club_name
                    to_c = c_links[1].get_text(strip=True) if len(c_links) > 1 and c_links[1].get_text(strip=True) else club_name

                    cells = row.select('td')
                    fee_text = cells[-1].get_text(strip=True).lower() if cells else ''
                    tipo = 'prestito' if 'prest' in fee_text or 'loan' in fee_text else 'definitivo'

                    seen_names.add(nome)
                    impact = _calc_impact(from_c, to_c, tipo)
                    transfers.append({
                        'nome_tm': nome,
                        'da_squadra': from_c,
                        'a_squadra': to_c,
                        'tipo': tipo,
                        **impact,
                    })
                except Exception:
                    continue

        if transfers:
            logger.info(f"Transfermarkt: {len(transfers)} trasferimenti trovati")
            return pd.DataFrame(transfers).drop_duplicates(subset=['nome_tm'])

        logger.warning("Transfermarkt: nessun trasferimento parsato")
        return None

    except Exception as e:
        logger.warning(f"Transfermarkt scraping fallito: {e}")
        return None


def parse_manual_transfers(text: str) -> pd.DataFrame:
    """
    Parsa trasferimenti inseriti manualmente.
    Formato: NomeGiocatore;DaSquadra;ASquadra;tipo (una riga per trasferimento)
    Tipo opzionale: "definitivo" (default) o "prestito"
    Esempio: Vlahovic;Juventus;Arsenal;definitivo
    """
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split(';')]
        if len(parts) < 3:
            continue
        nome = parts[0]
        from_c = parts[1]
        to_c = parts[2]
        tipo = parts[3] if len(parts) > 3 else 'definitivo'
        impact = _calc_impact(from_c, to_c, tipo)
        rows.append({'nome_tm': nome, 'da_squadra': from_c, 'a_squadra': to_c, 'tipo': tipo, **impact})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_and_save_transfers(season_year: int = 2026, manual_text: str = "") -> tuple[bool, str]:
    """
    Salva i trasferimenti (da input manuale o scraping automatico).
    Restituisce (successo, messaggio).
    """
    df = None
    source = ""

    if manual_text.strip():
        df = parse_manual_transfers(manual_text)
        source = "manuale"

    if df is None or df.empty:
        df = fetch_transfermarkt_transfers(season_year)
        source = "Transfermarkt"

    if df is not None and not df.empty:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(TRANSFERS_CACHE, index=False)
        return True, f"{len(df)} trasferimenti salvati (fonte: {source})"
    return False, "Nessun trasferimento trovato. Inserisci i dati manualmente."


def load_cached_transfers() -> pd.DataFrame | None:
    if os.path.exists(TRANSFERS_CACHE):
        try:
            return pd.read_csv(TRANSFERS_CACHE)
        except Exception:
            return None
    return None


def merge_transfers_with_quotazioni(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge le colonne cambio_squadra, upgrade_squadra, tipo_prestito al DataFrame."""
    transfers = load_cached_transfers()
    df = df.copy()
    for col, default in _TRANSFER_DEFAULTS.items():
        df[col] = default

    if transfers is None or transfers.empty:
        return df

    tm_names = transfers['nome_tm'].tolist()

    try:
        from rapidfuzz import process, fuzz
        def find_match(nome: str) -> str | None:
            result = process.extractOne(
                str(nome).lower(),
                [n.lower() for n in tm_names],
                scorer=fuzz.token_set_ratio,
                score_cutoff=70,
            )
            if result:
                return tm_names[[n.lower() for n in tm_names].index(result[0])]
            return None
    except ImportError:
        def find_match(nome: str) -> str | None:
            surname = str(nome).lower().split()[-1]
            for tm in tm_names:
                if surname in tm.lower():
                    return tm
            return None

    transfer_cols = list(_TRANSFER_DEFAULTS.keys())
    for idx, row in df.iterrows():
        match = find_match(str(row['nome']))
        if match:
            t = transfers[transfers['nome_tm'] == match].iloc[0]
            player_role = str(row.get('ruolo', 'C'))
            for col in transfer_cols:
                if col in t.index:
                    df.at[idx, col] = t[col]
            # Ricalcola coach_bonus con il ruolo corretto del giocatore
            if 'a_squadra' in t.index:
                df.at[idx, 'coach_bonus'] = get_coach_tactical_bonus(t['a_squadra'], player_role)
                df.at[idx, 'coach_style'] = get_coach_style(t['a_squadra'])

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok, msg = fetch_and_save_transfers(2024)
    print(msg)
    if ok:
        print(load_cached_transfers().head())
