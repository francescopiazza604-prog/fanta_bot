"""
Notizie calciomercato da Fantacalcio.it — sezione /calciomercato.
La pagina ha HTML statico (non è una SPA come le statistiche), quindi BeautifulSoup funziona.
"""

import re
import time
import logging

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL  = "https://www.fantacalcio.it"
NEWS_URL  = f"{BASE_URL}/calciomercato"
MAX_NEWS  = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "it-IT,it;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": BASE_URL,
}

# Pattern URL articolo: /calciomercato/GG_MM_YYYY/slug-id
_ARTICLE_RE = re.compile(r"^/calciomercato/\d{2}_\d{2}_\d{4}/")


def fetch_latest_fanta_news(max_items: int = MAX_NEWS) -> str:
    """
    Scarica le ultime notizie dalla sezione Calciomercato di Fantacalcio.it.
    Ritorna una stringa multi-riga pronta per la text area.
    """
    try:
        resp = requests.get(NEWS_URL, headers=HEADERS, timeout=12)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Fantacalcio.it calciomercato non raggiungibile: {e}")
        return _fallback()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Tutti i link che puntano a un articolo calciomercato con data nel path
    links = soup.find_all("a", href=_ARTICLE_RE)

    seen: set[str] = set()
    items: list[str] = []

    for a in links:
        href = a.get("href", "")
        if href in seen:
            continue
        seen.add(href)

        # Titolo: priorità img alt → testo diretto del link
        img = a.find("img")
        title = ""
        if img and img.get("alt"):
            title = img["alt"].strip()
        if not title:
            title = a.get_text(" ", strip=True)
            # Rimuovi eventuale sottotitolo appiccicato
            title = title.split("\n")[0].strip()

        # Rimuovi prefissi categoria tipo "Copertina: ", "Esclusiva: "
        for prefix in ("Copertina: ", "Esclusiva: ", "ESCLUSIVA: ", "UFFICIALE - "):
            if title.startswith(prefix):
                title = title[len(prefix):]

        if title.startswith("UFFICIALE"):
            title = "🔴 " + title

        if len(title) < 8:
            continue

        # Sottotitolo (tag <p> dentro il link)
        p = a.find("p")
        subtitle = p.get_text(strip=True) if p else ""

        line = f"• {title}"
        if subtitle and subtitle.lower() != title.lower():
            line += f": {subtitle}"

        items.append(line)
        if len(items) >= max_items:
            break

    if not items:
        logger.warning("Nessun articolo trovato nella pagina calciomercato")
        return _fallback()

    return "\n\n".join(items)


def _fallback() -> str:
    return (
        "⚠️ Fantacalcio.it non raggiungibile — notizie di esempio:\n\n"
        "• Ultim'ora: infortuni e rientri per la prossima giornata\n"
        "• Mercato: trattative in corso per l'estate\n"
        "• Rigoristi: aggiornamenti dopo l'ultima giornata"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(fetch_latest_fanta_news())
