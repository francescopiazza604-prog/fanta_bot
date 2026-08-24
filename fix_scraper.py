import re
with open("scraper_fantacalcio.py", "r") as f:
    content = f.read()

target = """STATS_API = f"{BASE_URL}/api/v1/Excel/stats/20/1"   # 20 = Serie A, 1 = stagione corrente"""
replacement = """STATS_API = f"{BASE_URL}/api/v1/Excel/stats/21/1"   # 21 = Serie A
PRICES_API = f"{BASE_URL}/api/v1/Excel/prices/21/1\""""
content = content.replace(target, replacement)

target_dl = """def _download_excel(session: requests.Session) -> pd.DataFrame | None:
    \"\"\"Scarica il file Excel statistiche dall'API autenticata.\"\"\"
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
        return None"""

replacement_dl = """def _download_excel(session: requests.Session) -> pd.DataFrame | None:
    \"\"\"Scarica il file Excel statistiche dall'API autenticata, fallback su Prices.\"\"\"
    try:
        resp = session.get(STATS_API, headers=HEADERS, timeout=30)
        if resp.status_code == 401 or resp.status_code == 403:
            logger.info("Account non Premium, fallback al file Quotazioni base.")
            resp = session.get(PRICES_API, headers=HEADERS, timeout=30)
            
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
        return None"""
content = content.replace(target_dl, replacement_dl)

with open("scraper_fantacalcio.py", "w") as f:
    f.write(content)
