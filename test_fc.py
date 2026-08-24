import cloudscraper
import json

s = cloudscraper.create_scraper()
r = s.get("https://www.fantacalcio.it/api/v1/Statistiche/GiocatoriByRuolo?ruolo=A&stagione=1")
print(r.status_code)
if r.status_code == 200:
    print(r.text[:200])
