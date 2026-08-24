import requests
from bs4 import BeautifulSoup
url = "https://www.pianetafanta.it/Giocatori-Infortunati-Squalificati-Diffidati.asp"
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/112.0'}
r = requests.get(url, headers=headers)
print(r.status_code)
if "Scamacca" in r.text or "Scalvini" in r.text:
    print("Found injured players!")
else:
    print("Not found")
