import requests
from bs4 import BeautifulSoup
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
resp = requests.get('https://www.fantacalcio.it/infortunati-e-squalificati', headers=headers)
print(resp.status_code)
if resp.status_code == 200:
    soup = BeautifulSoup(resp.content, 'html.parser')
    players = soup.select('.player-name')
    print(f"Found {len(players)} players")
    for p in players[:5]:
        print(p.text.strip())
