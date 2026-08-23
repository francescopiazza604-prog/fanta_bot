import requests
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

url = "https://www.fantacalcio.it/notizie-fantacalcio"
try:
    response = requests.get(url, headers=headers, timeout=10)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Try finding all links that look like news titles
        titles = soup.select('h3.title, a.title, .news-title')
        print(f"Found {len(titles)} titles with selectors")
        for t in titles[:5]:
            print(f"- {t.get_text().strip()}")
            
        # Fallback: any h3/h2
        if not titles:
            all_h = soup.find_all(['h2', 'h3'])
            print(f"Found {len(all_h)} generic h2/h3")
            for t in all_h[:5]:
                print(f"- {t.get_text().strip()}")
except Exception as e:
    print(f"Error: {e}")
