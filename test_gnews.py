import requests
from bs4 import BeautifulSoup
url = "https://news.google.com/rss/search?q=fantacalcio+infortunio&hl=it&gl=IT&ceid=IT:it"
resp = requests.get(url)
soup = BeautifulSoup(resp.content, 'xml')
items = soup.find_all('item')
print(f"Found {len(items)} items")
for item in items[:10]:
    print("-", item.title.text)
