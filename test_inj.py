import requests
from bs4 import BeautifulSoup
url = "https://www.agentefantacalcio.it/feed/"
resp = requests.get(url)
soup = BeautifulSoup(resp.content, 'html.parser') # this may have issues with XML? wait! 'html.parser' on XML might not parse correctly. Maybe 'xml' is needed!
items = soup.find_all('item')
print(f"Found {len(items)} items")
for item in items[:5]:
    print("Title:", item.find('title').get_text() if item.find('title') else None)
