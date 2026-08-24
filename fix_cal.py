with open("calendar_analyzer.py", "r") as f:
    content = f.read()

target = """    def get_diff(row):
        sq = str(row.get('squadra', '')).strip()
        if not sq:
            return 3.0"""

replacement = """    # Trova colonna squadra
    sq_col = next((c for c in df.columns if c.lower() in ('squadra', 'sq', 'team')), 'squadra')

    def get_diff(row):
        sq = str(row.get(sq_col, '')).strip()
        if not sq or str(sq).lower() == 'nan':
            return 3.0"""

content = content.replace(target, replacement)
with open("calendar_analyzer.py", "w") as f:
    f.write(content)
