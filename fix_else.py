with open("app.py", "r") as f:
    content = f.read()

target = """else:
    if os.path.exists(QUOTAZIONI_PATH):
        try:
            data_df = pd.read_csv(QUOTAZIONI_PATH)"""

replacement = """QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
data_df = None
if os.path.exists(QUOTAZIONI_PATH):
    try:
        data_df = pd.read_csv(QUOTAZIONI_PATH)"""

content = content.replace(target, replacement)
with open("app.py", "w") as f:
    f.write(content)
