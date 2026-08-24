with open("app.py", "r") as f:
    content = f.read()

content = content.replace(
    "def _get_cached_predictions(_df, strat):",
    "def _get_cached_predictions(df, strat):"
)
content = content.replace(
    "if _df is not None:\n        _df.to_csv(INPUT_TEMP, index=False)",
    "if df is not None:\n        df.to_csv(INPUT_TEMP, index=False)"
)
with open("app.py", "w") as f:
    f.write(content)
