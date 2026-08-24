import re
with open("app.py", "r") as f:
    content = f.read()

# Replace the line exactly
old_line = "squadre_presenti = data_df.get('squadra', data_df.get('Squadra', data_df.get('Sq', []))).dropna().unique().tolist()"
new_line = "                # Trova colonna squadra\n                sq_col = next((c for c in data_df.columns if c.lower() in ('squadra', 'sq', 'team')), None)\n                squadre_presenti = data_df[sq_col].dropna().unique().tolist() if sq_col else []"

content = content.replace(old_line, new_line)

with open("app.py", "w") as f:
    f.write(content)
