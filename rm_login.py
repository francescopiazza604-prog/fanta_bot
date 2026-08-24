import re

with open("app.py", "r") as f:
    content = f.read()

# Trova tutto tra "with st.sidebar.expander(\"🔑 SCARICA" e "# Opzione B:"
pattern = r'with st\.sidebar\.expander\("🔑 SCARICA.*?(?=# Opzione B: upload manuale Excel)'

# Rimuovi il blocco
new_content = re.sub(pattern, '', content, flags=re.DOTALL)

with open("app.py", "w") as f:
    f.write(new_content)
