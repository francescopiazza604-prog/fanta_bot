with open("app.py", "r") as f:
    content = f.read()

content = content.replace(
    '# Opzione B: upload manuale Excel (raccomandato se il login automatico non funziona)',
    ''
)

content = content.replace(
    'with st.sidebar.expander("📂 Upload Excel manuale (alternativa)"):',
    'with st.sidebar.expander("📥 Importa Statistiche da Fantacalcio.it", expanded=True):'
)

with open("app.py", "w") as f:
    f.write(content)
