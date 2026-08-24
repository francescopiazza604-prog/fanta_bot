with open("app.py", "r") as f:
    content = f.read()

content = content.replace(
    "if uploaded_file is not None:\n    st.info(f\"✨ Dati caricati: **{uploaded_file.name}**\")",
    "if fc_excel_file is not None:\n    st.info(f\"✨ Dati caricati: **{fc_excel_file.name}**\")"
)

with open("app.py", "w") as f:
    f.write(content)
