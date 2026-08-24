import re

with open("app.py", "r") as f:
    content = f.read()

# Rimuovi il blocco ridondante in basso
pattern = r"# ── File uploader \+ caricamento dati ──+[\s\S]*?(?=else:\s+if os\.path\.exists\(QUOTAZIONI_PATH\):)"
new_content = re.sub(pattern, '', content)

# Ora modifica il primo blocco per supportare sia CSV che Excel e chiamarlo chiaramente "Carica Listone / Statistiche"
target1 = """with st.sidebar.expander("📥 Importa Statistiche da Fantacalcio.it", expanded=True):"""
replacement1 = """with st.sidebar.expander("📥 CARICA LISTONE FANTACALCIO", expanded=True):"""
new_content = new_content.replace(target1, replacement1)

target2 = """    st.caption(
        "1. Vai su **fantacalcio.it** nel tuo browser e accedi\\n"
        "2. Apri la pagina **Statistiche Serie A**\\n"
        "3. Clicca il pulsante **Excel** per scaricare il file\\n"
        "4. Carica il file qui sotto"
    )
    fc_excel_file = st.file_uploader(
        "Carica Excel Fantacalcio.it", type=["xlsx", "xls"], key="fc_excel_upload"
    )"""

replacement2 = """    st.caption(
        "Carica qui il file Excel o CSV delle **Quotazioni** (o **Statistiche**) "
        "scaricato da fantacalcio.it"
    )
    fc_excel_file = st.file_uploader(
        "Carica File Listone", type=["xlsx", "xls", "csv"], key="fc_excel_upload"
    )"""
new_content = new_content.replace(target2, replacement2)

# Modifichiamo anche come viene gestito il file. Se è un CSV, tmp_name deve avere suffisso .csv, altrimenti .xlsx.
target3 = """                    from importer import import_real_quotazioni
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f_tmp:
                        f_tmp.write(fc_excel_file.getvalue())
                        tmp_name = f_tmp.name
                    new_df = import_real_quotazioni(tmp_name)
                    if new_df is not None:
                        QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
                        new_df.to_csv(QUOTAZIONI_PATH, index=False)
                    st.rerun()"""

replacement3 = """                    from importer import import_real_quotazioni
                    import tempfile
                    import os
                    ext = os.path.splitext(fc_excel_file.name)[1]
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f_tmp:
                        f_tmp.write(fc_excel_file.getvalue())
                        tmp_name = f_tmp.name
                    new_df = import_real_quotazioni(tmp_name)
                    if new_df is not None:
                        QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
                        new_df.to_csv(QUOTAZIONI_PATH, index=False)
                    st.rerun()"""
new_content = new_content.replace(target3, replacement3)

with open("app.py", "w") as f:
    f.write(new_content)
