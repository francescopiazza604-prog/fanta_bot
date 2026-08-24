import re

with open("app.py", "r") as f:
    content = f.read()

target = """    st.caption(
        "Carica qui il file Excel o CSV delle **Quotazioni** (o **Statistiche**) "
        "scaricato da fantacalcio.it"
    )
    fc_excel_file = st.file_uploader(
        "Carica File Listone", type=["xlsx", "xls", "csv"], key="fc_excel_upload"
    )"""

replacement = """    st.caption(
        "**Devi caricare SOLO il file delle QUOTAZIONI della nuova stagione!**\\n\\n"
        "Non ti serve scaricare le 'Statistiche' della scorsa stagione, perché "
        "l'Intelligenza Artificiale recupererà da sola i voti, i gol e gli infortuni "
        "dal web incrociandoli in automatico con il tuo listone nuovo."
    )
    fc_excel_file = st.file_uploader(
        "Seleziona il file (Excel o CSV)", type=["xlsx", "xls", "csv"], key="fc_excel_upload"
    )"""

content = content.replace(target, replacement)
with open("app.py", "w") as f:
    f.write(content)
