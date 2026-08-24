with open("app.py", "r") as f:
    content = f.read()

target = """                if ok:
                    import shutil
                    from scraper_fantacalcio import CACHE_PATH
                    QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
                    shutil.copy(CACHE_PATH, QUOTAZIONI_PATH)
                    st.rerun()"""

replacement = """                if ok:
                    from importer import import_real_quotazioni
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f_tmp:
                        f_tmp.write(fc_excel_file.getvalue())
                        tmp_name = f_tmp.name
                    new_df = import_real_quotazioni(tmp_name)
                    if new_df is not None:
                        QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
                        new_df.to_csv(QUOTAZIONI_PATH, index=False)
                    st.rerun()"""

content = content.replace(target, replacement)
with open("app.py", "w") as f:
    f.write(content)
