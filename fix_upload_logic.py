with open("app.py", "r") as f:
    content = f.read()

import re
pattern = r'            with st\.sidebar\.status\("Importazione Excel\.\.\."\) as s:.*?s\.update\(label=f"❌ \{e\}", state="error"\)'

new_block = """            with st.sidebar.status("Importazione Listone...") as s:
                try:
                    from importer import import_real_quotazioni
                    import tempfile
                    import os
                    ext = os.path.splitext(fc_excel_file.name)[1]
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f_tmp:
                        f_tmp.write(fc_excel_file.getvalue())
                        tmp_name = f_tmp.name
                    
                    new_df = import_real_quotazioni(tmp_name)
                    if new_df is not None and not new_df.empty:
                        QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
                        new_df.to_csv(QUOTAZIONI_PATH, index=False)
                        st.session_state["last_uploaded_file"] = file_id
                        _get_cached_predictions.clear()
                        s.update(label=f"✅ {len(new_df)} giocatori caricati!", state="complete", expanded=False)
                    else:
                        s.update(label="❌ Formato file non riconosciuto", state="error", expanded=False)
                except Exception as e:
                    s.update(label=f"❌ {e}", state="error")"""

content = re.sub(pattern, new_block, content, flags=re.DOTALL)

with open("app.py", "w") as f:
    f.write(content)
