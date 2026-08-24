with open("app.py", "r") as f:
    content = f.read()

target = """    if fc_excel_file is not None:
        file_id = fc_excel_file.name + str(fc_excel_file.size)
        if st.session_state.get("last_uploaded_file") != file_id:
            with st.sidebar.status("Importazione Excel...") as s:
                try:
                    ok, msg = save_manual_excel(fc_excel_file.getvalue())
                    if ok:
                        from importer import import_real_quotazioni
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
                    s.update(label=msg if ok else f"❌ {msg}", state="complete" if ok else "error", expanded=False)
                except Exception as e:
                    s.update(label=f"❌ {e}", state="error")"""

replacement = """    if fc_excel_file is not None:
        file_id = fc_excel_file.name + str(fc_excel_file.size)
        if st.session_state.get("last_uploaded_file") != file_id:
            with st.sidebar.status("Importazione Excel...") as s:
                try:
                    ok, msg = save_manual_excel(fc_excel_file.getvalue())
                    if ok:
                        from importer import import_real_quotazioni
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
                            st.session_state["last_uploaded_file"] = file_id
                    s.update(label=msg if ok else f"❌ {msg}", state="complete" if ok else "error", expanded=False)
                except Exception as e:
                    s.update(label=f"❌ {e}", state="error")"""

import re
# First I need to revert my broken replacement or just replace it since it's already there with bad indentation
# Let's just find the whole block and rewrite it
pattern = r'    if fc_excel_file is not None:.*?(?=cached_stats = load_cached_stats\(\))'
new_block = """    if fc_excel_file is not None:
        file_id = fc_excel_file.name + str(fc_excel_file.size)
        if st.session_state.get("last_uploaded_file") != file_id:
            with st.sidebar.status("Importazione Excel...") as s:
                try:
                    ok, msg = save_manual_excel(fc_excel_file.getvalue())
                    if ok:
                        from importer import import_real_quotazioni
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
                            st.session_state["last_uploaded_file"] = file_id
                    s.update(label=msg if ok else f"❌ {msg}", state="complete" if ok else "error", expanded=False)
                except Exception as e:
                    s.update(label=f"❌ {e}", state="error")

"""
content = re.sub(pattern, new_block, content, flags=re.DOTALL)
with open("app.py", "w") as f:
    f.write(content)
