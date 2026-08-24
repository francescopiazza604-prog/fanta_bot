with open("app.py", "r") as f:
    content = f.read()

target = """                        if new_df is not None:
                            QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
                            new_df.to_csv(QUOTAZIONI_PATH, index=False)
                            st.session_state["last_uploaded_file"] = file_id
                    s.update(label=msg if ok else f"❌ {msg}", state="complete" if ok else "error", expanded=False)"""

replacement = """                        if new_df is not None:
                            QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
                            new_df.to_csv(QUOTAZIONI_PATH, index=False)
                            st.session_state["last_uploaded_file"] = file_id
                            _get_cached_predictions.clear()
                    s.update(label=msg if ok else f"❌ {msg}", state="complete" if ok else "error", expanded=False)"""

content = content.replace(target, replacement)
with open("app.py", "w") as f:
    f.write(content)
