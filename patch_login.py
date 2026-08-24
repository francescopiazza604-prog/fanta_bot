with open("app.py", "r") as f:
    content = f.read()

target = """                ok, msg = fetch_and_save_stats(
                    email=st.session_state.get("fc_email", ""),
                    password=st.session_state.get("fc_password", ""),
                )
                s.update(label=msg if ok else f"❌ {msg}", state="complete" if ok else "error", expanded=False)"""

replacement = """                ok, msg = fetch_and_save_stats(
                    email=st.session_state.get("fc_email", ""),
                    password=st.session_state.get("fc_password", ""),
                )
                if ok:
                    # Se scarica le stats, usale automaticamente anche come listone base
                    import shutil
                    from scraper_fantacalcio import CACHE_PATH
                    QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
                    shutil.copy(CACHE_PATH, QUOTAZIONI_PATH)
                    st.rerun()
                s.update(label=msg if ok else f"❌ {msg}", state="complete" if ok else "error", expanded=False)"""

content = content.replace(target, replacement)

with open("app.py", "w") as f:
    f.write(content)
