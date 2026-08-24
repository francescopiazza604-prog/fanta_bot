with open("app.py", "r") as f:
    content = f.read()

# Fix 1: Infinite loop on file upload
target1 = """                    if new_df is not None:
                        QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
                        new_df.to_csv(QUOTAZIONI_PATH, index=False)
                    st.rerun()"""
replacement1 = """                    if new_df is not None:
                        QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
                        new_df.to_csv(QUOTAZIONI_PATH, index=False)"""
content = content.replace(target1, replacement1)

# Fix 2: FBref success message disappearing (just don't rerun)
target2 = """            with open(os.path.join(DATA_DIR, "fbref_stats_cache.csv"), "w", encoding="utf-8") as fb:
                fb.write(fbref_csv_text)
            st.success("✅ Statistiche FBref salvate!")
            st.rerun()"""
replacement2 = """            with open(os.path.join(DATA_DIR, "fbref_stats_cache.csv"), "w", encoding="utf-8") as fb:
                fb.write(fbref_csv_text)
            st.success("✅ Statistiche FBref salvate con successo!")"""
content = content.replace(target2, replacement2)

with open("app.py", "w") as f:
    f.write(content)
