with open("app.py", "r") as f:
    content = f.read()

target = """            with open(os.path.join(DATA_DIR, "fbref_stats_cache.csv"), "w", encoding="utf-8") as fb:
                fb.write(fbref_csv_text)
            st.success("✅ Statistiche FBref salvate con successo!")"""

replacement = """            with open(os.path.join(DATA_DIR, "fbref_stats_cache.csv"), "w", encoding="utf-8") as fb:
                fb.write(fbref_csv_text)
            _get_cached_predictions.clear()
            st.success("✅ Statistiche FBref salvate con successo!")"""

content = content.replace(target, replacement)
with open("app.py", "w") as f:
    f.write(content)
