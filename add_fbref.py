import re

with open("app.py", "r") as f:
    content = f.read()

target = """# ── Mercato ───────────────────────────────────────────────────────────────────"""

replacement = """# ── Dati FBref (Statistiche Storiche) ─────────────────────────────────────────
st.sidebar.subheader("📈 Statistiche Avanzate (FBref)")
with st.sidebar.expander("Importa Statistiche FBref (Opzionale)"):
    st.caption(
        "Essendo bloccato da Cloudflare, devi incollare i dati FBref manualmente per far "
        "imparare all'IA i dati della scorsa stagione:\\n"
        "1. Apri [fbref.com/it/comps/11/stats/Serie-A-Stats](https://fbref.com/it/comps/11/stats/Serie-A-Stats)\\n"
        "2. Scorri fino alla tabella 'Standard Stats'\\n"
        "3. Clicca 'Share & Export' → 'Get table as CSV'\\n"
        "4. Incolla il testo qui sotto:"
    )
    fbref_csv_text = st.text_area("Incolla CSV FBref", height=100)
    if st.button("💾 Salva FBref Cache"):
        if fbref_csv_text.strip():
            with open(os.path.join(DATA_DIR, "fbref_stats_cache.csv"), "w", encoding="utf-8") as fb:
                fb.write(fbref_csv_text)
            st.success("✅ Statistiche FBref salvate!")
            st.rerun()
        else:
            st.error("Incolla il testo prima di salvare.")

# ── Mercato ───────────────────────────────────────────────────────────────────"""

new_content = content.replace(target, replacement)
with open("app.py", "w") as f:
    f.write(new_content)
