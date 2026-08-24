with open("app.py", "r") as f:
    content = f.read()

cal_code = """
# ── Tab: Calendario e Portieri ────────────────────────────────────────────────
with tab_cal:
    st.subheader("📅 Analisi Calendario e Incroci Portieri")
    st.markdown("Questa sezione analizza il calendario della Serie A per trovare la difficoltà delle prossime partite per ogni squadra e calcolare le migliori coppie di portieri da acquistare all'asta.")
    
    if data_df is None:
        st.warning("Carica un file di quotazioni per vedere i dati.")
    else:
        from calendar_analyzer import fetch_calendar, get_all_teams_difficulty, find_best_gk_pairings
        
        with st.spinner("Scaricamento calendario in corso..."):
            calendar = fetch_calendar()
            
        if not calendar:
            st.error("Impossibile scaricare il calendario in questo momento.")
        else:
            col_cal1, col_cal2 = st.columns([1, 2])
            
            with col_cal1:
                st.markdown("### 📊 Difficoltà Squadre")
                num_matches = st.slider("Numero di prossime partite da analizzare:", min_value=1, max_value=10, value=5)
                start_match = st.number_input("Partita di partenza (Giornata):", min_value=1, max_value=38, value=1)
                
                diffs = get_all_teams_difficulty(calendar, from_matchday=start_match, num_matches=num_matches)
                diff_df = pd.DataFrame(list(diffs.items()), columns=["Squadra", "Difficoltà (1-5)"])
                diff_df = diff_df.sort_values(by="Difficoltà (1-5)").reset_index(drop=True)
                
                def color_diff(val):
                    if val <= 2.5: return 'background-color: #d4edda; color: black;' # verde
                    if val >= 3.8: return 'background-color: #f8d7da; color: black;' # rosso
                    return ''
                    
                st.dataframe(diff_df.style.map(color_diff, subset=['Difficoltà (1-5)']), use_container_width=True)
                st.caption("Verde = Calendario Facile, Rosso = Calendario Difficile")
                
            with col_cal2:
                st.markdown("### 🧤 Migliori Coppie Portieri")
                gk_matches = st.slider("Analizza incroci per le prossime N giornate:", min_value=1, max_value=38, value=38)
                
                squadre_presenti = data_df['squadra'].dropna().unique().tolist()
                
                # Assicuriamoci che i nomi matchino
                squadre_pulite = []
                from calendar_analyzer import TEAM_STRENGTH
                for sq in squadre_presenti:
                    if sq in TEAM_STRENGTH:
                        squadre_pulite.append(sq)
                
                if not squadre_pulite:
                    squadre_pulite = list(TEAM_STRENGTH.keys())
                    
                pairings = find_best_gk_pairings(calendar, squadre_pulite, from_matchday=start_match, num_matches=gk_matches)
                pair_df = pd.DataFrame(pairings)
                pair_df = pair_df.rename(columns={
                    "team1": "Portiere 1 (Squadra)",
                    "team2": "Portiere 2 (Squadra)",
                    "avg_difficulty": "Difficoltà Incrociata",
                    "alternation_pct": "% Alternanza Casa/Trasferta"
                })
                
                st.dataframe(pair_df.head(20), use_container_width=True)
                st.caption("Mostra le migliori 20 coppie in base alla difficoltà della partita più facile in ogni giornata. Una % Alternanza alta indica che quasi sempre uno gioca in casa e l'altro in trasferta.")

"""

content = content.replace("# ── Tab 6: Backtest IA ────────────────────────────────────────────────────────", cal_code + "\n# ── Tab 6: Backtest IA ────────────────────────────────────────────────────────")

with open("app.py", "w") as f:
    f.write(content)

