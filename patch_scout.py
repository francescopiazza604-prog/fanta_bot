with open("app.py", "r") as f:
    content = f.read()

target = """                    st.metric("⚽ Punteggio Previsto / giornata", punteggio_g)
                    st.metric("📅 Proiezione Stagionale (38 g.)", punteggio_s)
                    st.metric("💰 Costo Totale", int(team["costo_iniziale"].sum()))"""

replacement = """                    st.metric("⚽ Punteggio Previsto / giornata", punteggio_g)
                    st.metric("📅 Proiezione Stagionale (38 g.)", punteggio_s)
                    st.metric("💰 Costo Totale", int(team["costo_iniziale"].sum()))
                
                st.write("---")
                with st.expander("🤖 📋 Leggi i Report Scout dell'IA per la rosa scelta"):
                    from predict import build_player_explanation
                    for idx, row in team.sort_values(by=['ruolo', 'previsione_ia'], ascending=[False, False]).iterrows():
                        st.markdown(build_player_explanation(row))
                        st.write("---")"""

content = content.replace(target, replacement)

with open("app.py", "w") as f:
    f.write(content)
