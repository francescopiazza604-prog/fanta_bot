with open("app.py", "r") as f:
    content = f.read()

target = """                st.write("---")
                with st.expander("🤖 📋 Leggi i Report Scout dell'IA per la rosa scelta"):
                    from predict import build_player_explanation
                    for idx, row in team.sort_values(by=['ruolo', 'previsione_ia'], ascending=[False, False]).iterrows():
                        st.markdown(build_player_explanation(row))
                        st.write("---")
                with col2:"""

replacement = """                with col2:"""

content = content.replace(target, replacement)

target2 = """                    st.bar_chart(team.groupby("ruolo")["costo_iniziale"].sum())"""

replacement2 = """                    st.bar_chart(team.groupby("ruolo")["costo_iniziale"].sum())
                
                st.write("---")
                with st.expander("🤖 📋 Leggi i Report Scout dell'IA per i 25 scelti"):
                    from predict import build_player_explanation
                    for idx, row in team.sort_values(by=['ruolo', 'previsione_ia'], ascending=[False, False]).iterrows():
                        st.markdown(build_player_explanation(row))
                        st.write("---")"""

content = content.replace(target2, replacement2)

with open("app.py", "w") as f:
    f.write(content)
