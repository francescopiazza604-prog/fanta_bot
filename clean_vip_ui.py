with open("app.py", "r") as f:
    content = f.read()

target = """    with col_vip2:
        st.markdown("#### Aggiungi override tattico")
        vip_nome    = st.text_input("Nome giocatore", key="vip_add_nome", placeholder="es. Nico Paz")
        vip_pos     = st.selectbox("Posizione tattica", [
            "QUINTO_ATT", "QUINTO_DUTTILE", "ALA", "ALA_TREQUARTISTA",
            "TREQUARTISTA", "MEZZALA_ATT", "MEZZALA_DEF", "REGISTA",
        ], key="vip_pos")
        vip_bonus   = st.slider("Bonus moltiplicatore", 1.05, 1.50, 1.25, 0.01, key="vip_bonus",
                                help="1.0=nessun bonus, 1.5=+50% bonus atteso")
        vip_note    = st.text_input("Note (opzionale)", key="vip_note")
        if st.button("💾 Salva override tattico"):
            try:
                from vip import add_tactical_override
                ok = add_tactical_override(vip_nome, vip_pos, vip_bonus, vip_note)
                if ok:
                    st.success(f"✅ {vip_nome} salvato come {vip_pos} ({vip_bonus:.2f}×)")
                else:
                    st.error("Errore salvataggio")
            except Exception as e:
                st.error(f"Errore: {e}")

        st.markdown("#### Aggiungi giovane promessa")
        vip_nome_y  = st.text_input("Nome giocatore", key="vip_youth_nome", placeholder="es. Camarda")
        vip_anno    = st.number_input("Anno di nascita", 2000, 2010, 2004, 1, key="vip_anno")
        vip_trend   = st.slider("Trend minutaggio", 0.5, 2.0, 1.3, 0.05, key="vip_trend",
                                help="1.0=stabile, >1.0=crescente, <1.0=calante")
        vip_note_y  = st.text_input("Note (opzionale)", key="vip_note_youth")
        if st.button("💾 Salva giovane promessa"):
            try:
                from vip import add_youth_player
                ok = add_youth_player(vip_nome_y, int(vip_anno), vip_trend, vip_note_y)
                if ok:
                    st.success(f"✅ {vip_nome_y} ({CURRENT_YEAR - int(vip_anno)} anni, trend {vip_trend:.2f}) salvato")
                else:
                    st.error("Errore salvataggio")
            except Exception as e:
                st.error(f"Errore: {e}")

        # Riepilogo config attuale
        st.markdown("---")
        st.markdown("#### Config attuale")
        try:
            from vip import get_vip_config_summary
            summary = get_vip_config_summary()
            st.caption(
                f"Versione: {summary['version']} · "
                f"{summary['n_tactical']} override tattici · "
                f"{summary['n_youth']} giovani promesse"
            )
        except Exception:
            st.caption("Config non disponibile")"""

replacement = """    with col_vip2:
        st.success("✨ **Motore VIP Completamente Automatico**")
        st.markdown(
            "Il VIP Radar ora scansiona l'intero database in totale autonomia. "
            "L'algoritmo rileva da solo i giovani talenti U22 incrociando l'età con i trend di minutaggio, "
            "e deduce la posizione tattica reale (es. *Quinto d'attacco* per i difensori goleador) "
            "utilizzando l'inferenza statistica sui tassi di Gol e Assist P90.\\n\\n"
            "Non c'è più bisogno di alcun inserimento manuale!"
        )"""

content = content.replace(target, replacement)

with open("app.py", "w") as f:
    f.write(content)
