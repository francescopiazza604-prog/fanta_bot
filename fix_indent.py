import re

with open("app.py", "r") as f:
    content = f.read()

target = """QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
data_df = None
if os.path.exists(QUOTAZIONI_PATH):
    try:
        data_df = pd.read_csv(QUOTAZIONI_PATH)
            st.sidebar.caption(f"📋 Dataset Attivo: Listone Corrente ({len(data_df)} giocatori)")
        except Exception as e:
            st.sidebar.error(f"Errore caricamento cache: {e}")
    elif os.path.exists(DEFAULT_DATA):
        try:
            data_df = pd.read_csv(DEFAULT_DATA)
            st.sidebar.caption("📋 Dataset Attivo: Demo 23/24")
        except Exception as e:
            st.sidebar.error(f"Errore: {e}")
    else:
        st.sidebar.warning("Nessun dato. Carica un file.")"""

replacement = """QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
data_df = None

if os.path.exists(QUOTAZIONI_PATH):
    try:
        data_df = pd.read_csv(QUOTAZIONI_PATH)
        st.sidebar.caption(f"📋 Dataset Attivo: Listone Corrente ({len(data_df)} giocatori)")
    except Exception as e:
        st.sidebar.error(f"Errore caricamento cache: {e}")
elif os.path.exists(DEFAULT_DATA):
    try:
        data_df = pd.read_csv(DEFAULT_DATA)
        st.sidebar.caption("📋 Dataset Attivo: Demo 23/24")
    except Exception as e:
        st.sidebar.error(f"Errore: {e}")
else:
    st.sidebar.warning("Nessun dato. Carica un file.")"""

content = content.replace(target, replacement)
with open("app.py", "w") as f:
    f.write(content)
