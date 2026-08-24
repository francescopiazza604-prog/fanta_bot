import re
with open("app.py", "r") as f:
    content = f.read()

content = re.sub(
    r'strategy_label = st\.sidebar\.selectbox\(.*?\)\n',
    '''strategy_label = st.sidebar.selectbox(
    "🎯 Strategia Asta",
    ["🏆 Listone Master (Equilibrata)", "⚔️ Aggressiva (Trazione Anteriore)", "💎 Moneyball (VIP & Sottovalutati)", "🚀 Sprint Iniziale (Focus Calendario)"],
    help=(
        "Master: cerca i migliori 11 titolari e riserve a 1 cr equilibrando il budget.\\n"
        "Aggressiva: budget quasi tutto su centrocampo e attacco, difesa low cost.\\n"
        "Moneyball: cerca giocatori con statistiche tattiche nascoste ad alto potenziale.\\n"
        "Sprint Iniziale: massimizza i giocatori con calendario super-facile."
    ),
)\n''',
    content, flags=re.DOTALL
)

with open("app.py", "w") as f:
    f.write(content)
