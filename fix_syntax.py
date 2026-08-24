with open("app.py", "r") as f:
    content = f.read()

target = """    help=(
        "Master: cerca i migliori 11 titolari e riserve a 1 cr equilibrando il budget.
"
        "Aggressiva: budget quasi tutto su centrocampo e attacco, difesa low cost.
"
        "Moneyball: cerca giocatori con statistiche tattiche nascoste ad alto potenziale.
"
        "Sprint Iniziale: massimizza i giocatori con calendario super-facile."
    ),"""

replacement = """    help=(
        "Master: cerca i migliori 11 titolari e riserve a 1 cr equilibrando il budget.\\n"
        "Aggressiva: budget quasi tutto su centrocampo e attacco, difesa low cost.\\n"
        "Moneyball: cerca giocatori con statistiche tattiche nascoste ad alto potenziale.\\n"
        "Sprint Iniziale: massimizza i giocatori con calendario super-facile."
    ),"""

content = content.replace(target, replacement)

with open("app.py", "w") as f:
    f.write(content)
