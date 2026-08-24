import re
with open("optimizer.py", "r") as f:
    content = f.read()

target = """BUDGET_BANDS: dict[str, dict[str, tuple[float, float]]] = {
    # Con budget=500:
    #   Aggressiva C: min=150cr, max=210cr per 8 medi → avg 19-26cr/medi
    #   Aggressiva A: min=140cr, max=240cr per 6 att  → avg 23-40cr/att
    #   (MID aumentato da 22-33% → 30-42% per forzare qualità in mediana)
    'aggressiva': {
        'P': (0.02, 0.09),
        'D': (0.12, 0.20),
        'C': (0.30, 0.42),
        'A': (0.28, 0.48),
    },
    'conservativa': {
        'P': (0.04, 0.11),
        'D': (0.17, 0.26),
        'C': (0.30, 0.40),
        'A': (0.26, 0.44),
    },
    'scommesse': {
        'P': (0.02, 0.07),   # GK economico
        'D': (0.08, 0.16),   # DEF cheap
        'C': (0.23, 0.38),
        'A': (0.24, 0.42),   # cap basso: no star da 60cr
    },
    # VIP pesante: stesse band dell'aggressiva, ma il VIP guida la scelta dei giocatori
    'vip_pesante': {
        'P': (0.02, 0.09),
        'D': (0.13, 0.21),
        'C': (0.25, 0.40),
        'A': (0.28, 0.45),
    },
}"""

replacement = """BUDGET_BANDS: dict[str, dict[str, tuple[float, float]]] = {
    'master': {
        'P': (0.04, 0.10),
        'D': (0.15, 0.22),
        'C': (0.30, 0.40),
        'A': (0.28, 0.45),
    },
    'aggressiva': {
        'P': (0.02, 0.07),
        'D': (0.10, 0.18),
        'C': (0.32, 0.45),
        'A': (0.30, 0.55),
    },
    'moneyball': {
        'P': (0.03, 0.08),
        'D': (0.12, 0.20),
        'C': (0.25, 0.40),
        'A': (0.28, 0.45),
    },
    'sprint_calendario': {
        'P': (0.04, 0.12),
        'D': (0.15, 0.25),
        'C': (0.30, 0.40),
        'A': (0.25, 0.45),
    },
}"""
content = content.replace(target, replacement)

content = content.replace("bands   = BUDGET_BANDS.get(strategy, BUDGET_BANDS['conservativa'])", "bands   = BUDGET_BANDS.get(strategy, BUDGET_BANDS['master'])")
content = content.replace("min_avg = MIN_AVG_COST.get(strategy, MIN_AVG_COST['conservativa'])", "min_avg = MIN_AVG_COST.get(strategy, MIN_AVG_COST.get('master', {}))")

with open("optimizer.py", "w") as f:
    f.write(content)
