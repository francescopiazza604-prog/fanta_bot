with open("optimizer.py", "r") as f:
    content = f.read()

import re

# Rimuovi il vecchio BUDGET_BANDS e inserisci il nuovo
pattern = r"BUDGET_BANDS: dict\[str, dict\[str, tuple\[float, float\]\]\] = \{.*?\}(?=\n\n# Costruisce vincoli)"

new_bands = """BUDGET_BANDS: dict[str, dict[str, tuple[float, float]]] = {
    'master': {
        'P': (0.04, 0.11),
        'D': (0.17, 0.26),
        'C': (0.30, 0.40),
        'A': (0.26, 0.44),
    },
    'aggressiva': {
        'P': (0.02, 0.09),
        'D': (0.12, 0.20),
        'C': (0.30, 0.42),
        'A': (0.28, 0.48),
    },
    'moneyball': {
        'P': (0.02, 0.07),
        'D': (0.13, 0.21),
        'C': (0.28, 0.42),
        'A': (0.28, 0.48),
    },
    'sprint_calendario': {
        'P': (0.04, 0.11),
        'D': (0.15, 0.24),
        'C': (0.28, 0.40),
        'A': (0.28, 0.46),
    },
}"""

content = re.sub(pattern, new_bands, content, flags=re.DOTALL)
with open("optimizer.py", "w") as f:
    f.write(content)
