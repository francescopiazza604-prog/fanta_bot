import pandas as pd
from predict import build_player_explanation
row = pd.Series({
    'ruolo': 'A', 'nome': 'Kvaratskhelia', 'costo_iniziale': 180, 'previsione_ia': 7.8,
    'fanta_media': 7.5, 'eta': 25, 'gol_pg': 0.3, 'assist_pg': 0.25, 'titolarita_pct': 0.9,
    'cambio_squadra': 1, 'upgrade_squadra': 0, 'coach_bonus': 0.2,
    'forma_recente_score': 0.4, 'vip_total': 0.1, 'calendar_diff': 2.5, 'injury_modifier': 1.0,
    '_temporal_model': True
})
print(build_player_explanation(row))
