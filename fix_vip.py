import re

with open("vip.py", "r") as f:
    content = f.read()

# Fix compute_vip call
content = content.replace(
    'youth    = _compute_youth_factor(str(row.get("nome", "")), cfg)',
    'youth    = _compute_youth_factor(row, cfg)'
)

# Replace _compute_youth_factor definition
old_youth = '''def _compute_youth_factor(nome: str, cfg: dict) -> float:
    """
    Coefficiente di crescita esponenziale per giocatori under-22.

    Formula: bonus = exp((YOUTH_CUTOFF_AGE - age) / YOUTH_TAU) - 1
      age=22 → bonus=0     (zero bonus al confine)
      age=20 → bonus=0.28  (+28% * trend)
      age=18 → bonus=0.65  (+65% * trend → clippato a 40%)
      age=17 → bonus=0.87  (+87% * trend → clippato a 40%)

    Il trend (da config) moltiplica il bonus:
      trend=1.0 → stabile (minutaggio fisso)
      trend=1.5 → forte crescita di minutaggio → segnale di esplosione imminente
      trend=0.8 → minutaggio calante → riduce il bonus
    """
    youth_players: dict = cfg.get("youth_players", {})
    if not youth_players:
        return 0.0

    nome_clean = str(nome).strip()
    player_data = youth_players.get(nome_clean)

    if player_data is None:
        # Fallback cognome
        cognome = nome_clean.split()[-1].lower()
        for key, val in youth_players.items():
            if not key.startswith("_") and cognome == key.split()[-1].lower():
                player_data = val
                break

    if player_data is None:
        return 0.0

    birth_year    = int(player_data.get("birth_year", CURRENT_YEAR - 25))
    minutes_trend = float(player_data.get("minutes_trend", 1.0))
    age           = CURRENT_YEAR - birth_year

    if age > YOUTH_CUTOFF_AGE:
        return 0.0

    # Crescita esponenziale: più giovane = più potenziale non ancora prezzato dal mercato
    age_bonus = math.exp((YOUTH_CUTOFF_AGE - age) / YOUTH_TAU) - 1.0

    # Trend moltiplica: minutaggio crescente accelera la realizzazione del potenziale
    # trend < 1: minutaggio che cala → penalizza anche il bonus età
    raw_bonus = age_bonus * minutes_trend * 0.55

    # Cap: max +40% (evita che un 17enne con 0 presenze vada a strafare)
    return float(np.clip(raw_bonus, 0.0, 0.40))'''

new_youth = '''def _compute_youth_factor(row, cfg: dict) -> float:
    """
    Coefficiente di crescita esponenziale per giocatori under-22 (COMPLETAMENTE AUTOMATICO).
    """
    age = float(row.get("eta", 26) or 26)
    
    if age > YOUTH_CUTOFF_AGE:
        return 0.0
        
    # Stima del trend in base alla forma recente o presenze
    forma = float(row.get("forma_recente_score", 0) or 0)
    tit_pct = float(row.get("titolarita_pct", 0) or 0)
    
    minutes_trend = 1.0
    if forma > 0.15 or tit_pct > 0.5:
        minutes_trend = 1.3
    elif forma < -0.15:
        minutes_trend = 0.8

    age_bonus = math.exp((YOUTH_CUTOFF_AGE - age) / YOUTH_TAU) - 1.0
    raw_bonus = age_bonus * minutes_trend * 0.55
    return float(np.clip(raw_bonus, 0.0, 0.40))'''

content = content.replace(old_youth, new_youth)

with open("vip.py", "w") as f:
    f.write(content)
