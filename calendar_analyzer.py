import requests
import pandas as pd
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Mappatura nomi squadre fixturedownload -> fantacalcio
TEAM_MAPPING = {
    "Hellas Verona": "Verona",
    # Aggiungi altri se necessario
}

# Forza stimata delle squadre (1-5, dove 5 è la più forte) per calcolare la difficoltà
TEAM_STRENGTH = {
    "Inter": 5.0,
    "Juventus": 4.5, "Milan": 4.5, "Napoli": 4.5, "Atalanta": 4.5,
    "Roma": 4.0, "Lazio": 4.0,
    "Fiorentina": 3.5, "Bologna": 3.5,
    "Torino": 3.0, "Genoa": 3.0, "Monza": 3.0,
    "Lecce": 2.5, "Udinese": 2.5, "Verona": 2.5, "Empoli": 2.5, "Cagliari": 2.5,
    "Parma": 2.0, "Como": 2.0, "Venezia": 2.0
}

CALENDAR_URL = "https://fixturedownload.com/feed/json/serie-a-2024"

def fetch_calendar() -> list[dict]:
    """Scarica il calendario e normalizza i nomi delle squadre."""
    try:
        resp = requests.get(CALENDAR_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        for match in data:
            if match["HomeTeam"] in TEAM_MAPPING:
                match["HomeTeam"] = TEAM_MAPPING[match["HomeTeam"]]
            if match["AwayTeam"] in TEAM_MAPPING:
                match["AwayTeam"] = TEAM_MAPPING[match["AwayTeam"]]
                
        return data
    except Exception as e:
        logger.error(f"Errore download calendario: {e}")
        return []

def get_team_schedule(calendar: list[dict], team: str, from_matchday: int = 1, to_matchday: int = 38) -> list[dict]:
    """Restituisce le partite di una singola squadra."""
    schedule = []
    for match in calendar:
        rnd = match.get("RoundNumber")
        if not rnd or not isinstance(rnd, int):
            continue
            
        if from_matchday <= rnd <= to_matchday:
            if match["HomeTeam"] == team:
                schedule.append({"matchday": rnd, "opponent": match["AwayTeam"], "is_home": True})
            elif match["AwayTeam"] == team:
                schedule.append({"matchday": rnd, "opponent": match["HomeTeam"], "is_home": False})
    
    return sorted(schedule, key=lambda x: x["matchday"])

def evaluate_schedule_difficulty(calendar: list[dict], team: str, from_matchday: int = 1, num_matches: int = 5) -> float:
    """Calcola la difficoltà media delle prossime N partite per una squadra (da 1 a 5)."""
    schedule = get_team_schedule(calendar, team, from_matchday, from_matchday + num_matches - 1)
    if not schedule:
        return 3.0 # default medio
        
    diff_sum = 0.0
    for match in schedule:
        opp = match["opponent"]
        # Usa la forza dell'avversario
        opp_strength = TEAM_STRENGTH.get(opp, 3.0)
        # Bonus se giochi in casa, malus in trasferta (es. giocare in trasferta è più difficile)
        match_diff = opp_strength
        if not match["is_home"]:
            match_diff += 0.5
        else:
            match_diff -= 0.5
            
        diff_sum += match_diff
        
    avg_diff = diff_sum / len(schedule)
    return round(max(1.0, min(5.0, avg_diff)), 2)

def get_all_teams_difficulty(calendar: list[dict], from_matchday: int = 1, num_matches: int = 5) -> dict[str, float]:
    """Restituisce la difficoltà del calendario per tutte le squadre."""
    difficulties = {}
    teams = list(TEAM_STRENGTH.keys())
    for team in teams:
        difficulties[team] = evaluate_schedule_difficulty(calendar, team, from_matchday, num_matches)
    return difficulties

def find_best_gk_pairings(calendar: list[dict], teams: list[str], from_matchday: int = 1, num_matches: int = 38) -> list[dict]:
    """
    Trova i migliori incroci per i portieri. 
    Per ogni coppia di squadre, calcola la difficoltà *minima* tra le due per ogni giornata.
    Più è bassa, migliore è l'incrocio.
    """
    pairings = []
    
    schedules = {}
    for team in teams:
        schedules[team] = {m["matchday"]: m for m in get_team_schedule(calendar, team, from_matchday, from_matchday + num_matches - 1)}
        
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            teamA = teams[i]
            teamB = teams[j]
            
            total_difficulty = 0
            valid_matches = 0
            perfect_alternations = 0 # Quante volte una è in casa e l'altra in trasferta
            
            for rnd in range(from_matchday, from_matchday + num_matches):
                mA = schedules[teamA].get(rnd)
                mB = schedules[teamB].get(rnd)
                
                if not mA or not mB:
                    continue
                    
                valid_matches += 1
                if mA["is_home"] != mB["is_home"]:
                    perfect_alternations += 1
                    
                diffA = TEAM_STRENGTH.get(mA["opponent"], 3.0) + (0.5 if not mA["is_home"] else -0.5)
                diffB = TEAM_STRENGTH.get(mB["opponent"], 3.0) + (0.5 if not mB["is_home"] else -0.5)
                
                total_difficulty += min(diffA, diffB)
                
            if valid_matches > 0:
                avg_diff = total_difficulty / valid_matches
                pairings.append({
                    "team1": teamA,
                    "team2": teamB,
                    "avg_difficulty": round(avg_diff, 2),
                    "alternation_pct": round(perfect_alternations / valid_matches * 100, 1)
                })
                
    return sorted(pairings, key=lambda x: x["avg_difficulty"])


def apply_calendar_modifiers(df: pd.DataFrame, calendar: list[dict], from_matchday: int = 1, num_matches: int = 5, strategy: str = "master") -> pd.DataFrame:
    difficulties = get_all_teams_difficulty(calendar, from_matchday, num_matches)
    
    def _calc_mod(row):
        team = row.get("squadra", "Sconosciuta")
        if team not in difficulties:
            return row
            
        diff = difficulties[team]
        diff_delta = 3.0 - diff
        
        ruolo = row.get("ruolo", "C")
        if ruolo == "P":
            mod = diff_delta * 0.30
        elif ruolo == "D":
            mod = diff_delta * 0.15
        else:
            mod = diff_delta * 0.10
            
        if strategy == "sprint_calendario":
            mod *= 2.0
            
        row["calendar_diff"] = diff
        row["previsione_ia"] += mod
        return row

    if "squadra" in df.columns:
        return df.apply(_calc_mod, axis=1)
    return df
