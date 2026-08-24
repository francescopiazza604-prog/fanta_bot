with open("calendar_analyzer.py", "r") as f:
    content = f.read()

target = """def apply_calendar_modifiers(df: pd.DataFrame, num_matches: int = 10) -> pd.DataFrame:"""
replacement = """def apply_calendar_modifiers(df: pd.DataFrame, num_matches: int = 10, strategy: str = "master") -> pd.DataFrame:"""
content = content.replace(target, replacement)

target2 = """        if ruolo == 'P':
            mod = delta * 0.30
        elif ruolo == 'D':
            mod = delta * 0.15
        else:
            mod = delta * 0.10
            
        return row['previsione_ia'] + mod"""

replacement2 = """        if ruolo == 'P':
            mod = delta * 0.30
        elif ruolo == 'D':
            mod = delta * 0.15
        else:
            mod = delta * 0.10
            
        # Raddoppia il peso del calendario se la strategia lo richiede
        if strategy == 'sprint_calendario':
            mod *= 2.0
            
        return row['previsione_ia'] + mod"""
content = content.replace(target2, replacement2)

with open("calendar_analyzer.py", "w") as f:
    f.write(content)

with open("predict.py", "r") as f:
    p_content = f.read()

p_content = p_content.replace(
    "df = apply_calendar_modifiers(df)",
    "df = apply_calendar_modifiers(df, strategy=strategy)"
)

with open("predict.py", "w") as f:
    f.write(p_content)

