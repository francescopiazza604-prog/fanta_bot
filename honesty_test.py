import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
from sklearn.ensemble import RandomForestRegressor

def run_honesty_test(data_path, budget=500):
    df = pd.read_csv(data_path)
    
    # 1. ADDESTRAMENTO 'AL BUIO'
    # L'IA vede solo il passato (22/23) e i costi. NON vede 'fm_reale_23_24'.
    X = df[['fm_22_23', 'costo_asta_23_24']]
    y = df['fm_22_23'] # In un caso reale qui useremmo trend e stats avanzate
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Prevediamo il futuro (Stagione 23/24)
    df['previsione_ia_23_24'] = model.predict(X)
    
    # 2. OTTIMIZZAZIONE BASATA SULLA PREVISIONE
    prob = LpProblem("HonestyOptimization", LpMaximize)
    player_vars = LpVariable.dicts("Player", df.index, cat="Binary")
    
    prob += lpSum([df.loc[i, 'previsione_ia_23_24'] * player_vars[i] for i in df.index])
    prob += lpSum([df.loc[i, 'costo_asta_23_24'] * player_vars[i] for i in df.index]) <= budget
    
    # Vincoli minimi (Semplificati per il test: 1-4-4-2 per velocizzare)
    prob += lpSum([player_vars[i] for i in df.index if df.loc[i, 'ruolo'] == 'P']) == 1
    prob += lpSum([player_vars[i] for i in df.index if df.loc[i, 'ruolo'] == 'D']) == 4
    prob += lpSum([player_vars[i] for i in df.index if df.loc[i, 'ruolo'] == 'C']) == 4
    prob += lpSum([player_vars[i] for i in df.index if df.loc[i, 'ruolo'] == 'A']) == 2
    
    prob.solve()
    
    # 3. VERIFICA REALE (CONFRONTO COL FUTURO)
    selected_indices = [i for i in df.index if player_vars[i].varValue == 1]
    team_selected = df.loc[selected_indices]
    
    print("\n--- RISULTATO TEST DI ONESTÀ (Previsione vs Realtà) ---")
    print(f"Squadra scelta dall'IA con budget {budget} (senza sapere i voti reali):")
    
    total_fm_prevista = team_selected['previsione_ia_23_24'].sum()
    total_fm_reale = team_selected['fm_reale_23_24'].sum()
    
    for _, row in team_selected.iterrows():
        diff = round(row['fm_reale_23_24'] - row['previsione_ia_23_24'], 2)
        status = "🔥 SOPRA LE ATTESE" if diff > 0.3 else "❄️ SOTTO LE ATTESE" if diff < -0.3 else "✅ IN LINEA"
        print(f"- {row['nome'].ljust(18)} | Prev: {row['previsione_ia_23_24']} | Real: {row['fm_reale_23_24']} | {status}")
        
    print("\n" + "="*50)
    print(f"FantaMedia Totale Prevista: {round(total_fm_prevista, 2)}")
    print(f"FantaMedia Totale REALE:    {round(total_fm_reale, 2)}")
    accuracy = (1 - abs(total_fm_reale - total_fm_prevista) / total_fm_reale) * 100
    print(f"Accuratezza Previsione:     {round(accuracy, 2)}%")
    print("="*50)

if __name__ == "__main__":
    run_honesty_test('data/honesty_test_data.csv')
