import pandas as pd
import random

def generate_mock_data(n_players=200):
    roles = ['P', 'D', 'C', 'A']
    names = [f"Giocatore_{i}" for i in range(n_players)]
    
    data = []
    for i in range(n_players):
        role = random.choice(roles)
        # Costi medi indicativi
        if role == 'P':
            cost = random.randint(1, 25)
            pts = random.uniform(4.5, 6.5)
        elif role == 'D':
            cost = random.randint(1, 40)
            pts = random.uniform(5.0, 7.5)
        elif role == 'C':
            cost = random.randint(1, 60)
            pts = random.uniform(5.5, 8.5)
        else: # A
            cost = random.randint(20, 150)
            pts = random.uniform(6.0, 10.5)
            
        data.append({
            'id': i,
            'nome': names[i],
            'ruolo': role,
            'costo': cost,
            'punti_scorsa_stagione': round(pts, 2),
            'punti_previsti': round(pts * random.uniform(0.9, 1.1), 2)
        })
        
    df = pd.DataFrame(data)
    df.to_csv('data/players_mock.csv', index=False)
    print("Dataset mock generato in data/players_mock.csv")

if __name__ == "__main__":
    generate_mock_data()
