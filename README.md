# FantaBot - Ottimizzatore Fantacalcio

Questo progetto mira a creare uno strumento avanzato per il Fantacalcio che permetta di:
1. Calcolare la miglior combinazione di giocatori basandosi su un budget prefissato.
2. Analizzare le performance storiche.
3. Prevedere le performance future tramite modelli di Intelligenza Artificiale.

## Struttura del Progetto
- `data/`: Contiene i dataset dei giocatori (CSV/JSON).
- `optimizer.py`: Logica di ottimizzazione (Integer Programming).
- `predict.py`: Modelli di previsione (AI/ML).
- `utils.py`: Funzioni di supporto.

## Requisiti
- Python 3.8+
- pandas
- pulp (per l'ottimizzazione lineare)
