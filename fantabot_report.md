# 🤖 FantaBot AI 2026/27: Analisi e Architettura del Bot

Questo report tecnico illustra l'architettura, i modelli matematici e le logiche di Intelligenza Artificiale alla base di **FantaBot AI 2026/27**. 

Puoi girare questo report ai tuoi amici per dimostrargli che non stanno competendo contro una semplice "tabella Excel", ma contro un sistema avanzato di machine learning e ottimizzazione operativa.

---

## 1. Il Motore Predittivo (Intelligenza Artificiale)
Il cuore predittivo del sistema (`predict.py`) non si limita a guardare la "fantamedia" dell'anno scorso, ma utilizza modelli di **Machine Learning** (Scikit-learn) per stimare le performance future.

* **Modelli Differenziati per Ruolo:** L'IA non mischia difensori e attaccanti. Utilizza un *GradientBoostingRegressor* (o *Ridge Regression* per dataset più piccoli) separato per ciascun ruolo. In questo modo il modello impara a capire, ad esempio, che un certo numero di xG per un difensore vale molto di più rispetto allo stesso numero per un attaccante.
* **Feature Engineering Dinamico:** Il modello ingerisce dati tattici avanzati da FBRef e Transfermarkt. Valuta: xG (Expected Goals), xA, gol subiti, ammonizioni, ma anche il **Transfer Effect** (valutando se un giocatore è andato in una squadra superiore/inferiore) e il **Coach Bonus** (stile di gioco dell'allenatore, es. possesso vs contropiede).
* **Shrinkage Bayesiano (Regressione alla Media):** L'algoritmo penalizza i giocatori con poche presenze che hanno avuto "sculate" o picchi di forma isolati. La previsione viene "tirata" verso la media del ruolo in base all'incertezza (meno partite = meno certezza del dato).

## 2. Il Modulo V.I.P. (Valore Intrinseco Predittivo)
Il machine learning puro a volte non capisce il calcio giocato. Il modulo `vip.py` codifica la **conoscenza umana e tattica** per trovare valore nascosto (i famosi "bug di listone"):

* **Tactical Position Modifier (TPM):** Applica un moltiplicatore ai giocatori che in campo giocano in posizioni più avanzate rispetto al loro ruolo al Fantacalcio. Ad esempio, un "Quinto d'attacco" (come Dimarco o Theo) listato come difensore, o un "Trequartista" listato come centrocampista. L'algoritmo inferisce questo ruolo automaticamente dalle statistiche (se non inserito manualmente) e ne boosta lo score fino al +30%.
* **P90 Discovery Factor:** Uno stimatore bayesiano di Laplace scova i talenti emergenti. Analizza la produzione per 90 minuti (P90) dei panchinari/subentrati. Se un ragazzo entra 15 minuti e produce sempre pericoli, il sistema lo intercetta prima che diventi famoso.
* **Under-22 Growth Coefficient:** I giocatori sotto i 22 anni con un minutaggio in crescita ricevono un moltiplicatore esponenziale che simula la tipica curva di "esplosione" dei giovani prospetti.

## 3. L'Ottimizzatore (Ricerca Operativa)
Costruire la rosa perfetta con un budget limitato è un classico problema matematico (Knapsack Problem). `optimizer.py` utilizza un solver **MILP** (Mixed Integer Linear Programming) tramite la libreria `PuLP`.

Tuttavia, l'ottimizzatore di FantaBot non si limita a massimizzare i punti (che produrrebbe rose piatte con 25 giocatori mediocri), ma impone **6 livelli gerarchici di vincoli matematici ("Anti-piattume")**:
1. **Composizione esatta:** 3P, 8D, 8C, 6A.
2. **Budget Totale:** Spesa massima (es. 500 crediti).
3. **Fasce di Budget (Bands):** Assicura che una percentuale specifica del budget vada a ogni reparto in base alla *Strategia* (es. l'Aggressiva impone di spendere il ~40% solo sugli attaccanti).
4. **Slot "Top Tier":** Obbliga il solver a prendere ALMENO un certo numero di giocatori "Premium" (top 20% dei più costosi) per evitare squadre di soli low-cost.
5. **Costo Medio Minimo:** Impedisce di riempire la difesa con soli giocatori da 1 credito.
6. **Qualità GK:** Forza la selezione di almeno un Portiere da una squadra difensivamente d'élite (o dal top 40% degli score).

## 4. Live Auction Copilot (Copilota per l'Asta in Diretta)
Durante l'asta, entra in gioco `auction_copilot.py`. Questo modulo ricalcola costantemente in tempo reale:
* **L'Inflazione del Mercato:** Analizzando quanti crediti rimangono a te e ai tuoi avversari rispetto agli slot da riempire.
* **La Scarsità Dinamica:** Se sei l'unico a cui manca ancora un portiere, il sistema ricalcola il prezzo massimo (Max Bid) facendolo crollare, perché sa che nessuno rilancerà.
* **Tracciamento Avversari:** Il bot monitora i budget e le tendenze strategiche degli avversari per dirti esattamente fino a quanto puoi spingerti per un giocatore.

---

> [!TIP]
> **Strategie disponibili per il Solver:**
> L'IA può generare rose secondo diverse filosofie: **Listone Master** (coperture perfette per campionati a listone), **Conservativa** (bassa varianza, alta sicurezza), **Aggressiva** (massimizzazione offensiva per leghe a buste chiuse), **Scommesse / Moneyball** (ricerca del valore puro e dei crack tattici).

*Preparati a vincere, perché i tuoi amici stanno giocando a dadi, tu stai giocando a scacchi con un supercomputer.*
