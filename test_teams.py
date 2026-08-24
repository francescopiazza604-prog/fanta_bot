import pandas as pd
df = pd.read_csv('data/fantacalcio_stats_cache.csv')
print(sorted(df['squadra'].dropna().unique().tolist()))
