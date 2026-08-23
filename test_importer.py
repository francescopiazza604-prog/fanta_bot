import traceback
import importer
try:
    df = importer.import_real_quotazioni('data/serie_a_23_24_backtest.csv')
    print("SUCCESS!")
    print(df.head())
except Exception as e:
    print("ERROR!")
    traceback.print_exc()
