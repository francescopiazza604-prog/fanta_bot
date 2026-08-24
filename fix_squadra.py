with open("app.py", "r") as f:
    content = f.read()

content = content.replace("data_df['squadra']", "data_df.get('squadra', data_df.get('Squadra', data_df.get('Sq', [])))")

with open("app.py", "w") as f:
    f.write(content)
