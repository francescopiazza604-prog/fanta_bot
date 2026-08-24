with open("app.py", "r") as f:
    content = f.read()

content = content.replace(
    "df_pred = train_prediction_model(INPUT_TEMP, strategy=strategy)",
    "df_pred = _get_cached_predictions(data_df, strategy).copy() if _get_cached_predictions(data_df, strategy) is not None else None"
)

with open("app.py", "w") as f:
    f.write(content)
