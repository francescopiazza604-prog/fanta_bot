with open("app.py", "r") as f:
    content = f.read()

target_import = "from predict import train_prediction_model"
replacement_import = """from predict import train_prediction_model

@st.cache_data(show_spinner=False)
def _get_cached_predictions(_df, strat):
    # Usiamo _df solo per invalidare la cache se cambia
    if _df is not None:
        _df.to_csv(INPUT_TEMP, index=False)
        return train_prediction_model(INPUT_TEMP, strategy=strat)
    return None"""
content = content.replace(target_import, replacement_import)

content = content.replace("df_cop = train_prediction_model(INPUT_TEMP, strategy=strategy)", "df_cop = _get_cached_predictions(data_df, strategy)")
content = content.replace("df_s = train_prediction_model(INPUT_TEMP, strategy=strategy)", "df_s = _get_cached_predictions(data_df, strategy)")
content = content.replace("df_all = train_prediction_model(INPUT_TEMP, strategy=strategy)", "df_all = _get_cached_predictions(data_df, strategy)")
content = content.replace("df_vip = train_prediction_model(INPUT_TEMP, strategy=strategy)", "df_vip = _get_cached_predictions(data_df, strategy)")

with open("app.py", "w") as f:
    f.write(content)
