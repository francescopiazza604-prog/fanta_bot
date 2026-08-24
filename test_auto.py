with open("app.py", "r") as f:
    content = f.read()

# Trova "Login automatico" e metti un bell'avviso
content = content.replace(
    'st.sidebar.expander("🔑 Login automatico (email + password)")',
    'st.sidebar.expander("🔑 SCARICA LISTONE AUTOMATICO (Login)", expanded=True)'
)

with open("app.py", "w") as f:
    f.write(content)
