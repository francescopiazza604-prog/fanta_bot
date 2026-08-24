with open("calendar_analyzer.py", "r") as f:
    content = f.read()

target = """        if ruolo == 'P':
            mod = delta * 0.15
        elif ruolo == 'D':
            mod = delta * 0.07
        else:
            mod = delta * 0.04"""

replacement = """        if ruolo == 'P':
            mod = delta * 0.30
        elif ruolo == 'D':
            mod = delta * 0.15
        else:
            mod = delta * 0.10"""

content = content.replace(target, replacement)
with open("calendar_analyzer.py", "w") as f:
    f.write(content)
