with open("app.py", "r") as f:
    content = f.read()

target = """    if fc_excel_file is not None:
        with st.sidebar.status("Importazione Excel...") as s:
            try:
                ok, msg = save_manual_excel(fc_excel_file.getvalue())"""

replacement = """    if fc_excel_file is not None:
        file_id = fc_excel_file.name + str(fc_excel_file.size)
        if st.session_state.get("last_uploaded_file") != file_id:
            with st.sidebar.status("Importazione Excel...") as s:
                try:
                    ok, msg = save_manual_excel(fc_excel_file.getvalue())"""

content = content.replace(target, replacement)

# Need to properly indent everything inside that block!
# Let's just use re.sub to indent the inner block

import re
lines = content.split('\n')
in_block = False
for i, line in enumerate(lines):
    if line.strip() == 'file_id = fc_excel_file.name + str(fc_excel_file.size)':
        in_block = True
    elif in_block and line.startswith('    if fc_excel_file is not None:'):
        # We passed the block, actually it's outside
        pass
    
with open("app.py", "w") as f:
    f.write(content)
