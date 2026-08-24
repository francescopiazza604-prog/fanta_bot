import json
import os
import subprocess

transcript_path = os.path.expanduser('~/.gemini/antigravity-cli/brain/e9e38392-5a95-4e8c-b8c9-40402aa3982b/.system_generated/logs/transcript_full.jsonl')
with open(transcript_path, 'r') as f:
    lines = f.readlines()

def do_replace(args):
    target = args['TargetFile']
    old = args['TargetContent']
    new = args['ReplacementContent']
    print(f"Replaying replace_file_content on {target}...")
    try:
        with open(target, 'r') as f:
            content = f.read()
        if old in content:
            content = content.replace(old, new)
            with open(target, 'w') as f:
                f.write(content)
            print("Success.")
        else:
            print("WARNING: Target content not found!")
    except Exception as e:
        print(f"File error: {e}")

def do_command(cmd):
    if 'git restore' in cmd or 'rm *.py' in cmd or 'git checkout' in cmd:
        return # DANGER
    is_mutation = False
    if "cat << 'EOF'" in cmd or 'cat << "EOF"' in cmd:
        is_mutation = True
    elif cmd.startswith('python3 fix_') or cmd.startswith('python3 patch_') or cmd.startswith('python3 add_') or cmd.startswith('python3 clean_') or cmd.startswith('python3 unify_') or cmd.startswith('python3 rm_'):
        is_mutation = True
    elif 'EOF' in cmd and 'python3' in cmd:
        is_mutation = True
        
    if is_mutation:
        print(f"Executing command: {cmd[:60]}...")
        subprocess.run(cmd, shell=True, check=False)

for line in lines:
    try:
        data = json.loads(line)
        if data.get('source') == 'MODEL' and 'tool_calls' in data:
            for tc in data['tool_calls']:
                if tc['name'] == 'replace_file_content':
                    do_replace(tc['args'])
                elif tc['name'] == 'run_command':
                    cmd = tc['args'].get('CommandLine', '')
                    do_command(cmd)
    except Exception as e:
        pass

