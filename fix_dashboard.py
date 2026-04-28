with open('app/dashboard.py', 'r') as f:
    content = f.read()

# Remove any existing if __name__ blocks that might be in the middle
lines = content.split('\n')
cleaned_lines = []
in_main_block = False
for line in lines:
    if line.strip().startswith('if __name__'):
        in_main_block = True
        continue
    if in_main_block and line and not line[0].isspace():
        in_main_block = False
    if not in_main_block:
        cleaned_lines.append(line)

content = '\n'.join(cleaned_lines)

# Add if __name__ block at the end
if not content.strip().endswith('\n'):
    content += '\n'
content += '\nif __name__ == "__main__":\n    main()\n'

with open('app/dashboard.py', 'w') as f:
    f.write(content)

print('✅ Added if __name__ block at the end')
