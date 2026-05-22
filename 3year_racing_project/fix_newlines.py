import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace newlines in f-strings for helper functions and table rows

content = re.sub(r'return f\"\"\"\n\s+<div', 'return f\"\"\"<div', content)
content = re.sub(r'return f\"\"\"\n\s+<span', 'return f\"\"\"<span', content)
content = re.sub(r'return \"\"\"\n\s+<div', 'return \"\"\"<div', content)

content = re.sub(r'</div>\n\s+\"\"\"', '</div>\"\"\"', content)
content = re.sub(r'</span>\n\s+\"\"\"', '</span>\"\"\"', content)

content = re.sub(r'table_rows_html \+= f\"\"\"\n\s+<tr', 'table_rows_html += f\"\"\"<tr', content)
content = re.sub(r'table_rows_b \+= f\"\"\"\n\s+<tr', 'table_rows_b += f\"\"\"<tr', content)

content = re.sub(r'</tr>\n\s+\"\"\"', '</tr>\"\"\"', content)

# Remove internal blank lines within table_rows_html f-string
# Actually, the safest way is to just compress table_rows_html completely so there are NO newlines between <tr> and </tr>
# But re.sub with multiline might be tricky. Let's just remove all empty lines inside the string.

lines = content.split('\n')
new_lines = []
in_table_row = False
for line in lines:
    if 'table_rows_html += f"""' in line or 'table_rows_b += f"""' in line:
        in_table_row = True
    if in_table_row and line.strip() == '':
        continue # skip empty lines
    if in_table_row and '"""' in line and not line.startswith('table_rows'):
        in_table_row = False
    new_lines.append(line)

content = '\n'.join(new_lines)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed newlines.")
