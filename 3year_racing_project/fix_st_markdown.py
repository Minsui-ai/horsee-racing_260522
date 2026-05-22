import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace """, unsafe_allow_html=True) with """.replace('\n', ''), unsafe_allow_html=True)
content = content.replace('""", unsafe_allow_html=True)', '""".replace("\\n", ""), unsafe_allow_html=True)')
content = content.replace('", unsafe_allow_html=True)', '".replace("\\n", ""), unsafe_allow_html=True)')
content = content.replace("', unsafe_allow_html=True)", "'.replace('\\n', ''), unsafe_allow_html=True)")

# Handle cases where they might be already replaced to avoid double replace
content = content.replace('.replace("\\n", "").replace("\\n", "")', '.replace("\\n", "")')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Applied newline replacement to st.markdown.")
