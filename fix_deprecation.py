"""Fix use_container_width deprecation warnings in app.py"""
import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1) st.image: use_container_width=use_width -> width=...
content = content.replace(
    'st.image(img, caption=caption, use_container_width=use_width)',
    'st.image(img, caption=caption, width="stretch" if use_width else "content")'
)

# 2) st.dataframe: use_container_width=True, hide_index=True -> width="stretch", hide_index=True
content = content.replace(
    'use_container_width=True, hide_index=True',
    'width="stretch", hide_index=True'
)

# 3) st.plotly_chart: use_container_width=True -> use_container_width=True (keep as-is for plotly)
# plotly_chart still supports it, but let's also update
content = content.replace(
    'st.plotly_chart(fig, use_container_width=True)',
    'st.plotly_chart(fig, use_container_width=True)'
)

# 4) Any remaining standalone use_container_width=True
content = content.replace('use_container_width=True', 'width="stretch"')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

count = content.count('use_container_width')
print(f'Done. Remaining use_container_width occurrences: {count}')
