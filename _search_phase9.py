import json

with open('train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Total cells: {len(cells)}")

keywords = ['Phase 9', '정보 채널', 'scenario1_features', 'scenario2_features', 
            'scenario3_features', 'scenario4_features', '근본 원인']

for i, c in enumerate(cells):
    src = ''.join(c['source'])
    if any(kw in src for kw in keywords):
        ct = c['cell_type']
        print(f"\n{'='*60}")
        print(f"Cell {i} ({ct})")
        print('='*60)
        print(src[:2000])
        if len(src) > 2000:
            print('...(truncated)')
