import json

with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell.get('source', []))
        if '11-3' in src:
            print(f"--- Cell {i} (MD) ---")
            print(src.strip()[:500])
    elif cell['cell_type'] == 'code':
        src = ''.join(cell.get('source', []))
        if '11-3' in src:
            print(f"--- Cell {i} (Code prints) ---")
            for line in src.split('\n'):
                if '11-3' in line:
                    print(line.strip())
