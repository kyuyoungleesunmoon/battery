import json

with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('md_cells.txt', 'w', encoding='utf-8') as out:
    for i in [81, 85, 91]:
        cell = nb['cells'][i]
        src = ''.join(cell.get('source', []))
        out.write(f"--- Cell {i} ---\n{src}\n\n")
