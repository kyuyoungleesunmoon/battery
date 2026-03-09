import json

with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('check_11_3_full.txt', 'w', encoding='utf-8') as out:
    for i in range(94, 98):
        if i >= len(nb['cells']):
            break
        cell = nb['cells'][i]
        src = ''.join(cell.get('source', []))
        out.write(f"--- Cell {i} ({cell['cell_type']}) ---\n")
        out.write(src + '\n\n')
