import json

with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('dump_bcp.txt', 'w', encoding='utf-8') as out:
    for i in range(81, 91):
        cell = nb['cells'][i]
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell.get('source', []))
            out.write(f"--- Cell {i} (MD) ---\n")
            out.write(src.strip()[:200] + '\n\n')
        elif cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            lines = [line.strip() for line in src.split('\n') if 'print' in line and ('Phase' in line or '10-' in line)]
            if lines:
                out.write(f"--- Cell {i} (Code prints) ---\n")
                for l in lines:
                    out.write(l + '\n')
                out.write('\n')
