import json

with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('bcp_cells.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        if i >= 79 and i <= 95:
            if cell['cell_type'] == 'markdown':
                src = ''.join(cell.get('source', []))
                out.write(f"Cell {i} (MD): {src[:120].replace(chr(10), ' ')}\n")
            elif cell['cell_type'] == 'code':
                src = ''.join(cell.get('source', []))
                lines = src.split('\n')
                for line in lines:
                    if 'print' in line and 'Phase' in line:
                        out.write(f"Cell {i} (Code): {line.strip()}\n")
