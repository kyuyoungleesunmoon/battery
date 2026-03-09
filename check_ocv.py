import json
import sys

with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell.get('source', []))
        if 'ocv_deviation' in src:
            print(f"--- Code Cell {i} contains ocv_deviation ---")
            for line in src.split('\n'):
                if 'ocv_deviation' in line:
                    print(line.strip())
