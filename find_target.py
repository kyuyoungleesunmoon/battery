import json

def find_target_col():
    with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            if 'TARGET' in src or 'capacity' in src.lower():
                print(f"--- Cell {i} ---")
                for line in src.split('\n'):
                    if 'TARGET =' in line or 'TARGET=' in line:
                        print(line.strip())
                    elif 'Capacity' in line or 'capacity' in line:
                        print(line.strip())

find_target_col()
