import json

def trace_data():
    with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            for line in src.split('\n'):
                if line.strip().startswith('data = ') or line.strip().startswith('data='):
                    print(f"Cell {i}: {line.strip()}")

trace_data()
