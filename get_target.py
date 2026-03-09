import json

def get_target_name():
    with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            for line in src.split('\n'):
                if 'TARGET =' in line or 'TARGET=' in line:
                    print(line)
                    return

get_target_name()
