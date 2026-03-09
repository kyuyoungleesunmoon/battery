import json

def check_phase_11():
    with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            if 'Cluster_' in src or 'ari_score' in src:
                print(f"--- Cell {i} ---")
                print(src)

check_phase_11()
