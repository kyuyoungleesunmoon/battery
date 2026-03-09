import json

def check_phase_11():
    with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            if 'Cluster_Pred' in src or 'Cluster_GT' in src or 'Phase 11' in src or '11-3' in src or '11-4' in src:
                print(f"--- Cell {i} ---")
                for line in src.split('\n'):
                    if 'Cluster_' in line or 'Phase 11' in line or '11-' in line:
                        print(line)

check_phase_11()
