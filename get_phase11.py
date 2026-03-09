import json

def get_phase_11():
    with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    with open('phase_11_full.txt', 'w', encoding='utf-8') as out:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                src = ''.join(cell.get('source', []))
                if 'ari_score' in src or 'Cluster_GT' in src or 'Cluster_Pred' in src or 'Phase 11' in src:
                    out.write(f"--- Cell {i} ---\n")
                    out.write(src + "\n\n")

get_phase_11()
