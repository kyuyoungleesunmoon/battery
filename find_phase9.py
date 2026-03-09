import json

def find_phase9_scenarios():
    with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    with open('phase9_scenarios.txt', 'w', encoding='utf-8') as out:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                src = ''.join(cell.get('source', []))
                if 'phase9' in src.lower() or 'scenario' in src.lower() or 'best_model' in src.lower() or 'S4_ALL' in src or 'S1_' in src:
                    out.write(f"--- Cell {i} ---\n")
                    out.write(src + "\n\n")

find_phase9_scenarios()
