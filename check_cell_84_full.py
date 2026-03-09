import json

with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('check_cell_84.txt', 'w', encoding='utf-8') as out:
    out.write(f"--- Cell 83 (Code) ---\n")
    out.write(''.join(nb['cells'][83].get('source', [])) + '\n\n')
    out.write(f"--- Cell 84 (Code) ---\n")
    out.write(''.join(nb['cells'][84].get('source', [])) + '\n\n')
    out.write(f"--- Cell 85 (Code) ---\n")
    out.write(''.join(nb['cells'][85].get('source', [])) + '\n\n')
