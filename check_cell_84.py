import json

with open('battery_capacity_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"--- Cell 83 (Code) ---")
print(''.join(nb['cells'][83].get('source', [])))
print(f"--- Cell 84 (Code) ---")
print(''.join(nb['cells'][84].get('source', [])))
