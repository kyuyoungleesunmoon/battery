import json

with open('train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell['source'])
    if 'scenarios =' in source:
        print(f"Cell {i} defines 'scenarios':")
        print(source)
    if 'def get_scenario_data' in source:
        print(f"Cell {i} defines 'get_scenario_data':")
        print(source)
