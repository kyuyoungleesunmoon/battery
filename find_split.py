import json

with open('train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell['source'])
    if 'X_train' in source and 'train_test_split' in source:
        print(f"Cell {i} defines 'X_train' using train_test_split:")
        print(source)
