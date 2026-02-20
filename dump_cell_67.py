import json

with open('train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('cell_67_content.txt', 'w', encoding='utf-8') as f:
    f.write("".join(nb['cells'][67]['source']))
