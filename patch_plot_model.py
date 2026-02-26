import json

notebook_path = r'c:\6.1 밧데리_학습\train.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        new_source = []
        changed = False
        for line in source:
            if str(line).strip() == "plot_model(tuned_model, plot='feature')":
                new_source.extend([
                    "try:\n",
                    "    plot_model(tuned_model, plot='feature')\n",
                    "except TypeError as e:\n",
                    "    print(f'Feature importance plot is not supported for this estimator: {e}')\n"
                ])
                changed = True
            else:
                new_source.append(line)
        if changed:
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1, separators=(',', ': '))

print('REPLACED_SUCCESSFULLY')
