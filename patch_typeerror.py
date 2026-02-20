import json

with open('train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Cell 91 (Phase 5-1)
target_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('# 5-1.' in line for line in cell['source']):
        target_index = i
        break

if target_index != -1:
    print(f"Found Phase 5-1 at index {target_index}")
    source = nb['cells'][target_index]['source']
    new_source = []
    for line in source:
        # Fix 1: Change mean_squared_error(..., squared=False) to np.sqrt(mean_squared_error(...))
        if 'mean_squared_error(y_test, y_pred, squared=False)' in line:
            new_line = line.replace('mean_squared_error(y_test, y_pred, squared=False)', 'np.sqrt(mean_squared_error(y_test, y_pred))')
            new_source.append(new_line)
        else:
            new_source.append(line)
    
    nb['cells'][target_index]['source'] = new_source

    with open('train.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Successfully patched Phase 5-1 code.")
else:
    print("Could not find Phase 5-1 code cell.")
