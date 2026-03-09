import json
import sys

def check_file(filename, out_filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except FileNotFoundError:
        return

    with open(out_filename, 'w', encoding='utf-8') as out:
        out.write(f"--- Markdown Cells in {filename} ---\n")
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'markdown':
                src = ''.join(cell.get('source', []))
                for line in src.split('\n'):
                    if line.strip().startswith('#') and 'Phase 10' in line:
                        out.write(f"Cell {i}: {line.strip()}\n")
                    elif line.strip().startswith('#') and 'Phase 11' in line:
                        out.write(f"Cell {i}: {line.strip()}\n")

check_file('battery_capacity_prediction.ipynb', 'bcp_phases.txt')
check_file('train.ipynb', 'train_phases_detailed.txt')
