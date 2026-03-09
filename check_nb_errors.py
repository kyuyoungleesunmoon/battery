import json

def check_notebook_errors(filepath='battery_capacity_prediction.ipynb'):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    errors = []
    for i, cell in enumerate(nb.get('cells', [])):
        if cell['cell_type'] == 'code':
            outputs = cell.get('outputs', [])
            for out in outputs:
                if out.get('output_type') == 'error':
                    ename = out.get('ename', '')
                    evalue = out.get('evalue', '')
                    errors.append({
                        'cell_idx': i,
                        'ename': ename,
                        'evalue': evalue,
                        'traceback': out.get('traceback', [])
                    })
                    
    if not errors:
        print("No errors found in the notebook outputs!")
    else:
        for err in errors:
            print(f"--- ERROR AT CELL {err['cell_idx']} ---")
            print(f"{err['ename']}: {err['evalue']}")
            # print trace snippet
            for tb in err['traceback'][-3:]:
                print(tb)
            print()

if __name__ == '__main__':
    check_notebook_errors()
