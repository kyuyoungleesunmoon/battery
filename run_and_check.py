import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import glob

def run_notebook(notebook_path):
    print(f"Executing {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=1800, kernel_name='python3', allow_errors=True)
    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
    except Exception as e:
        print(f"Preprocessor Exception: {e}")
        
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    # Check for any errors in the executed notebook
    error_found = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            for output in cell.get('outputs', []):
                if output['output_type'] == 'error':
                    print(f"\n--- ERROR AT CELL {i} ---")
                    print(f"Error Name: {output['ename']}")
                    print(f"Error Value: {output['evalue']}")
                    print("Traceback:")
                    print('\n'.join(output['traceback']))
                    error_found = True
                    return False
    if not error_found:
        print("\nSUCCESS! No errors found.")
        return True

if __name__ == '__main__':
    run_notebook('battery_capacity_prediction.ipynb')
