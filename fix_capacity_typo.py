import json
import re

def fix_all_capacity_typographical_errors():
    notebook_path = 'battery_capacity_prediction.ipynb'
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error loading notebook: {e}")
        return

    changes_made = 0
    # Process cells from the beginning of Phase 10 (approx index 80 to the end)
    # Be careful not to replace markdown cells blindly, only code cells.
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source_lines = cell.get('source', [])
            new_source_lines = []
            for line in source_lines:
                # If we see hardcoded indexing for Capacity and we know the col is capacity
                # We do a basic string replacement.
                # data['Capacity'] -> data['capacity']
                # df['Capacity'] -> df['capacity']
                # df_cluster['Capacity'] -> df_cluster['capacity']
                original_line = line
                line = line.replace("data['Capacity']", "data['capacity']")
                line = line.replace('data["Capacity"]', "data['capacity']")
                line = line.replace("df['Capacity']", "df['capacity']")
                line = line.replace('df["Capacity"]', "df['capacity']")
                line = line.replace("['Capacity']", "['capacity']")
                
                if line != original_line:
                    changes_made += 1
                new_source_lines.append(line)
                
            cell['source'] = new_source_lines

    if changes_made > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
             json.dump(nb, f, ensure_ascii=False, indent=2)
        print(f"Successfully replaced {changes_made} instances of hardcoded 'Capacity' with 'capacity'.")
    else:
        print("No remaining instances of hardcoded 'Capacity' found.")

if __name__ == '__main__':
    fix_all_capacity_typographical_errors()
