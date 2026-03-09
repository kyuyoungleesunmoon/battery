import json

def main():
    try:
        with open('train.ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print("Error:", e)
        return

    with open('phases_output.txt', 'w', encoding='utf-8') as out_f:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'markdown':
                source = cell.get('source', [])
                if source:
                    text = source[0].strip()
                    if text.startswith('# Phase'):
                        out_f.write(f"Cell {i}: {text}\n")
                    elif 'Phase 10' in ''.join(source) or 'Phase10' in ''.join(source):
                        out_f.write(f"Cell {i} (contains Phase 10): {text[:60].replace('\n', ' ')}...\n")
                    elif 'Phase 11' in ''.join(source) or 'Phase11' in ''.join(source):
                        out_f.write(f"Cell {i} (contains Phase 11): {text[:60].replace('\n', ' ')}...\n")

if __name__ == '__main__':
    main()
