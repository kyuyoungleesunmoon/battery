import json

with open(r"c:\6.1 밧데리_학습\train.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get("cells", [])):
    if cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        if "plot_model" in source:
            print(f"--- Cell {i} ---")
            print(source)
