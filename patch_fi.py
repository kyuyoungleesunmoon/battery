import json

notebook_path = r'c:\6.1 밧데리_학습\train.ipynb'
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb.get('cells', [])):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if "plot_model(tuned_model, plot='feature')" in source:
                new_source = [
                    "# 피처 중요도 시각화 (Feature Importance)\n",
                    "try:\n",
                    "    # 기본 지원되는 모델의 경우\n",
                    "    plot_model(tuned_model, plot='feature')\n",
                    "except TypeError:\n",
                    "    # 기본 피처 중요도를 지원하지 않는 모델일 경우 Permutation Importance 사용\n",
                    "    print('해당 모델은 기본 피처 중요도를 지원하지 않아 Permutation Importance로 대체합니다.')\n",
                    "    from sklearn.inspection import permutation_importance\n",
                    "    import pandas as pd\n",
                    "    import matplotlib.pyplot as plt\n",
                    "    from pycaret.regression import get_config\n\n",
                    "    X_train = get_config('X_train')\n",
                    "    y_train = get_config('y_train')\n",
                    "    result = permutation_importance(tuned_model, X_train, y_train, n_repeats=10, random_state=42)\n",
                    "    importance = pd.Series(result.importances_mean, index=X_train.columns).sort_values(ascending=True)\n",
                    "    \n",
                    "    plt.figure(figsize=(10, 6))\n",
                    "    importance.plot(kind='barh', color='skyblue')\n",
                    "    plt.title('Permutation Feature Importance')\n",
                    "    plt.xlabel('Importance')\n",
                    "    plt.ylabel('Features')\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n"
                ]
                cell['source'] = new_source
                print(f"Cell {i} updated.")

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1, separators=(',', ': '))
    
    print("REPLACED_SUCCESSFULLY")
except Exception as e:
    print(f"Error: {e}")
