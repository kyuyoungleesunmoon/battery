import json

def patch_subsequent_capacity_errors():
    notebook_path = 'battery_capacity_prediction.ipynb'
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error loading notebook: {e}")
        return

    # Find the problematic cell (boxplot capacity grouping)
    target_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            if "median_capacity = data['Capacity'].median()" in src:
                target_idx = i
                break

    if target_idx == -1:
        print("Error: Could not find the boxplot visualization cell.")
        return

    # Create the foolproof patched code content
    code_content = """import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 시각화 전용 데이터소스 확보 (원본 훼손 방지)
if 'data_df' in globals() and type(data_df) is pd.DataFrame:
    plot_data = data_df.copy()
elif 'df' in globals() and type(df) is pd.DataFrame:
    plot_data = df.copy()
else:
    plot_data = data.copy() if hasattr(data, 'copy') else data

# 타겟 변수 이름 자동 추출 (대소문자 방어 로직)
TARGET = 'capacity' if 'capacity' in plot_data.columns else ('Capacity' if 'Capacity' in plot_data.columns else 'capacity [Ah]')

# Capacity 기준을 정하여 고/저 분류 (예: 중위수 기준)
median_capacity = plot_data[TARGET].median()
plot_data['Capacity_Group'] = ['High' if c >= median_capacity else 'Low' for c in plot_data[TARGET]]

# 그룹별 주요 측정치 비교 Boxplot
features_to_compare = ['initial_ocv', 'v42_ocv', 'v36_ocv', 'initial_impedance']

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for i, feature in enumerate(features_to_compare):
    if feature in plot_data.columns:
        sns.boxplot(data=plot_data, x='Capacity_Group', y=feature, ax=axes[i], palette='Set2')
        axes[i].set_title(f'{feature} by Capacity')
    else:
        axes[i].set_title(f'{feature} (Not Found)')

plt.tight_layout()
plt.show()"""

    source_lines = [line + '\n' for line in code_content.split('\n')]
    if source_lines:
        source_lines[-1] = source_lines[-1].rstrip('\n')

    nb['cells'][target_idx]['source'] = source_lines

    with open(notebook_path, 'w', encoding='utf-8') as f:
         json.dump(nb, f, ensure_ascii=False, indent=2)

    print(f"Successfully patched code cell at index {target_idx} with foolproof target selection.")

if __name__ == '__main__':
    patch_subsequent_capacity_errors()
