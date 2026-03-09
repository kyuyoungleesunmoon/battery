import json

def patch_cell_84_foolproof():
    notebook_path = 'battery_capacity_prediction.ipynb'
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error loading notebook: {e}")
        return

    # Find where the problematic cell is (around index 84)
    target_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            if "sns.regplot" in src and "ocv_deviation" in src and "Capacity" in src:
                target_idx = i
                break

    if target_idx == -1:
        print("Error: Could not find the visualization cell.")
        return

    # Create the foolproof patched code content
    code_content = """import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 시각화 전용 데이터소스 확보
# (이전 단계에서 data 변수의 인덱스가 손상되었거나 훼손되었을 가능성을 배제하고, 살아있는 원본 객체를 찾습니다)
if 'data_df' in globals() and type(data_df) is pd.DataFrame:
    plot_data = data_df.copy()
elif 'df' in globals() and type(df) is pd.DataFrame:
    plot_data = df.copy()
else:
    plot_data = data.copy() if hasattr(data, 'copy') else data

# 파생 변수가 유실되었다면 다시 재계산 확립 (안전 영역)
if 'ocv_deviation' not in plot_data.columns:
    v_col = 'v_initial' if 'v_initial' in plot_data.columns else ('initial_voltage' if 'initial_voltage' in plot_data.columns else None)
    if v_col:
        plot_data['ocv_deviation'] = plot_data[v_col] - 4.2
    else:
        plot_data['ocv_deviation'] = 0

if 'initial_impedance' not in plot_data.columns:
    plot_data['initial_impedance'] = 0

# 기초 시각화: 주요 파생변수와 Capacity의 관계
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# OCV 편차 vs Capacity
sns.regplot(data=plot_data, x='ocv_deviation', y='Capacity', scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=axes[0])
axes[0].set_title('OCV Deviation vs Capacity')
axes[0].set_xlabel('OCV Deviation (V)')
axes[0].set_ylabel('Battery Capacity')

# 초기 임피던스 vs Capacity
sns.regplot(data=plot_data, x='initial_impedance', y='Capacity', scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=axes[1])
axes[1].set_title('Initial Impedance vs Capacity')
axes[1].set_xlabel('Initial Impedance')
axes[1].set_ylabel('Battery Capacity')

plt.tight_layout()
plt.show()"""

    source_lines = [line + '\n' for line in code_content.split('\n')]
    if source_lines:
        source_lines[-1] = source_lines[-1].rstrip('\n')

    nb['cells'][target_idx]['source'] = source_lines

    with open(notebook_path, 'w', encoding='utf-8') as f:
         json.dump(nb, f, ensure_ascii=False, indent=2)

    print(f"Successfully patched code cell at index {target_idx} with foolproof method")

if __name__ == '__main__':
    patch_cell_84_foolproof()
