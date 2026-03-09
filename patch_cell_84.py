import json

def patch_cell_84():
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
            if "sns.regplot(data=data, x='ocv_deviation', y='Capacity'" in src:
                target_idx = i
                break

    if target_idx == -1:
        print("Error: Could not find the visualization cell.")
        return

    # Create the patched code content
    code_content = """import seaborn as sns
import matplotlib.pyplot as plt

# --- 파생변수 복원 (시각화용) 방어 로직 ---
# 이전 Phase들에서 data 변수가 축소/변경되면서 ocv_deviation이 유실되었을 수 있으므로 복원 시도
if 'ocv_deviation' not in data.columns:
    # 1. 만약 원본 df/data_df가 살아있고 거기엔 있다면 가져오기
    if 'data_df' in globals() and 'ocv_deviation' in data_df.columns:
        data['ocv_deviation'] = data_df['ocv_deviation']
    elif 'df' in globals() and 'ocv_deviation' in df.columns:
        data['ocv_deviation'] = df['ocv_deviation']
    # 2. 그것도 안된다면 직접 재계산
    else:
        v_col = 'v_initial' if 'v_initial' in data.columns else ('initial_voltage' if 'initial_voltage' in data.columns else None)
        if v_col:
            NOMINAL_V = 4.2 # Phase 2에서 주로 사용된 단위
            data['ocv_deviation'] = data[v_col] - NOMINAL_V
        else:
            print("경고: 전압 컬럼을 찾을 수 없어 ocv_deviation 계산 불가. 0으로 임시 대체함")
            data['ocv_deviation'] = 0

if 'initial_impedance' not in data.columns:
    # df나 data_df에서 가져오기 시도
    if 'data_df' in globals() and 'initial_impedance' in data_df.columns:
        data['initial_impedance'] = data_df['initial_impedance']
    elif 'df' in globals() and 'initial_impedance' in df.columns:
        data['initial_impedance'] = df['initial_impedance']
    else:
         print("경고: initial_impedance 컬럼을 찾을 수 없습니다. 0으로 임시 대체함")
         data['initial_impedance'] = 0
# ----------------------------------------------

# 기초 시각화: 주요 파생변수 (ocv_deviation, initial_impedance 등) 와 Capacity의 관계
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# OCV 편차 vs Capacity
sns.regplot(data=data, x='ocv_deviation', y='Capacity', scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=axes[0])
axes[0].set_title('OCV Deviation vs Capacity')
axes[0].set_xlabel('OCV Deviation (V)')
axes[0].set_ylabel('Battery Capacity')

# 초기 임피던스 vs Capacity
sns.regplot(data=data, x='initial_impedance', y='Capacity', scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=axes[1])
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

    print(f"Successfully patched code cell at index {target_idx}")

if __name__ == '__main__':
    patch_cell_84()
