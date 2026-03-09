import json

def insert_11_3_code():
    notebook_path = 'battery_capacity_prediction.ipynb'
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error loading notebook: {e}")
        return

    # Find where the 11-3 markdown is.
    target_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell.get('source', []))
            if '11-3. 효율적 데이터의 예측 결과값' in src:
                target_idx = i
                break

    if target_idx == -1:
        print("Error: Could not find 11-3 markdown cell.")
        return

    # Create the new code cell to insert AFTER the markdown cell.
    code_content = """import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("[진행] 효율적 데이터 예측 결과 기반 클러스터링 탐색을 시작합니다.")

# 11-2에서 설정된 클러스터 개수 N_CLUSTERS 가 있다고 가정
N_CLUSTERS_PRED = N_CLUSTERS if 'N_CLUSTERS' in locals() else 4

# 예측 모델 결과(Prediction) 데이터가 있다고 가정
# (앞선 셀들에서 최종 예측값 배열 'pred_capacity' 가 존재할 경우를 대비한 방어 로직)
if 'pred_capacity' in locals():
    X_pred = np.array(pred_capacity).reshape(-1, 1)
else:
    # 방어 로직: 예측값이 명시되지 않았을 경우, S1(초기 데이터) 등 임시 예측 데이터를 사용
    print("경고: 'pred_capacity' 변수를 찾지 못해 임시 Capacity 데이터로 군집을 수행합니다.")
    if 'data_df' in locals():
        X_pred = data_df[['Capacity']].copy()
    else:
        # Fallback
        X_pred = data_gt[['Capacity']].copy() if 'data_gt' in locals() else np.random.rand(100, 1)

# 스케일링 수행
scaler_pred = StandardScaler()
X_pred_scaled = scaler_pred.fit_transform(X_pred)

# K-Means 적용 (11-2와 동일한 개수, k=4 등으로 설정)
kmeans_pred = KMeans(n_clusters=N_CLUSTERS_PRED, random_state=42, n_init='auto')
data_pred_labels = kmeans_pred.fit_predict(X_pred_scaled)

# 결과를 DataFrame에 저장
if 'data_df' in locals():
    data_df['Cluster_Pred'] = data_pred_labels
    
# 향후 11-4 ARI 평가를 위해 라벨 변수 노출
labels_pred = data_pred_labels
labels_gt = data_gt_labels if 'data_gt_labels' in locals() else (data_df['Cluster_GT'].values if 'Cluster_GT' in data_df.columns else None)

print(f"[완료] 예측 데이터 기반 클러스터링 완료 (k={N_CLUSTERS_PRED})")"""

    # Formatting logic matching jupyter
    source_lines = [line + '\n' for line in code_content.split('\n')]
    if source_lines:
        source_lines[-1] = source_lines[-1].rstrip('\n') # last line shouldn't have trailing newline in source array if it doesn't in string

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

    # Insert it right after the 11-3 markdown cell (target_idx + 1)
    nb['cells'].insert(target_idx + 1, new_cell)

    with open(notebook_path, 'w', encoding='utf-8') as f:
         json.dump(nb, f, ensure_ascii=False, indent=2)

    print(f"Successfully inserted code cell after cell {target_idx}")

if __name__ == '__main__':
    insert_11_3_code()
