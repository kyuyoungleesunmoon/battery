import json
import os

notebook_path = r'c:\6.1 밧데리_학습\battery_capacity_prediction.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 11-2 부분 코드가 있는 셀 찾기
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and len(cell['source']) > 0 and 'S4_ALL이 앞선 Phase' in cell['source'][0]:
            print(f'Found 11-2 Cell at index {i}')
            
            # 셀 내용 모두 덮어씌움
            new_source = """# S4_ALL이 앞선 Phase에서 선언되어 있다고 가정합니다. (17개 특징)
# 'data' 변수의 타입 에러를 방지하기 위해, 원본 데이터프레임을 명시적으로 df로 불러오거나 타입을 확인합니다.

# 'data'가 DataFrame이 아니고 Series일 경우(위 셀 등에서 덮어씌워진 경우),
# 원본 df 이름이 'df_clean' 혹은 'df' 일 수 있으므로 이를 새로 할당합니다.
if 'df' in locals() and type(df).__name__ == 'DataFrame':
    data_df = df.copy()
else:
    # 만약 df라는 이름도 없다면 csv 다시 불러오기
    import pandas as pd
    try:
        data_df = pd.read_csv('data.csv')
    except:
        data_df = pd.DataFrame() # 최후의 방어
        print('경고: data.csv를 찾지 못했습니다.')

gt_features = S4_ALL_features.copy() if 'S4_ALL_features' in locals() else list(data_df.select_dtypes(include=[np.number]).columns)

if 'Capacity' not in gt_features and 'Capacity' in data_df.columns:
    gt_features.append('Capacity')

# 결측치 확인 및 제거 (원본 인덱스 유지)
data_gt = data_df[gt_features].copy().dropna()

# 스케일링 수행
scaler_gt = StandardScaler()
X_gt_scaled = scaler_gt.fit_transform(data_gt)

# 최적 군집 수 탐색
inertias = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE_CLUSTER, n_init='auto')
    kmeans.fit(X_gt_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertias, marker='o')
plt.title('Elbow Method For Optimal k (Ground Truth)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()

# k=4 적용
N_CLUSTERS = 4
kmeans_gt = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE_CLUSTER, n_init='auto')
data_gt_labels = kmeans_gt.fit_predict(X_gt_scaled)

# dropna로 남은 인덱스에만 매핑
data_df.loc[data_gt.index, 'Cluster_GT'] = data_gt_labels

# 사용자가 다음 셀들에서 'data' 변수를 쓴다면 DataFrame 형태로 다시 돌려줌
data = data_df.copy()

print(f"[완료] Ground Truth 클러스터링 (k={N_CLUSTERS})")
"""
            nb['cells'][i]['source'] = [line + '\n' for line in new_source.split('\n')][:-1]
            break

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print('11-2 Cell replaced successfully.')

except Exception as e:
    print(f'Error: {e}')
