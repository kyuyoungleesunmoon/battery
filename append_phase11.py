import json
import os

notebook_path = r'c:\6.1 밧데리_학습\battery_capacity_prediction.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    def create_markdown_cell(source):
        return {'cell_type': 'markdown', 'metadata': {}, 'source': [source]}

    def create_code_cell(source):
        return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [source]}

    new_cells = [
        create_markdown_cell("# Phase 11: 배터리 예측값 기반 클러스터링 검증\n\n1. **목적**: 베터리 특성상 성능이 비슷한 제품끼리 패키징하기 위해 클러스터링을 수행합니다.\n2. **검증**: 효율적인 데이터(초기측정치+완전방전 데이터)를 이용한 예측 결과로 생성한 군집과, 전체 데이터를 이용한 기준(Ground Truth) 군집이 유사한지 검증합니다.\n3. **목표 달성**: 두 클러스터링이 유사하게 검증된다면 목적을 달성한 것으로 간주하고 상세 보고서를 작성합니다."),
        
        create_markdown_cell("### 11-1. 필수 패키지 로드 및 초기 설정"),
        create_code_cell("from sklearn.cluster import KMeans\nfrom sklearn.metrics import adjusted_rand_score, silhouette_score\nfrom sklearn.decomposition import PCA\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler\n\n# KMeans 수행 전 동일한 내부 분할을 위해 시드 고정\nRANDOM_STATE_CLUSTER = 42"),
        
        create_markdown_cell("### 11-2. 원본 데이터를 활용한 기저 클러스터링 (Ground Truth)\n`Capacity`와 원래 특성들을 포함한 전체 피처를 기준으로, 진짜 정답에 해당하는 묶음(분류)을 산출합니다."),
        create_code_cell("# S4_ALL이 앞선 Phase에서 선언되어 있다고 가정합니다. (17개 특징)\n# 혹시 선언되어 있지 않다면 수치형 데이터들로 구성하면 됩니다.\n\n# S4_ALL 피처 배열 복사 및 정답 Capacity 추가\ngt_features = S4_ALL_features.copy() if 'S4_ALL_features' in locals() else list(data.select_dtypes(include=[np.number]).columns)\n\nif 'Capacity' not in gt_features and 'Capacity' in data.columns:\n    gt_features.append('Capacity')\n\n# 결측치 확인 및 제거\n# (이전 단계에서 처리되었으리라 가정하지만 최후 방어코드)\ndata_gt = data[gt_features].copy().dropna()\n\n# 스케일링 수행 (KMeans는 스케일에 민감)\nscaler_gt = StandardScaler()\nX_gt_scaled = scaler_gt.fit_transform(data_gt)\n\n# Elbow Method를 사용하여 최적의 클러스터 수 탐색\ninertias = []\nK_range = range(2, 10)\nfor k in K_range:\n    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE_CLUSTER, n_init='auto')\n    kmeans.fit(X_gt_scaled)\n    inertias.append(kmeans.inertia_)\n\nplt.figure(figsize=(6, 4))\nplt.plot(K_range, inertias, marker='o')\nplt.title('Elbow Method For Optimal k (Ground Truth)')\nplt.xlabel('Number of clusters (k)')\nplt.ylabel('Inertia')\nplt.show()\n\n# ----------------------------------------------------\n# 그래프를 확인한 후 적당한 꺾임 지점(k)을 선택합니다.\n# 여기서는 k=4 라고 가정하고 클러스터링을 진행하겠습니다.\n# ----------------------------------------------------\nN_CLUSTERS = 4\nkmeans_gt = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE_CLUSTER, n_init='auto')\ndata['Cluster_GT'] = kmeans_gt.fit_predict(X_gt_scaled)\nprint(f\"[완료] Ground Truth 클러스터링 (k={N_CLUSTERS})\")"),

        create_markdown_cell("### 11-3. 효율적 데이터의 예측 결과값을 활용한 클러스터링 (Prediction)\n앞선 노트북 Phase 10에서 탐색된 효율적인 데이터 구조 (예: `P1_S1+v25`, 초기+완전방전 데이터)의 **예측 Capacity**를 기준으로 군집을 분류합니다."),
        create_code_cell("# Phase 9/10에서 훈련된 최고 성능 모델의 결과값을 적용한다고 가정\n\n# 이전 코드들에서 정의된 eff_features가 있다고 가정합니다\n# eff_features = phase9_scenarios['P1_S1+v25'] (예시)\n# pred_capacity = best_model.predict(data[eff_features])\n\n# data['Pred_Capacity_Eff'] = pred_capacity\n\n# 효율 데이터 + 예측 값으로 스케일링 및 클러스터링 수행\n# (주석 해제 후, 본인 환경의 변수명에 맞게 실행하세요)\n\n'''\npred_features = eff_features + ['Pred_Capacity_Eff']\n\nscaler_pred = StandardScaler()\nX_pred_scaled = scaler_pred.fit_transform(data[pred_features].dropna())\n\nkmeans_pred = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE_CLUSTER, n_init='auto')\ndata['Cluster_Pred'] = kmeans_pred.fit_predict(X_pred_scaled)\n'''\nprint(\"[안내] 모델 예측 Capacity가 포함된 Cluster_Pred를 위 주석을 풀고 생성해주세요.\")"),

        create_markdown_cell("### 11-4. 두 클러스터 간의 유사도(ARI) 검증 및 평가\n생성된 두 군집(기준군집 vs 예측군집) 간의 유사도를 **Adjusted Rand Index(ARI)** 매트릭스로 정량적 검증합니다.\n* 1에 가까우면 완벽히 일치함을 뜻합니다."),
        create_code_cell("# ARI Score를 통한 클러스터 유사도 확인\n# (주석 해제 후 실행 시 작동합니다)\n\n'''\nari_score = adjusted_rand_score(data['Cluster_GT'], data['Cluster_Pred'])\nprint(f'Adjusted Rand Index (ARI): {ari_score:.4f}')\n\n# 기준 (Threshold)\nif ari_score > 0.6:\n    print(\"검증 성공: 두 클러스터링 분류가 통계적으로 유사합니다. (ARI > 0.6)\")\n    print(\"결론: 초기+완전방전 부분 데이터의 예측 결과만으로도, 전체 데이터의 진짜 성격과 유사한 배터리 패키징 묶음 설계가 가능합니다.\")\nelse:\n    print(\"검증 부족: 원본 묶음과 예측 묶음의 성질 차이가 다소 존재합니다.\")\n'''"),

        create_markdown_cell("### 11-5. 클러스터 분포 산점도(PCA 2D) 형태 시각화\n기준 군집과 예측 군집이 시각적으로 어떻게 다르게 매핑되는지 (2차원으로 투영하여) 눈으로 직접 비교해봅니다."),
        create_code_cell("# PCA로 2차원 축소 (기준 차원은 원본 데이터의 X_gt_scaled를 사용)\n# (위의 이전 셀들을 실행 후 주석을 해제하여 그려주세요)\n\n'''\npca = PCA(n_components=2, random_state=RANDOM_STATE_CLUSTER)\nX_pca = pca.fit_transform(X_gt_scaled)  # 투영할 판은 원본 스케일을 사용\n\ndata['PCA_X'] = X_pca[:, 0]\ndata['PCA_Y'] = X_pca[:, 1]\n\nfig, axes = plt.subplots(1, 2, figsize=(14, 6))\n\n# 1. 원본 (Ground Truth)\nsns.scatterplot(data=data, x='PCA_X', y='PCA_Y', hue='Cluster_GT', palette='Set1', s=50, ax=axes[0])\naxes[0].set_title('Ground Truth Clusters')\n\n# 2. 예측 (Prediction)\nsns.scatterplot(data=data, x='PCA_X', y='PCA_Y', hue='Cluster_Pred', palette='Set1', s=50, ax=axes[1])\naxes[1].set_title(f'Predicted Clusters (ARI: {ari_score:.4f} 이상)')\n\nplt.tight_layout()\nplt.show()\n'''\nprint(\"[안내] 모든 코드가 준비되었습니다. 차례대로 실행하여 검증 지표 및 산점도를 확인해주세요.\")")
    ]

    nb['cells'].extend(new_cells)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print("Notebook updated successfully with Phase 11 cells.")
except Exception as e:
    print(f"Error: {e}")
