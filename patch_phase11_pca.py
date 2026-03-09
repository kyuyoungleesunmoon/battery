import json

def patch_phase_11_pca():
    notebook_path = 'battery_capacity_prediction.ipynb'
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error loading notebook: {e}")
        return

    # Find the PCA scatter cell
    target_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            if "sns.scatterplot(data=data, x='PCA_X', y='PCA_Y', hue='Cluster_GT'" in src:
                target_idx = i
                break

    if target_idx == -1:
        print("Error: Could not find the PCA scatter plot cell.")
        return

    # Create the foolproof patched code content
    code_content = """# PCA로 2차원 축소 (기준 차원은 원본 데이터의 X_gt_scaled를 사용)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- 시각화 전용 데이터소스 확보 및 동기화 방어 로직 ---
if 'data_df' in globals():
    plot_data = data_df.copy()
else:
    plot_data = data.copy()

if 'Cluster_Pred' not in plot_data.columns and 'data' in globals() and 'Cluster_Pred' in data.columns:
    plot_data['Cluster_Pred'] = data['Cluster_Pred']
if 'Cluster_GT' not in plot_data.columns and 'data' in globals() and 'Cluster_GT' in data.columns:
    plot_data['Cluster_GT'] = data['Cluster_GT']
        
if 'Cluster_Pred' not in plot_data.columns or 'Cluster_GT' not in plot_data.columns:
    print("경고: 'Cluster_GT' 또는 'Cluster_Pred' 데이터를 찾을 수 없어 클러스터별 색상을 표시할 수 없습니다.")
    has_labels = False
else:
    has_labels = True
# ----------------------------------------------

# PCA 수행
if 'X_gt_scaled' in globals():
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_gt_scaled)  # 투영할 판은 원본 스케일을 사용

    plot_data['PCA_X'] = X_pca[:, 0]
    plot_data['PCA_Y'] = X_pca[:, 1]
else:
    print("경고: 'X_gt_scaled' 데이터가 없어 PCA를 적용할 수 없습니다. Phase 11-2를 먼저 실행해주세요.")
    has_labels = False

if has_labels:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 원본 (Ground Truth)
    sns.scatterplot(data=plot_data, x='PCA_X', y='PCA_Y', hue='Cluster_GT', palette='Set1', s=50, ax=axes[0])
    axes[0].set_title('Ground Truth Clusters')

    # 2. 예측 (Prediction)
    sns.scatterplot(data=plot_data, x='PCA_X', y='PCA_Y', hue='Cluster_Pred', palette='Set1', s=50, ax=axes[1])
    
    # 앞선 셀에서 ari_score가 등록되어 있을 경우만 표시
    if 'ari_score' in globals():
        axes[1].set_title(f'Predicted Clusters (ARI: {ari_score:.4f})')
    else:
        axes[1].set_title('Predicted Clusters')

    plt.tight_layout()
    plt.show()"""

    source_lines = [line + '\n' for line in code_content.split('\n')]
    if source_lines:
        source_lines[-1] = source_lines[-1].rstrip('\n')

    nb['cells'][target_idx]['source'] = source_lines

    with open(notebook_path, 'w', encoding='utf-8') as f:
         json.dump(nb, f, ensure_ascii=False, indent=2)

    print(f"Successfully patched PCA plot code cell at index {target_idx} with cluster synchronization.")

if __name__ == '__main__':
    patch_phase_11_pca()
