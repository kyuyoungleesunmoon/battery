import json

def patch_phase11_ari():
    notebook_path = 'battery_capacity_prediction.ipynb'
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error loading notebook: {e}")
        return

    # Find the ARI score cell
    target_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            if "ari_score = adjusted_rand_score(data['Cluster_GT'], data['Cluster_Pred'])" in src:
                target_idx = i
                break

    if target_idx == -1:
        print("Error: Could not find the ARI cell.")
        return

    # Create the foolproof patched code content
    code_content = """# ARI Score를 통한 클러스터 유사도 확인

from sklearn.metrics import adjusted_rand_score

# --- 클러스터 변수 동기화 방어 로직 ---
# 이전 셀에서 Cluster_Pred가 data_df에는 저장되었으나 data 객체로 넘어오지 않았을 경우를 대비합니다.
if 'data_df' in globals():
    if 'Cluster_Pred' not in data.columns and 'Cluster_Pred' in data_df.columns:
        data['Cluster_Pred'] = data_df['Cluster_Pred']
    if 'Cluster_GT' not in data.columns and 'Cluster_GT' in data_df.columns:
        data['Cluster_GT'] = data_df['Cluster_GT']
        
if 'Cluster_Pred' not in data.columns or 'Cluster_GT' not in data.columns:
    print("경고: 'Cluster_GT' 또는 'Cluster_Pred' 데이터가 없습니다. 11-2, 11-3 셀이 정상적으로 실행되었는지 확인해주세요.")
else:
    # ----------------------------------------------
    
    ari_score = adjusted_rand_score(data['Cluster_GT'], data['Cluster_Pred'])
    print(f'Adjusted Rand Index (ARI): {ari_score:.4f}')
    
    # 기준 (Threshold)
    if ari_score > 0.6:
        print("검증 성공: 두 클러스터링 분류가 통계적으로 유사합니다. (ARI > 0.6)")
        print("결론: 초기+완전방전 부분 데이터의 예측 결과만으로도, 전체 데이터의 진짜 성격과 유사한 배터리 패키징 묶음 설계가 가능합니다.")
    else:
        print("검증 부족: 원본 묶음과 예측 묶음의 성질 차이가 다소 존재합니다.")"""

    source_lines = [line + '\n' for line in code_content.split('\n')]
    if source_lines:
        source_lines[-1] = source_lines[-1].rstrip('\n')

    nb['cells'][target_idx]['source'] = source_lines

    with open(notebook_path, 'w', encoding='utf-8') as f:
         json.dump(nb, f, ensure_ascii=False, indent=2)

    print(f"Successfully patched ARI code cell at index {target_idx} with cluster synchronization.")

if __name__ == '__main__':
    patch_phase11_ari()
