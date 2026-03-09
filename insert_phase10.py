import json
import os

notebook_path = r'c:\6.1 밧데리_학습\battery_capacity_prediction.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Phase 10 코드 셀들
    def create_markdown_cell(source):
        return {'cell_type': 'markdown', 'metadata': {}, 'source': [source]}

    def create_code_cell(source):
        return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [source]}

    phase10_cells = [
        create_markdown_cell("# Phase 10: 배터리 노화/효율 특성 분석 (추가 분석)\n\n여기서는 모델의 예측 결과와 별개로, 어떤 원인으로 용량(Capacity) 저하가 발생하는지 심층적으로 탐색합니다.\n\n### 1. 전압/임피던스 증감율과 Capacity 저하의 상관성"),
        create_code_cell("import seaborn as sns\nimport matplotlib.pyplot as plt\n\n# 기초 시각화: 주요 파생변수 (ocv_deviation, initial_impedance 등) 와 Capacity의 관계\nfig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n# OCV 편차 vs Capacity\nsns.regplot(data=data, x='ocv_deviation', y='Capacity', scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=axes[0])\naxes[0].set_title('OCV Deviation vs Capacity')\naxes[0].set_xlabel('OCV Deviation (V)')\naxes[0].set_ylabel('Battery Capacity')\n\n# 초기 임피던스 vs Capacity\nsns.regplot(data=data, x='initial_impedance', y='Capacity', scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=axes[1])\naxes[1].set_title('Initial Impedance vs Capacity')\naxes[1].set_xlabel('Initial Impedance')\naxes[1].set_ylabel('Battery Capacity')\n\nplt.tight_layout()\nplt.show()"),
        
        create_markdown_cell("### 2. 고효율/저효율 배터리의 특징량(Feature) 차이 검증"),
        create_code_cell("# Capacity 기준을 정하여 고/저 분류 (예: 중위수 기준)\nmedian_capacity = data['Capacity'].median()\ndata['Capacity_Group'] = ['High' if c >= median_capacity else 'Low' for c in data['Capacity']]\n\n# 그룹별 주요 측정치 비교 Boxplot\nfeatures_to_compare = ['initial_ocv', 'v42_ocv', 'v36_ocv', 'initial_impedance']\n\nfig, axes = plt.subplots(1, 4, figsize=(18, 5))\nfor i, feature in enumerate(features_to_compare):\n    if feature in data.columns:\n        sns.boxplot(data=data, x='Capacity_Group', y=feature, ax=axes[i], palette='Set2')\n        axes[i].set_title(f'{feature} by Capacity')\n\nplt.tight_layout()\nplt.show()"),
    ]

    # 노트북을 순회하며 'Phase 10' 마크다운 셀을 찾고, 그 바로 뒤에 셀들을 삽입합니다.
    insert_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            content = cell['source'][0] if cell['source'] else ''
            if 'Phase 10' in content:
                # Phase 10 선언 셀을 찾음
                insert_idx = i
                break
    
    if insert_idx != -1:
        # 기존 Phase 10 선언 셀의 역할을 대체하거나 이어붙이기 위해 그 뒤에 10 셀 삽입
        print(f"Found Phase 10 at cell {insert_idx}. Inserting codes right after it.")
        nb['cells'] = nb['cells'][:insert_idx+1] + phase10_cells + nb['cells'][insert_idx+1:]
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook updated successfully with Phase 10 cells.")
    else:
        print("Could not find 'Phase 10' markdown cell in the notebook. Appending to the end before Phase 11...")
        # (혹시 못 찾으면 Phase 11 바로 앞에 넣거나 에러 처리)

except Exception as e:
    print(f"Error: {e}")
