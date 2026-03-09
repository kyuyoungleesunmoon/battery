import json

def patch_notebook():
    notebook_path = 'battery_capacity_prediction.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Cell 85: # Phase 10: 배터리 노화/효율 특성 분석 (추가 분석)
    # -> ## 10-3. 배터리 노화/효율 특성 분석 (추가 분석)
    if '# Phase 10: 배터리 노화/효율 특성 분석' in nb['cells'][85]['source'][0]:
        nb['cells'][85]['source'][0] = nb['cells'][85]['source'][0].replace(
            '# Phase 10: 배터리 노화/효율 특성 분석 (추가 분석)',
            '## 10-3. 배터리 노화/효율 특성 분석 (추가 분석)'
        )

    # 2. Cell 86: ### 10-2. 고효율/저효율 배터리의 특징량(Feature) 차이 검증
    # -> ### 10-3-1. 고효율/저효율 배터리의 특징량(Feature) 차이 검증
    for i, line in enumerate(nb['cells'][86]['source']):
        if '### 10-2.' in line:
            nb['cells'][86]['source'][i] = line.replace('### 10-2.', '### 10-3-1.')

    # 3. Cell 89: ### 10-3. 최종 보고서 자동 생성
    # -> ## 10-4. 최종 보고서 자동 생성
    for i, line in enumerate(nb['cells'][89]['source']):
        if '### 10-3. 최종 보고서 자동 생성' in line:
            nb['cells'][89]['source'][i] = line.replace('### 10-3.', '## 10-4.')

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

    print("Successfully patched battery_capacity_prediction.ipynb")

if __name__ == '__main__':
    patch_notebook()
