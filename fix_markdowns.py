import json

notebook_path = r'c:\6.1 밧데리_학습\battery_capacity_prediction.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    
    # 1. 대상 인덱스별 교체 내용 지정 (phase_cells_dump.txt 기반 분석)
    # [M 85] ### 10-3. 최종 보고서 자동 생성 (기존 유지)
    # [M 87] ### 2. 고효율/저효율 배터리의 특징량(Feature) 차이 검증 -> 10-2 (기존 10-2였어야 함)
    # [M 101] # Phase 11: 배터리 노화/효율 특성 분석 (추가 분석) -> "# Phase 10: 배터리 노화/효율 특성 분석 (추가 분석)" (이름 오류)

    modifications = {
        87: "### 10-2. 고효율/저효율 배터리의 특징량(Feature) 차이 검증",
        98: "### 11-5. 클러스터 시각화 및 운영 인사이트",  
        # 100 은 "### 11-5. 클러스터 분포 산점도(PCA 2D) 형태 시각화" 로 중복됨. 삭제 혹은 11-6으로. (아래에서 병합/삭제 처리)
        101: "# Phase 10: 배터리 노화/효율 특성 분석 (추가 분석)"
    }

    # 셀의 인덱스 기준 교정 수행 (dump 인덱스는 슬라이싱 없이 추출된 상태 기준)
    for idx, new_text in modifications.items():
        if cells[idx]['cell_type'] == 'markdown':
            cells[idx]['source'] = [new_text]
            print(f"[{idx}] 셀 제목 수정 완료: {new_text}")

    # 순서 재배치를 위한 구획 나누기
    # Phase 10의 헤더가 101에 배치되어 있고, 내부 내용은 87등에 있음.
    # 올바른 순서:
    # 101 (Phase 10 헤더) + 87 (10-2 마크다운) + 88, 89 (관련 코드/결과) + 85 (10-3 마크다운) + 86 (관련 코드)
    # 그리고 Phase 11 블록 (90 번대 부터 100까지)
    
    # Phase 10 파트 조립
    part_10_header = [cells[101]]  # # Phase 10: 배터리 노화/효율...
    part_10_body1 = cells[87:90]   # 10-2. 고효율/저효율 특성 + 코드 x 2
    part_10_body2 = cells[85:87]   # 10-3 최종 보고서 + 코드

    # Phase 11 파트 조립
    # 90 부터 97 까지 연속 (11 헤더 ~ 11-4)
    # 98 (11-5. 시각화 인사이트) + 99 (코드) + 100 (11-5 PCA 헤더 중복 -> 삭제 혹은 병합)
    part_11_main = cells[90:98]
    
    # 중복 헤더 통합 (98 제목을 남기고 100은 버림)
    part_11_viz = [cells[98], cells[99]]
    
    # 조립된 새로운 뒷부분
    reordered_tail = part_10_header + part_10_body1 + part_10_body2 + part_11_main + part_11_viz
    
    # 앞부분 유지 (0 ~ 84까지)
    new_cells = cells[:85] + reordered_tail

    nb['cells'] = new_cells

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print("Notebook cell reordering and header fixing completed successfully! (Phase 10 -> Phase 11 순서 확립)")

except Exception as e:
    print(f"Error: {e}")
