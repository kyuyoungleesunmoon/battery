import json

notebook_path = r'c:\6.1 밧데리_학습\battery_capacity_prediction.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    
    # 1. Phase 10과 Phase 11 셀들의 시작/끝 인덱스 찾기
    phase10_start = -1
    phase10_end = -1
    phase11_start = -1
    phase11_end = -1
    
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            content = cell['source'][0] if cell['source'] else ''
            
            # Phase 10 헤더 찾기
            if content.startswith('# Phase 10:'):
                phase10_start = i
            # Phase 11 헤더 찾기
            elif content.startswith('# Phase 11:'):
                phase11_start = i

    # 끝 인덱스 찾기 로직: 
    # Phase 10은 그 다음 Phase 헤더 전이나 끝까지
    if phase10_start != -1:
        for i in range(phase10_start + 1, len(cells)):
            content = cells[i]['source'][0] if cells[i]['source'] else ''
            if cells[i]['cell_type'] == 'markdown' and content.startswith('# Phase '):
                phase10_end = i
                break
        if phase10_end == -1:
            phase10_end = len(cells)

    if phase11_start != -1:
        for i in range(phase11_start + 1, len(cells)):
            content = cells[i]['source'][0] if cells[i]['source'] else ''
            if cells[i]['cell_type'] == 'markdown' and content.startswith('# Phase '):
                phase11_end = i
                break
        if phase11_end == -1:
            phase11_end = len(cells)

    print(f"Phase 10: {phase10_start} ~ {phase10_end}")
    print(f"Phase 11: {phase11_start} ~ {phase11_end}")

    # 순서가 잘못되었는지 확인 (Phase 11이 Phase 10보다 앞에 있는 경우)
    if phase11_start != -1 and phase10_start != -1 and phase11_start < phase10_start:
        print("순서 뒤바뀜 감지. 재배열 시작...")
        
        # 1. 앞부분 (Phase 11 이전)
        part1 = cells[:phase11_start]
        # 2. Phase 11 부분
        part_phase11 = cells[phase11_start:phase11_end]
        
        # Phase 11과 10 사이에 다른 내용이 있다면 그 부분
        part_middle = cells[phase11_end:phase10_start]
        
        # 3. Phase 10 부분
        part_phase10 = cells[phase10_start:phase10_end]
        
        # 4. 뒷부분 (Phase 10 이후)
        part_last = cells[phase10_end:]

        # 올바른 순서로 재조립: part1 + part_middle + Phase 10 + Phase 11 + part_last
        new_cells = part1 + part_middle + part_phase10 + part_phase11 + part_last
        nb['cells'] = new_cells
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("순서 교정 완료: Phase 10 -> Phase 11")
    else:
        print("순서가 이미 정상이거나 셀을 찾지 못했습니다.")

except Exception as e:
    print(f"Error: {e}")
