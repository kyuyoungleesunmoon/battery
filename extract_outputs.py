"""
노트북 셀 출력(텍스트 + 이미지)을 추출하여 report_outputs/ 폴더에 저장합니다.
Phase 12~13 셀의 실행 결과를 Streamlit에서 정적으로 보여주기 위한 전처리 스크립트입니다.
"""
import json, os, base64

def extract_outputs():
    notebook_path = 'battery_capacity_prediction.ipynb'
    output_dir = 'report_outputs'
    os.makedirs(output_dir, exist_ok=True)

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Phase 12~13 관련 셀들을 식별하는 키워드
    phase_keywords = [
        ('phase12_1', 'Phase 12-1: 다변량 예측'),
        ('phase12_2', 'Phase 12-2: UMAP'),
        ('phase12_3', 'Phase 12-3: 클러스터링 알고리즘'),
        ('phase12_4', 'Phase 12-4: 종합 비교'),
        ('phase13_1', 'Phase 13: 최종 결론 리포트'),
        ('phase13_2', '13-2. 비용·시간 절감'),
        ('phase13_3', '최종 결론'),  # 마지막 셀
    ]

    # 모든 code 셀에서 Phase 12~13과 관련된 것들을 추출
    report_data = []
    
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        
        # 마크다운 셀: Phase 12~13 제목
        if cell['cell_type'] == 'markdown':
            if 'Phase 12' in src or 'Phase 13' in src or '12-' in src or '13-' in src:
                report_data.append({
                    'type': 'markdown',
                    'cell_index': i,
                    'content': src,
                })
            continue
        
        if cell['cell_type'] != 'code':
            continue

        # Phase 12~13 코드 셀인지 확인
        phase_id = None
        for pid, keyword in phase_keywords:
            if keyword in src:
                phase_id = pid
                break
        
        if phase_id is None:
            continue

        outputs = cell.get('outputs', [])
        if not outputs:
            continue

        # 출력 추출
        text_outputs = []
        image_files = []
        img_counter = 0

        for out in outputs:
            # 텍스트 출력
            if out.get('output_type') == 'stream':
                text_outputs.append(''.join(out.get('text', [])))
            elif out.get('output_type') == 'execute_result':
                data = out.get('data', {})
                if 'text/plain' in data:
                    text_outputs.append(''.join(data['text/plain']) if isinstance(data['text/plain'], list) else data['text/plain'])
            
            # 이미지 출력
            if out.get('output_type') in ('display_data', 'execute_result'):
                data = out.get('data', {})
                if 'image/png' in data:
                    img_b64 = data['image/png']
                    if isinstance(img_b64, list):
                        img_b64 = ''.join(img_b64)
                    img_filename = f'{phase_id}_img{img_counter}.png'
                    img_path = os.path.join(output_dir, img_filename)
                    with open(img_path, 'wb') as img_f:
                        img_f.write(base64.b64decode(img_b64))
                    image_files.append(img_filename)
                    img_counter += 1

        report_data.append({
            'type': 'code_output',
            'phase_id': phase_id,
            'cell_index': i,
            'text': '\n'.join(text_outputs),
            'images': image_files,
        })

    # 메타데이터 저장
    meta_path = os.path.join(output_dir, 'report_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # 통계
    n_text = sum(1 for r in report_data if r['type'] == 'code_output' and r.get('text'))
    n_img = sum(len(r.get('images', [])) for r in report_data)
    print(f'추출 완료: {len(report_data)}개 항목, 텍스트 {n_text}개, 이미지 {n_img}개')
    print(f'저장 위치: {output_dir}/')

if __name__ == '__main__':
    extract_outputs()
