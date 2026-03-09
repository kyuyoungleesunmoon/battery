import json

FONT_BLOCK = '''# --- 한글 폰트 설정 (Windows/Linux/Mac 호환) ---
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform, os

def set_korean_font():
    """시스템에 맞는 한글 폰트를 자동 탐지하여 matplotlib에 설정합니다."""
    system = platform.system()
    candidate_fonts = []
    
    if system == 'Windows':
        candidate_fonts = ['Malgun Gothic', 'NanumGothic', 'Gulim', 'Dotum', 'Batang']
    elif system == 'Darwin':
        candidate_fonts = ['AppleGothic', 'NanumGothic']
    else:
        candidate_fonts = ['NanumGothic', 'NanumBarunGothic', 'UnDotum']
    
    available = set(f.name for f in fm.fontManager.ttflist)
    for font_name in candidate_fonts:
        if font_name in available:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            print(f'[폰트] {font_name} 설정 완료')
            return font_name
    
    # 후보에 없으면 직접 ttf 파일 탐색 (Windows)
    if system == 'Windows':
        font_dir = os.path.join(os.environ.get('WINDIR', 'C:\\\\Windows'), 'Fonts')
        for ttf_name in ['malgun.ttf', 'NanumGothic.ttf', 'gulim.ttc']:
            ttf_path = os.path.join(font_dir, ttf_name)
            if os.path.exists(ttf_path):
                fm.fontManager.addfont(ttf_path)
                prop = fm.FontProperties(fname=ttf_path)
                plt.rcParams['font.family'] = prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False
                print(f'[폰트] {prop.get_name()} (ttf 직접 로드) 설정 완료')
                return prop.get_name()
    
    print('[경고] 한글 폰트를 찾지 못했습니다. 한글이 깨질 수 있습니다.')
    return None

KOREAN_FONT = set_korean_font()
# --- 한글 폰트 설정 끝 ---
'''

def patch_font():
    notebook_path = 'battery_capacity_prediction.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    patched_count = 0
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell.get('source', []))
        
        # Phase 12-1, 12-2, 12-3, 12-4, 13-1, 13-2 셀들 찾기
        # 이미 set_korean_font 가 들어있으면 건너뜀
        if 'set_korean_font' in src:
            continue
        
        needs_patch = False
        
        # Phase 12-1: 첫 번째 큰 코드 셀
        if 'Phase 12-1: 다변량 예측' in src:
            needs_patch = True
        # Phase 12-2
        elif 'Phase 12-2: UMAP' in src:
            needs_patch = True
        # Phase 12-3
        elif 'Phase 12-3: 클러스터링 알고리즘' in src:
            needs_patch = True
        # Phase 12-4
        elif 'Phase 12-4: 종합 비교' in src:
            needs_patch = True
        # Phase 13-1
        elif 'Phase 13: 최종 결론 리포트' in src:
            needs_patch = True
        # Phase 13-2
        elif '13-2. 비용·시간 절감' in src:
            needs_patch = True
        
        if needs_patch:
            # 기존 plt.rcParams 행 제거하고 FONT_BLOCK 삽입
            lines = src.split('\n')
            new_lines = []
            font_inserted = False
            skip_old_font = False
            
            for line in lines:
                # 기존 단순 폰트 설정 삭제
                if "plt.rcParams['font.family']" in line or "plt.rcParams['axes.unicode_minus']" in line:
                    skip_old_font = True
                    continue
                
                # import matplotlib.pyplot as plt 뒤에 폰트 블록 삽입
                if not font_inserted and ('import matplotlib.pyplot as plt' in line):
                    # 이 line은 FONT_BLOCK에 이미 포함되어 있으므로 건너뜀
                    font_inserted = True
                    new_lines.append(FONT_BLOCK.rstrip())
                    continue
                
                new_lines.append(line)
            
            if not font_inserted:
                # import문을 못 찾은 경우 맨 앞에 삽입
                new_lines = [FONT_BLOCK.rstrip()] + new_lines
            
            new_src = '\n'.join(new_lines)
            source_lines = [line + '\n' for line in new_src.split('\n')]
            if source_lines:
                source_lines[-1] = source_lines[-1].rstrip('\n')
            
            nb['cells'][i]['source'] = source_lines
            patched_count += 1
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    
    print(f'Successfully patched {patched_count} cells with Korean font configuration.')

if __name__ == '__main__':
    patch_font()
