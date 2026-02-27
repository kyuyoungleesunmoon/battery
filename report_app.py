"""
배터리 스마트 패키징 리포트 — 노트북 출력 기반 Streamlit 뷰어
노트북의 Phase 12~13 셀 출력(텍스트+차트)을 정적으로 표시합니다.
출력이 없을 경우 실시간 분석으로 Fallback합니다.

사용법:
  1. 노트북을 실행하여 Phase 12~13 출력을 생성합니다.
  2. python extract_outputs.py   ← 출력을 추출합니다.
  3. streamlit run report_app.py ← 리포트를 봅니다.
"""
import streamlit as st
import json, os, base64
from pathlib import Path

st.set_page_config(page_title="배터리 스마트 패키징 리포트", page_icon="🔋",
                   layout="wide", initial_sidebar_state="expanded")

OUTPUT_DIR = Path('report_outputs')
META_FILE = OUTPUT_DIR / 'report_meta.json'

# ============================================================
# 1. 노트북 출력 로드 시도
# ============================================================
@st.cache_data
def load_notebook_outputs():
    """report_outputs/report_meta.json에서 추출된 출력을 로드합니다."""
    if not META_FILE.exists():
        return None
    with open(META_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 유효한 출력이 있는 항목만 필터링
    valid = [d for d in data if d.get('text') or d.get('images') or d['type'] == 'markdown']
    return valid if valid else None

@st.cache_data
def load_notebook_raw():
    """노트북 파일에서 직접 Phase 12~13 셀을 읽어옵니다."""
    nb_path = 'battery_capacity_prediction.ipynb'
    if not os.path.exists(nb_path):
        return None
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    results = []
    in_phase12_13 = False
    
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        
        # Phase 12 시작 감지
        if 'Phase 12' in src and ('클러스터링 ARI' in src or '다변량 예측' in src):
            in_phase12_13 = True
        
        if not in_phase12_13:
            continue
        
        # 마크다운 셀
        if cell['cell_type'] == 'markdown':
            results.append({'type': 'markdown', 'content': src, 'cell_index': i})
            continue
        
        if cell['cell_type'] != 'code':
            continue
        
        # 코드 셀 출력 추출
        outputs = cell.get('outputs', [])
        texts = []
        images = []
        
        for out in outputs:
            if out.get('output_type') == 'stream':
                texts.append(''.join(out.get('text', [])))
            elif out.get('output_type') == 'execute_result':
                d = out.get('data', {})
                if 'text/plain' in d:
                    t = d['text/plain']
                    texts.append(''.join(t) if isinstance(t, list) else t)
            
            if out.get('output_type') in ('display_data', 'execute_result'):
                d = out.get('data', {})
                if 'image/png' in d:
                    img = d['image/png']
                    if isinstance(img, list):
                        img = ''.join(img)
                    images.append(img)  # base64 문자열 직접 전달
        
        results.append({
            'type': 'code_output',
            'cell_index': i,
            'source_preview': src[:100] + '...' if len(src) > 100 else src,
            'text': '\n'.join(texts) if texts else '',
            'images_b64': images,
            'has_output': bool(texts or images),
        })
    
    return results

@st.cache_data
def run_live_analysis():
    """Fallback: 노트북 출력이 없을 때 실시간 분석을 수행합니다."""
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.decomposition import PCA

    raw = pd.read_csv('data.csv')
    cols = ['cell_id','initial_label','initial_voltage','initial_impedance',
            'v42_label','v42_voltage','v42_impedance','v25_label','v25_voltage','v25_impedance',
            'v36_label','v36_voltage','v36_empty','v36_impedance','capacity']
    raw.columns = cols
    df = raw.drop(columns=['initial_label','v42_label','v25_label','v36_label','v36_empty'])
    for c in [c for c in df.columns if c != 'cell_id']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    m = (df['v36_impedance'] - df['initial_impedance']).abs() > 100
    df_clean = df[~m].copy()

    ORIG = ['initial_voltage','initial_impedance','v42_voltage','v42_impedance',
            'v25_voltage','v25_impedance','v36_voltage','v36_impedance']
    X_all = df_clean[ORIG + ['capacity']].dropna()
    N = 4

    sc = StandardScaler()
    Xg = sc.fit_transform(X_all[ORIG + ['capacity']])
    lg = KMeans(n_clusters=N, random_state=42, n_init='auto').fit_predict(Xg)

    X1d = StandardScaler().fit_transform(X_all[['capacity']])
    l1d = KMeans(n_clusters=N, random_state=42, n_init='auto').fit_predict(X1d)

    inp = ['initial_voltage','initial_impedance','v25_voltage','v25_impedance']
    out = ['v42_voltage','v42_impedance','v36_voltage','v36_impedance','capacity']
    Xtr, Xte, Ytr, Yte = train_test_split(X_all[inp], X_all[out], test_size=0.2, random_state=42)

    try:
        from catboost import CatBoostRegressor
        base = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_seed=42)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        base = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)

    mo = MultiOutputRegressor(base)
    mo.fit(Xtr, Ytr)
    Yp = mo.predict(X_all[inp])
    Ypdf = pd.DataFrame(Yp, columns=out, index=X_all.index)

    ps = {}
    for c in out:
        ps[c] = {'R2': r2_score(X_all[c], Ypdf[c]),
                 'RMSE': np.sqrt(mean_squared_error(X_all[c], Ypdf[c]))}

    Xpf = pd.concat([X_all[inp].reset_index(drop=True), Ypdf.reset_index(drop=True)], axis=1)
    Xps = StandardScaler().fit_transform(Xpf)
    lp = KMeans(n_clusters=N, random_state=42, n_init='auto').fit_predict(Xps)

    X2d = PCA(n_components=2, random_state=42).fit_transform(Xg)

    return {
        'ari_old': adjusted_rand_score(lg, l1d),
        'ari_new': adjusted_rand_score(lg, lp),
        'nmi_new': normalized_mutual_info_score(lg, lp),
        'sil_gt': silhouette_score(Xg, lg),
        'sil_pred': silhouette_score(Xps, lp),
        'pred_scores': ps,
        'X2d': X2d, 'lg': lg, 'lp': lp, 'l1d': l1d, 'N': N,
        'n_samples': len(X_all),
    }

# ============================================================
# 2. 사이드바
# ============================================================
with st.sidebar:
    st.title("🔋 배터리 리포트")
    st.markdown("---")
    st.caption("Phase 12~13 결과 뷰어")

# ============================================================
# 3. 데이터 로드
# ============================================================
nb_data = load_notebook_raw()

# 노트북 출력이 있는지 확인
has_nb_outputs = False
if nb_data:
    has_nb_outputs = any(d.get('has_output', False) for d in nb_data if d['type'] == 'code_output')

st.title("🔋 배터리 품질 예측 기반 스마트 패키징 리포트")

if has_nb_outputs:
    # ============================================================
    # MODE A: 노트북 출력 표시
    # ============================================================
    st.info("📓 노트북 셀 출력을 표시합니다.")
    st.markdown("---")
    
    for item in nb_data:
        if item['type'] == 'markdown':
            st.markdown(item['content'])
        elif item['type'] == 'code_output' and item.get('has_output'):
            # 텍스트 출력
            if item.get('text'):
                st.code(item['text'], language='text')
            # 이미지 출력
            for img_b64 in item.get('images_b64', []):
                img_bytes = base64.b64decode(img_b64)
                st.image(img_bytes, use_container_width=True)

else:
    # ============================================================
    # MODE B: 실시간 분석 Fallback
    # ============================================================
    st.warning("⚠️ 노트북 셀이 아직 실행되지 않았습니다. 실시간 분석 결과를 표시합니다.\n\n"
               "노트북에서 Phase 12~13을 실행한 후 `python extract_outputs.py`를 실행하면 실제 출력을 볼 수 있습니다.")
    st.markdown("---")
    
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    
    with st.spinner("🔄 분석 중... (최초 1회만 소요)"):
        R = run_live_analysis()
    
    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 ARI (개선 후)", f"{R['ari_new']:.4f}", delta=f"+{R['ari_new']-R['ari_old']:.4f}")
    c2.metric("📊 NMI", f"{R['nmi_new']:.4f}")
    c3.metric("⏱️ 공정 단축", "4→2단계", delta="-50%")
    c4.metric("📦 분석 대상", f"{R['n_samples']:,}개")
    
    st.markdown("---")
    
    # ARI 비교
    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['기존 (1D)'], y=[max(R['ari_old'],0)],
                             marker_color='#e74c3c', text=[f"{R['ari_old']:.4f}"], textposition='outside'))
        fig.add_trace(go.Bar(x=['개선 (9D)'], y=[R['ari_new']],
                             marker_color='#2ecc71', text=[f"{R['ari_new']:.4f}"], textposition='outside'))
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange", annotation_text="Good")
        fig.update_layout(title="ARI 비교", yaxis_range=[0,1.15], showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_r:
        X2d = R['X2d']
        df_v = pd.DataFrame({'PC1':X2d[:,0],'PC2':X2d[:,1],
                             'GT':[f'군집{c}' for c in R['lg']],
                             'Pred':[f'군집{c}' for c in R['lp']]})
        tab1, tab2 = st.tabs(["GT 클러스터", "예측 클러스터"])
        with tab1:
            st.plotly_chart(px.scatter(df_v, x='PC1', y='PC2', color='GT',
                           title="Ground Truth").update_traces(marker_size=4), use_container_width=True)
        with tab2:
            st.plotly_chart(px.scatter(df_v, x='PC1', y='PC2', color='Pred',
                           title=f"예측 (ARI={R['ari_new']:.4f})").update_traces(marker_size=4), use_container_width=True)
    
    # 예측 정확도
    st.markdown("### 피처별 예측 정확도")
    ps = pd.DataFrame(R['pred_scores']).T.reset_index()
    ps.columns = ['피처', 'R²', 'RMSE']
    st.dataframe(ps.style.background_gradient(subset=['R²'], cmap='RdYlGn'), hide_index=True)
    
    # 비용 절감
    st.markdown("### 💰 비용·시간 절감 효과")
    st.markdown(f"""
    | 항목 | 기존 | 개선 후 |
    |------|------|---------|
    | 측정 공정 | 4단계 | **2단계** |
    | 소요 시간 | ~100분/개 | **~50분/개** |
    | ARI Score | {R['ari_old']:.4f} | **{R['ari_new']:.4f}** |
    | 절감율 | - | **~50%** |
    """)
    
    # 결론
    if R['ari_new'] > 0.6:
        st.success(f"✅ **검증 성공**: 클러스터링 동등성 입증 완료 (ARI = {R['ari_new']:.4f})")
    else:
        st.warning(f"⚠️ ARI = {R['ari_new']:.4f} — 추가 개선 여지 있음")

st.markdown("---")
st.caption("🏁 Battery Smart Packaging Report — Powered by AI & Streamlit")
