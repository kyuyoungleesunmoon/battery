"""Phase 9 측정 조합 최적화 리포트 — CSV 기반 즉시 로딩."""
import warnings
warnings.filterwarnings('ignore')

import os
import platform
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import seaborn as sns

# ── 페이지 설정 ──
st.set_page_config(page_title='Phase 9 리포트', page_icon='🔋', layout='wide')

# ── 한글 폰트 설정 (Windows: Malgun Gothic, Linux/Cloud: NanumGothic) ──
def _setup_korean_font():
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        # Streamlit Cloud (Linux) — fonts-nanum 패키지 사용
        font_dirs = ['/usr/share/fonts/truetype/nanum']
        font_files = fm.findSystemFonts(fontpaths=font_dirs)
        for f in font_files:
            fm.fontManager.addfont(f)
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False

_setup_korean_font()

# ── 상수 ──
COMBOS = {
    'P1_S1+v42': ['initial_voltage', 'initial_impedance', 'v42_voltage', 'v42_impedance'],
    'P1_S1+v25': ['initial_voltage', 'initial_impedance', 'v25_voltage', 'v25_impedance'],
    'P1_S1+v36': ['initial_voltage', 'initial_impedance', 'v36_voltage', 'v36_impedance'],
    'P2_v42_only': ['v42_voltage', 'v42_impedance'],
    'P2_v25_only': ['v25_voltage', 'v25_impedance'],
    'P2_v36_only': ['v36_voltage', 'v36_impedance'],
    'P3_all_voltage': ['initial_voltage', 'v42_voltage', 'v25_voltage', 'v36_voltage'],
    'P3_all_impedance': ['initial_impedance', 'v42_impedance', 'v25_impedance', 'v36_impedance'],
    'P3_S1v25_volt': ['initial_voltage', 'v25_voltage'],
    'P3_S1v25_imp': ['initial_impedance', 'v25_impedance'],
    'P3_v25_volt_1': ['v25_voltage'],
    'REF_S1_only': ['initial_voltage', 'initial_impedance'],
    'REF_S4_ALL': [
        'initial_voltage', 'initial_impedance',
        'v42_voltage', 'v42_impedance',
        'v25_voltage', 'v25_impedance',
        'v36_voltage', 'v36_impedance',
    ],
}

STEP_MAP = {
    'P1_S1+v42': '2단계(초기+풀충전)',
    'P1_S1+v25': '2단계(초기+완전방전)',
    'P1_S1+v36': '2단계(초기+방전중간)',
    'P2_v42_only': '1단계(풀충전만)',
    'P2_v25_only': '1단계(완전방전만)',
    'P2_v36_only': '1단계(방전중간만)',
    'P3_all_voltage': '4단계(전압만)',
    'P3_all_impedance': '4단계(임피던스만)',
    'P3_S1v25_volt': '2단계(전압만)',
    'P3_S1v25_imp': '2단계(임피던스만)',
    'P3_v25_volt_1': '1단계(v25전압1개)',
    'REF_S1_only': '1단계(초기만)',
    'REF_S4_ALL': '4단계(전체)',
}

COMBO_DESC = {
    'P1_S1+v42': '초기 OCV/임피던스 + 4.2V 풀충전 후 전압/임피던스',
    'P1_S1+v25': '초기 OCV/임피던스 + 2.5V 완전방전 후 전압/임피던스',
    'P1_S1+v36': '초기 OCV/임피던스 + 3.6V 방전중간 전압/임피던스',
    'P2_v42_only': '4.2V 풀충전 상태 전압/임피던스만',
    'P2_v25_only': '2.5V 완전방전 상태 전압/임피던스만',
    'P2_v36_only': '3.6V 방전중간 상태 전압/임피던스만',
    'P3_all_voltage': '4개 공정 전압만 (임피던스 제외)',
    'P3_all_impedance': '4개 공정 임피던스만 (전압 제외)',
    'P3_S1v25_volt': '초기+완전방전 전압만',
    'P3_S1v25_imp': '초기+완전방전 임피던스만',
    'P3_v25_volt_1': '완전방전 전압 단독 (1개 피처)',
    'REF_S1_only': '초기 OCV/임피던스만 (기준선)',
    'REF_S4_ALL': '전체 4공정 8피처 (최대 정보)',
}

# ── 데이터 로드 ──
@st.cache_data
def load_results():
    p9 = pd.read_csv('phase9_results.csv')
    corr = pd.read_csv('phase9_corr.csv')
    return p9, corr

p9_df, corr_df = load_results()

# 조합별 최고 성능
combo_best = p9_df.sort_values('RMSE').groupby('Combo', as_index=False).first()
r2_s4 = float(combo_best.loc[combo_best['Combo'] == 'REF_S4_ALL', 'R2'].iloc[0])
combo_best['S4_R2_%'] = (combo_best['R2'] / r2_s4 * 100).round(1)
combo_best = combo_best.sort_values('RMSE')

# FLAML 강화 결과
enh_df = p9_df[p9_df['Source'] == 'flaml_enhanced'].copy()
base_df = p9_df[p9_df['Source'] == 'base'].copy()

# ══════════════════════════════════════════════════════════════
# 헤더
# ══════════════════════════════════════════════════════════════
st.title('🔋 Phase 9: 최적 측정 조합 탐색 리포트')
st.caption('파생변수 제외 · 원본 피처 · 13개 조합 × 7개 모델 · 측정 공정 효율성 분석')

st.divider()

# ── 상단 KPI ──
k1, k2, k3, k4 = st.columns(4)
k1.metric('정제 후 샘플', '1,039')
k2.metric('실험 조합', f'{len(COMBOS)}개')
k3.metric('총 실험 수', f'{len(p9_df)}개')
best_overall = p9_df.loc[p9_df['RMSE'].idxmin()]
k4.metric('최고 R²', f"{best_overall['R2']:.4f}")

# ══════════════════════════════════════════════════════════════
# 9-1. 실험 설계
# ══════════════════════════════════════════════════════════════
st.header('9-1. 실험 설계: 측정 조합 정의')

st.markdown("""
**배경**: 중고 배터리 패키징 시 측정 공정을 최소화하면서 최대 예측 성능을 달성하기 위해,  
파생변수 없이 원본 피처만으로 다양한 측정 조합을 실험합니다.

**측정 단계**: `Initial(초기 OCV)` → `v42(4.2V 풀충전)` → `v36(3.6V 방전중간)` → `v25(2.5V 완전방전)`
""")

col_l, col_r = st.columns([3, 2])

with col_l:
    st.subheader('조합 목록')
    combo_info = pd.DataFrame([
        {
            '조합': k,
            '측정 공정': STEP_MAP[k],
            '피처 수': len(v),
            '설명': COMBO_DESC[k],
            '피처': ', '.join(v),
        }
        for k, v in COMBOS.items()
    ])
    st.dataframe(combo_info, width='stretch', hide_index=True, height=500)

with col_r:
    st.subheader('피처-Capacity 상관계수')
    corr_sorted = corr_df.sort_values('Correlation', key=np.abs, ascending=False)
    fig_c, ax_c = plt.subplots(figsize=(6, 4))
    colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in corr_sorted['Correlation']]
    ax_c.barh(corr_sorted['Feature'], corr_sorted['Correlation'], color=colors)
    ax_c.set_xlabel('Pearson r')
    ax_c.set_title('원본 피처 vs Capacity')
    ax_c.grid(axis='x', alpha=0.3)
    ax_c.invert_yaxis()
    st.pyplot(fig_c)

st.divider()

# ══════════════════════════════════════════════════════════════
# 9-2/9-3. 모델 성능 비교
# ══════════════════════════════════════════════════════════════
st.header('9-2/9-3. 조합별 모델 성능 비교')

tab_table, tab_heatmap, tab_bar = st.tabs(['📋 전체 결과 테이블', '🗺️ RMSE 히트맵', '📊 R² 막대 차트'])

with tab_table:
    view_cols = ['Phase', 'Combo', 'Step', 'Model', 'N_Features', 'RMSE', 'R2', 'MAPE', 'MAE']
    st.dataframe(
        p9_df[view_cols].sort_values(['Combo', 'RMSE']),
        width='stretch',
        hide_index=True,
        height=600,
    )

with tab_heatmap:
    pivot = base_df.pivot_table(index='Combo', columns='Model', values='RMSE', aggfunc='min')
    # 정렬: RMSE 평균이 낮은 조합 위에
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    fig_h, ax_h = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd_r', linewidths=0.5, ax=ax_h)
    ax_h.set_title('조합 × 모델 RMSE 히트맵 (낮을수록 우수)', fontsize=14, fontweight='bold')
    ax_h.set_ylabel('')
    st.pyplot(fig_h)

with tab_bar:
    # 모델별 평균 R2 비교 (기본 모델만)
    model_avg = base_df.groupby('Model')['R2'].mean().sort_values()
    fig_m, ax_m = plt.subplots(figsize=(10, 5))
    ax_m.barh(model_avg.index, model_avg.values, color=sns.color_palette('viridis', len(model_avg)))
    for i, (model, val) in enumerate(model_avg.items()):
        ax_m.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=10)
    ax_m.set_xlabel('평균 R² (전 조합)')
    ax_m.set_title('모델별 평균 R² Score', fontsize=14, fontweight='bold')
    ax_m.grid(axis='x', alpha=0.3)
    st.pyplot(fig_m)

st.divider()

# ══════════════════════════════════════════════════════════════
# 9-4. 종합 분석 & 효율성
# ══════════════════════════════════════════════════════════════
st.header('9-4. 종합 결과 및 측정 공정 효율성')

m1, m2, m3 = st.columns(3)
m1.metric('🏆 최고 조합', str(best_overall['Combo']))
m2.metric('🏆 최고 모델', str(best_overall['Model']))
m3.metric('🏆 RMSE / R²', f"{best_overall['RMSE']:.4f} / {best_overall['R2']:.4f}")

st.subheader('조합별 최고 성능 (RMSE 기준)')
display_cols = ['Combo', 'Step', 'N_Features', 'Model', 'RMSE', 'R2', 'MAPE', 'S4_R2_%']
st.dataframe(combo_best[display_cols], width='stretch', hide_index=True)

# Top 10
st.subheader('전체 Top 10')
top10 = p9_df.nsmallest(10, 'RMSE').reset_index(drop=True)
top10.index = top10.index + 1
top10.index.name = '순위'
st.dataframe(
    top10[['Combo', 'Step', 'Model', 'N_Features', 'RMSE', 'R2', 'MAPE', 'MAE']],
    width='stretch',
)

# 시각화 2종
ch_l, ch_r = st.columns(2)

with ch_l:
    st.subheader('조합별 최고 R² 비교')
    sorted_cb = combo_best.sort_values('R2', ascending=True)
    colors_cb = []
    for _, row in sorted_cb.iterrows():
        if row['Combo'].startswith('REF'):
            colors_cb.append('#2196F3')
        elif row['Combo'].startswith('P1'):
            colors_cb.append('#4CAF50')
        elif row['Combo'].startswith('P2'):
            colors_cb.append('#FF9800')
        else:
            colors_cb.append('#9C27B0')

    fig_cb, ax_cb = plt.subplots(figsize=(8, 7))
    bars = ax_cb.barh(sorted_cb['Combo'], sorted_cb['R2'], color=colors_cb, edgecolor='white')
    for bar, (_, row) in zip(bars, sorted_cb.iterrows()):
        ax_cb.text(
            bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{row['R2']:.4f}  {row['Model']}", va='center', fontsize=8,
        )
    ax_cb.axvline(x=r2_s4, color='red', linestyle='--', alpha=0.6, label=f'S4 기준 R²={r2_s4:.4f}')
    ax_cb.set_xlabel('R² Score')
    ax_cb.set_title('조합별 최고 R²')
    ax_cb.legend(fontsize=9)
    ax_cb.grid(axis='x', alpha=0.3)
    st.pyplot(fig_cb)

with ch_r:
    st.subheader('피처 수 vs R² (효율성)')
    fig_ef, ax_ef = plt.subplots(figsize=(8, 7))
    for _, row in combo_best.iterrows():
        if row['Combo'].startswith('REF'):
            mk, cl = 's', '#2196F3'
        elif row['Combo'].startswith('P1'):
            mk, cl = 'o', '#4CAF50'
        elif row['Combo'].startswith('P2'):
            mk, cl = '^', '#FF9800'
        else:
            mk, cl = 'D', '#9C27B0'
        ax_ef.scatter(row['N_Features'], row['R2'], s=130, marker=mk, c=cl, edgecolors='black', zorder=5)
        short = row['Combo'].split('_', 1)[-1] if '_' in row['Combo'] else row['Combo']
        ax_ef.annotate(short, (row['N_Features'], row['R2']),
                       textcoords='offset points', xytext=(6, 5), fontsize=7)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2196F3', markersize=9, label='REF (기준선)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50', markersize=9, label='P1 (S1+단일)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#FF9800', markersize=9, label='P2 (단일만)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#9C27B0', markersize=9, label='P3 (전압/임피던스)'),
    ]
    ax_ef.legend(handles=legend_elements, fontsize=8, loc='lower right')
    ax_ef.set_xlabel('피처 수')
    ax_ef.set_ylabel('R² Score')
    ax_ef.set_title('효율성 프론티어')
    ax_ef.grid(True, alpha=0.3)
    ax_ef.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    st.pyplot(fig_ef)

st.divider()

# ══════════════════════════════════════════════════════════════
# 9-5. FLAML 강화 결과
# ══════════════════════════════════════════════════════════════
st.header('9-5. FLAML 강화 학습 결과')

if len(enh_df) > 0:
    enh_display = enh_df[['Combo', 'Step', 'Model', 'N_Features', 'RMSE', 'R2', 'MAPE', 'MAE']].sort_values('RMSE')
    st.dataframe(enh_display, width='stretch', hide_index=True)

    # FLAML vs 기본모델 최고 비교
    compare_rows = []
    for _, erow in enh_df.iterrows():
        combo = erow['Combo']
        base_best = base_df[base_df['Combo'] == combo].sort_values('RMSE').iloc[0]
        compare_rows.append({
            '조합': combo,
            '기본모델 최고': base_best['Model'],
            '기본 RMSE': round(base_best['RMSE'], 4),
            '기본 R²': round(base_best['R2'], 4),
            'FLAML 모델': erow['Model'],
            'FLAML RMSE': round(erow['RMSE'], 4),
            'FLAML R²': round(erow['R2'], 4),
            'RMSE 개선': round(base_best['RMSE'] - erow['RMSE'], 4),
        })
    compare_df = pd.DataFrame(compare_rows).sort_values('FLAML RMSE')
    st.subheader('기본 모델 vs FLAML 비교')
    st.dataframe(compare_df, width='stretch', hide_index=True)
else:
    st.info('FLAML 강화 결과가 없습니다.')

st.divider()

# ══════════════════════════════════════════════════════════════
# 9-6. 최종 결론
# ══════════════════════════════════════════════════════════════
st.header('9-6. 최종 결론')

s4 = combo_best[combo_best['Combo'] == 'REF_S4_ALL'].iloc[0]
s1v25 = combo_best[combo_best['Combo'] == 'P1_S1+v25'].iloc[0]
v25 = combo_best[combo_best['Combo'] == 'P2_v25_only'].iloc[0]

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('#### 🥇 최고 성능')
    st.metric('조합', s4['Combo'])
    st.metric('모델', s4['Model'])
    st.metric('RMSE', f"{s4['RMSE']:.4f}")
    st.metric('R²', f"{s4['R2']:.4f}")
with c2:
    st.markdown('#### 🥈 최적 효율 (2단계)')
    st.metric('조합', s1v25['Combo'])
    st.metric('모델', s1v25['Model'])
    st.metric('RMSE', f"{s1v25['RMSE']:.4f}")
    st.metric('R²', f"{s1v25['R2']:.4f}")
with c3:
    st.markdown('#### 🥉 최소 측정 (1단계)')
    st.metric('조합', v25['Combo'])
    st.metric('모델', v25['Model'])
    st.metric('RMSE', f"{v25['RMSE']:.4f}")
    st.metric('R²', f"{v25['R2']:.4f}")

st.markdown('---')

st.markdown("""
### 📌 데이터 조합 설명

| 구분 | 조합 | 측정 공정 | 설명 |
|------|------|----------|------|
| **Phase 1** | S1+v42 | 초기 → 풀충전 | 초기 OCV/임피던스 측정 후 4.2V까지 충전하여 재측정 |
| | S1+v25 | 초기 → 완전방전 | 초기 측정 후 2.5V까지 방전하여 재측정 |
| | S1+v36 | 초기 → 방전중간 | 초기 측정 후 3.6V(공칭전압)까지 방전 재측정 |
| **Phase 2** | v42 only | 풀충전만 | 4.2V 풀충전 상태에서 전압/임피던스만 측정 |
| | v25 only | 완전방전만 | 2.5V 완전방전 상태에서 전압/임피던스만 측정 |
| | v36 only | 방전중간만 | 3.6V 방전중간 상태에서 전압/임피던스만 측정 |
| **Phase 3** | all_voltage | 전압만 | 4개 공정의 전압값만 사용 (임피던스 미사용) |
| | all_impedance | 임피던스만 | 4개 공정의 임피던스만 사용 (전압 미사용) |
| | S1v25_volt | 전압 2개 | 초기+완전방전 전압만 |
| | S1v25_imp | 임피던스 2개 | 초기+완전방전 임피던스만 |
| | v25_volt_1 | 전압 1개 | 완전방전 전압 단독 |
| **REF** | S1_only | 초기만 | 초기 OCV/임피던스만 (최소 기준선) |
| | S4_ALL | 전체 4공정 | 전체 측정 8피처 (최대 기준선) |

### 💡 핵심 인사이트

1. **v25(완전방전) 측정이 Capacity 예측의 핵심** — v25 관련 조합이 상위권 독점
2. **S4(전체) 대비 S1+v25는 피처 절반(4개)으로 유사 성능** — 측정 공정 50% 절감 가능
3. **v25 단독(2피처)도 실용적 수준** — 1단계 공정으로 빠른 스크리닝 가능
4. **CatBoost/XGBoost 계열이 전 조합에서 우수** — 트리 기반 모델이 배터리 데이터에 적합
5. **전압 > 임피던스** — 같은 공정이라도 전압이 임피던스보다 예측력이 높음
""")

# 다운로드
csv_bytes = p9_df.to_csv(index=False).encode('utf-8-sig')
st.download_button('📥 전체 결과 CSV 다운로드', data=csv_bytes, file_name='phase9_results.csv', mime='text/csv')
