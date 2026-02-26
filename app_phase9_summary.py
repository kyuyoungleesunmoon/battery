import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

try:
    from flaml import AutoML
    HAS_FLAML = True
except Exception:
    AutoML = None
    HAS_FLAML = False

st.set_page_config(page_title='Phase 9 리포트', page_icon='🔋', layout='wide')

ORIG_FEATURES = [
    'initial_voltage', 'initial_impedance',
    'v42_voltage', 'v42_impedance',
    'v25_voltage', 'v25_impedance',
    'v36_voltage', 'v36_impedance'
]
TARGET = 'capacity'

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
    'REF_S4_ALL': ORIG_FEATURES,
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


@st.cache_data
def load_and_clean_data(path: str = './data.csv'):
    raw = pd.read_csv(path)
    raw.columns = [
        'cell_id', 'initial_label', 'initial_voltage', 'initial_impedance',
        'v42_label', 'v42_voltage', 'v42_impedance',
        'v25_label', 'v25_voltage', 'v25_impedance',
        'v36_label', 'v36_voltage', 'v36_empty', 'v36_impedance',
        'capacity'
    ]
    drop_cols = ['initial_label', 'v42_label', 'v25_label', 'v36_label', 'v36_empty']
    df = raw.drop(columns=drop_cols)

    numeric_cols = [c for c in df.columns if c != 'cell_id']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    extreme_mask = (df['v36_impedance'] - df['initial_impedance']).abs() > 100
    df_clean = df[~extreme_mask].copy()
    return df_clean, int(extreme_mask.sum())


@st.cache_data
def run_phase9(df_clean: pd.DataFrame):
    X_orig = df_clean[ORIG_FEATURES]
    y_orig = df_clean[TARGET]
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42
    )

    models = {
        'LinearReg': LinearRegression(),
        'SVR': SVR(C=1.0, epsilon=0.01),
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_seed=42),
    }

    scale_needed = {'LinearReg', 'SVR'}
    results = []

    for combo_name, feats in COMBOS.items():
        X_tr = X_train_o[feats]
        X_te = X_test_o[feats]
        scaler = RobustScaler()
        X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=feats, index=X_tr.index)
        X_te_sc = pd.DataFrame(scaler.transform(X_te), columns=feats, index=X_te.index)

        for model_name, model in models.items():
            start = time.time()
            if model_name in scale_needed:
                model.fit(X_tr_sc, y_train_o)
                y_pred = model.predict(X_te_sc)
            else:
                model.fit(X_tr, y_train_o)
                y_pred = model.predict(X_te)

            results.append({
                'Phase': combo_name.split('_')[0],
                'Combo': combo_name,
                'Model': model_name,
                'N_Features': len(feats),
                'RMSE': float(np.sqrt(mean_squared_error(y_test_o, y_pred))),
                'R2': float(r2_score(y_test_o, y_pred)),
                'MAPE': float(mean_absolute_percentage_error(y_test_o, y_pred)),
                'MAE': float(mean_absolute_error(y_test_o, y_pred)),
                'TrainSec': round(time.time() - start, 2),
            })

    p9_df = pd.DataFrame(results)

    combo_best = p9_df.sort_values('RMSE').groupby('Combo', as_index=False).first()
    r2_s4 = float(combo_best.loc[combo_best['Combo'] == 'REF_S4_ALL', 'R2'].iloc[0])

    combo_best = combo_best.copy()
    combo_best['Step'] = combo_best['Combo'].map(STEP_MAP)
    combo_best['S4_R2_%'] = (combo_best['R2'] / r2_s4 * 100).round(1)

    top10 = p9_df.nsmallest(10, 'RMSE').reset_index(drop=True)

    phase_best_rows = []
    for phase in ['P1', 'P2', 'P3', 'REF']:
        subset = p9_df[p9_df['Phase'] == phase]
        if len(subset) > 0:
            phase_best_rows.append(subset.loc[subset['RMSE'].idxmin()])
    phase_best_df = pd.DataFrame(phase_best_rows)

    # 9-5 대응: 상위 조합 + 핵심 조합에 대해 FLAML 강화
    enh_df = pd.DataFrame()
    if HAS_FLAML and AutoML is not None:
        top5_combos = combo_best.sort_values('RMSE').head(5)['Combo'].tolist()
        for must in ['P1_S1+v25', 'P2_v25_only', 'REF_S4_ALL']:
            if must not in top5_combos:
                top5_combos.append(must)

        enh_rows = []
        for combo_name in top5_combos:
            feats = COMBOS[combo_name]
            X_tr = X_train_o[feats]
            X_te = X_test_o[feats]

            automl = AutoML()
            automl.fit(
                X_train=X_tr,
                y_train=y_train_o,
                time_budget=20,
                metric='r2',
                task='regression',
                eval_method='cv',
                n_splits=5,
                verbose=0,
                log_file_name=f'flaml_phase9_report_{combo_name}.log',
            )
            pred = automl.predict(X_te)
            pred_arr = np.asarray(pred, dtype=float)
            enh_rows.append({
                'Combo': combo_name,
                'Model': f'FLAML_20s({automl.best_estimator})',
                'N_Features': len(feats),
                'RMSE': float(np.sqrt(mean_squared_error(y_test_o, pred_arr))),
                'R2': float(r2_score(y_test_o, pred_arr)),
                'MAPE': float(mean_absolute_percentage_error(y_test_o, pred_arr)),
                'MAE': float(mean_absolute_error(y_test_o, pred_arr)),
            })

        enh_df = pd.DataFrame(enh_rows).sort_values('RMSE')

    return p9_df, combo_best.sort_values('RMSE'), top10, phase_best_df, enh_df


def draw_combo_best_chart(combo_best_df: pd.DataFrame):
    sorted_df = combo_best_df.sort_values('R2', ascending=True)
    colors = []
    for _, row in sorted_df.iterrows():
        if row['Combo'].startswith('REF'):
            colors.append('#2196F3')
        elif row['Combo'].startswith('P1'):
            colors.append('#4CAF50')
        elif row['Combo'].startswith('P2'):
            colors.append('#FF9800')
        else:
            colors.append('#9C27B0')

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(sorted_df['Combo'], sorted_df['R2'], color=colors, edgecolor='white')
    for bar, (_, row) in zip(bars, sorted_df.iterrows()):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"R2={row['R2']:.4f} | {row['Model']}",
            va='center',
            fontsize=9,
        )
    ax.set_xlabel('R2 Score')
    ax.set_ylabel('Combo')
    ax.set_title('조합별 최고 성능(R2) 비교')
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig)


def draw_efficiency_frontier(combo_best_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in combo_best_df.iterrows():
        if row['Combo'].startswith('REF'):
            marker, color = 's', '#2196F3'
        elif row['Combo'].startswith('P1'):
            marker, color = 'o', '#4CAF50'
        elif row['Combo'].startswith('P2'):
            marker, color = '^', '#FF9800'
        else:
            marker, color = 'D', '#9C27B0'

        ax.scatter(row['N_Features'], row['R2'], s=140, marker=marker, c=color, edgecolors='black')
        ax.annotate(
            row['Combo'].replace('REF_', '').replace('P1_', '').replace('P2_', '').replace('P3_', ''),
            (row['N_Features'], row['R2']),
            textcoords='offset points',
            xytext=(6, 5),
            fontsize=8,
        )

    ax.set_xlabel('피처 수')
    ax.set_ylabel('R2 Score')
    ax.set_title('피처 수 vs 예측 성능 (효율성 프론티어)')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def draw_model_heatmap(p9_df: pd.DataFrame):
    pivot = p9_df.pivot_table(index='Combo', columns='Model', values='RMSE', aggfunc='min')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd_r', ax=ax)
    ax.set_title('조합-모델 RMSE 히트맵 (낮을수록 우수)')
    st.pyplot(fig)


st.title('🔋 Phase 9 측정 조합 최적화 리포트')
st.caption('파생변수 제외 조건에서 조합별 모델 성능과 측정 공정 효율을 시각화합니다.')

df_clean, n_outlier = load_and_clean_data('./data.csv')

with st.spinner('Phase 9 리포트 데이터를 생성 중입니다...'):
    p9_df, combo_best_df, top10_df, phase_best_df, enh_df = run_phase9(df_clean)

col1, col2, col3, col4 = st.columns(4)
col1.metric('정제 후 샘플 수', f'{len(df_clean):,}')
col2.metric('제거 이상치', str(n_outlier))
col3.metric('실험 조합 수', str(len(COMBOS)))
col4.metric('총 실험 수', f"{len(p9_df):,}")

st.markdown('## 9-1. 실험 조합 정의 및 상관 분석')

combo_desc_df = pd.DataFrame([
    {
        'Combo': k,
        'Phase': k.split('_')[0],
        'N_Features': len(v),
        '측정 공정': STEP_MAP[k],
        'Features': ', '.join(v)
    }
    for k, v in COMBOS.items()
]).sort_values(['Phase', 'Combo'])

st.dataframe(combo_desc_df, width='stretch', hide_index=True)

corr_df = pd.DataFrame({
    'Feature': ORIG_FEATURES,
    'Correlation': [df_clean[f].corr(df_clean[TARGET]) for f in ORIG_FEATURES]
}).sort_values('Correlation', key=np.abs, ascending=False)

fig_corr, ax_corr = plt.subplots(figsize=(10, 4))
sns.barplot(data=corr_df, x='Feature', y='Correlation', ax=ax_corr, palette='viridis')
ax_corr.set_title('원본 피처 vs Capacity 상관계수')
ax_corr.tick_params(axis='x', rotation=25)
ax_corr.grid(axis='y', alpha=0.3)
st.pyplot(fig_corr)

st.markdown('## 9-2/9-3. 조합별 모델 성능 비교')

st.dataframe(
    p9_df[['Phase', 'Combo', 'Model', 'N_Features', 'RMSE', 'R2', 'MAPE', 'MAE', 'TrainSec']]
    .sort_values(['Combo', 'RMSE']),
    width='stretch',
    hide_index=True,
)

draw_model_heatmap(p9_df)

st.markdown('## 9-4. 종합 결과 및 측정 공정 효율성')

best_overall = p9_df.loc[p9_df['RMSE'].idxmin()]
left, mid, right = st.columns(3)
left.metric('전체 최고 조합', str(best_overall['Combo']))
mid.metric('최고 모델', str(best_overall['Model']))
right.metric('최고 RMSE', f"{best_overall['RMSE']:.4f}")

st.markdown('### 조합별 최고 성능')
st.dataframe(
    combo_best_df[['Combo', 'Step', 'N_Features', 'Model', 'RMSE', 'R2', 'MAPE', 'S4_R2_%']],
    width='stretch',
    hide_index=True,
)

st.markdown('### 전체 Top 10 (RMSE 기준)')
st.dataframe(
    top10_df[['Combo', 'Model', 'N_Features', 'RMSE', 'R2', 'MAPE', 'MAE']],
    width='stretch',
    hide_index=True,
)

st.markdown('### 시각화: 조합별 R2 및 효율성 프론티어')
draw_combo_best_chart(combo_best_df)
draw_efficiency_frontier(combo_best_df)

st.markdown('## 9-5. FLAML 강화 학습 결과')
if len(enh_df) > 0:
    st.dataframe(enh_df, width='stretch', hide_index=True)
else:
    st.info('FLAML 미설치 환경입니다. 기본/부스팅 모델 결과로 리포트를 구성했습니다.')

st.markdown('## 9-6. 최종 결론')

s4_row = combo_best_df[combo_best_df['Combo'] == 'REF_S4_ALL'].iloc[0]
s1v25_row = combo_best_df[combo_best_df['Combo'] == 'P1_S1+v25'].iloc[0]
v25_row = combo_best_df[combo_best_df['Combo'] == 'P2_v25_only'].iloc[0]

st.markdown(
    f"""
- **최고 성능**: {s4_row['Combo']} / {s4_row['Model']} (RMSE={s4_row['RMSE']:.4f}, R2={s4_row['R2']:.4f})
- **최적 효율 (2단계 공정)**: {s1v25_row['Combo']} / {s1v25_row['Model']} (RMSE={s1v25_row['RMSE']:.4f}, R2={s1v25_row['R2']:.4f})
- **최소 측정 (1단계 공정)**: {v25_row['Combo']} / {v25_row['Model']} (RMSE={v25_row['RMSE']:.4f}, R2={v25_row['R2']:.4f})

**데이터 조합 운영 가이드**
- `REF_S4_ALL`: 초기/4.2V/3.6V/2.5V 전체 측정 기반의 최고 정확도 기준선
- `P1_S1+v25`: 초기 + 완전방전(v25) 2단계로 정확도/공정 균형 지점
- `P2_v25_only`: 완전방전 1단계 최소 측정 조건의 실용 대안
"""
)
