"""
배터리 Capacity 예측 — 최적 측정 조합 탐색
============================================
목표: S1(초기측정) + 1종류 측정만으로 최대 성능 달성

Phase 1: S1 + 단일 측정 조합 (3가지)
  - S1+v42: 초기+풀충전 (=기존S2)
  - S1+v25: 초기+완전방전 ★핵심
  - S1+v36: 초기+방전중간

Phase 2: 단일 측정만 (initial 없이, 3가지)
  - v25 only / v42 only / v36 only

Phase 3: 전압 vs 임피던스 분리 (5가지)
  - S1+v25 voltage only / All voltage only / All impedance only
  - S1+v25 impedance only / v25_voltage+v25_impedance

Phase 4: 최적 조합 FLAML 강화 학습

총 11개 조합 × 9모델 = 99 실험
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from flaml import AutoML
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_percentage_error, mean_absolute_error
)

# ============================================================
# 1. 데이터 로드 및 전처리
# ============================================================
print('=' * 70)
print('📦 데이터 로드 — 최적 측정 조합 탐색')
print('=' * 70)

raw = pd.read_csv('./data.csv')

new_columns = [
    'cell_id', 'initial_label', 'initial_voltage', 'initial_impedance',
    'v42_label', 'v42_voltage', 'v42_impedance',
    'v25_label', 'v25_voltage', 'v25_impedance',
    'v36_label', 'v36_voltage', 'v36_empty', 'v36_impedance',
    'capacity'
]
raw.columns = new_columns

drop_cols = ['initial_label', 'v42_label', 'v25_label', 'v36_label', 'v36_empty']
df = raw.drop(columns=drop_cols)

numeric_cols = [c for c in df.columns if c != 'cell_id']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

TARGET = 'capacity'

# 극단 이상치 제거
extreme_mask = (df['v36_impedance'] - df['initial_impedance']).abs() > 100
df_clean = df[~extreme_mask].copy()
print(f'데이터: {len(df)}개 → {len(df_clean)}개 (이상치 {extreme_mask.sum()}개 제거)')

# ============================================================
# 2. 실험 조합 정의
# ============================================================
combos = {}

# --- Phase 1: S1 + 단일 측정 ---
combos['P1_S1+v42'] = ['initial_voltage', 'initial_impedance',
                        'v42_voltage', 'v42_impedance']
combos['P1_S1+v25'] = ['initial_voltage', 'initial_impedance',
                        'v25_voltage', 'v25_impedance']
combos['P1_S1+v36'] = ['initial_voltage', 'initial_impedance',
                        'v36_voltage', 'v36_impedance']

# --- Phase 2: 단일 측정만 (initial 없이) ---
combos['P2_v42_only'] = ['v42_voltage', 'v42_impedance']
combos['P2_v25_only'] = ['v25_voltage', 'v25_impedance']
combos['P2_v36_only'] = ['v36_voltage', 'v36_impedance']

# --- Phase 3: 전압 vs 임피던스 분리 ---
combos['P3_all_voltage'] = ['initial_voltage', 'v42_voltage',
                             'v25_voltage', 'v36_voltage']
combos['P3_all_impedance'] = ['initial_impedance', 'v42_impedance',
                               'v25_impedance', 'v36_impedance']
combos['P3_S1v25_volt_only'] = ['initial_voltage', 'v25_voltage']
combos['P3_S1v25_imp_only'] = ['initial_impedance', 'v25_impedance']
combos['P3_v25_single_volt'] = ['v25_voltage']

# --- 기준선 비교용 ---
combos['REF_S1_only'] = ['initial_voltage', 'initial_impedance']
combos['REF_S4_ALL'] = ['initial_voltage', 'initial_impedance',
                         'v42_voltage', 'v42_impedance',
                         'v25_voltage', 'v25_impedance',
                         'v36_voltage', 'v36_impedance']

print(f'\n총 {len(combos)}개 조합 실험')
print('=' * 70)

# 상관분석 출력
print('\n📊 피처별 capacity 상관계수:')
all_feats = ['initial_voltage', 'initial_impedance',
             'v42_voltage', 'v42_impedance',
             'v25_voltage', 'v25_impedance',
             'v36_voltage', 'v36_impedance']
for f in all_feats:
    r = df_clean[f].corr(df_clean[TARGET])
    bar = '█' * int(abs(r) * 50)
    print(f'  {f:25s}: r={r:+.4f}  {bar}')

print('\n📋 실험 조합 목록:')
for name, feats in combos.items():
    max_r = max(abs(df_clean[f].corr(df_clean[TARGET])) for f in feats)
    print(f'  {name:25s} ({len(feats)}개 피처) | 최대|r|={max_r:.3f} | {feats}')

# ============================================================
# 3. Train/Test 분할
# ============================================================
ALL_FEATURES = all_feats
X = df_clean[ALL_FEATURES]
y = df_clean[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'\nTrain: {len(X_train)}개, Test: {len(X_test)}개')

# ============================================================
# 4. Phase 1~3: 기본 모델 학습 (5모델 × 13조합)
# ============================================================
print('\n' + '=' * 70)
print('🚀 Phase 1~3: 기본 모델 학습 (5모델 × 13조합 = 65 실험)')
print('=' * 70)

scale_needed = ['LinearReg', 'SVR']
all_results = []

for c_name, feats in combos.items():
    print(f'\n[{c_name}] ({len(feats)}개 피처: {", ".join(feats)})')
    print('-' * 60)

    X_tr = X_train[feats]
    X_te = X_test[feats]

    scaler = RobustScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=feats, index=X_tr.index)
    X_te_sc = pd.DataFrame(scaler.transform(X_te), columns=feats, index=X_te.index)

    models = {
        'LinearReg': LinearRegression(),
        'SVR': SVR(C=1.0, epsilon=0.01),
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    }

    for m_name, model in models.items():
        if m_name in scale_needed:
            X_train_f, X_test_f = X_tr_sc, X_te_sc
        else:
            X_train_f, X_test_f = X_tr, X_te

        start = time.time()
        model.fit(X_train_f, y_train)
        elapsed = time.time() - start

        y_pred = model.predict(X_test_f)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f'  {m_name:15s} | RMSE={rmse:.4f}, R²={r2:.4f}, MAPE={mape:.4f} ({elapsed:.1f}s)')

        all_results.append({
            'Phase': c_name.split('_')[0],
            'Combo': c_name, 'Model': m_name, 'N_Features': len(feats),
            'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'MAE': mae
        })

# ============================================================
# 5. Phase 1~3: AutoML 학습 (상위 조합만)
# ============================================================
print('\n' + '=' * 70)
print('🚀 AutoML 벤치마킹 (CatBoost + FLAML × 13조합)')
print('=' * 70)

for c_name, feats in combos.items():
    print(f'\n▶️ {c_name} ({len(feats)}개 피처)')

    X_tr = X_train[feats]
    X_te = X_test[feats]

    # CatBoost
    cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                           verbose=0, random_seed=42)
    cb.fit(X_tr, y_train)
    pred_cb = cb.predict(X_te)
    r2_val = r2_score(y_test, pred_cb)
    rmse_val = np.sqrt(mean_squared_error(y_test, pred_cb))
    mae_val = mean_absolute_error(y_test, pred_cb)
    mape_val = mean_absolute_percentage_error(y_test, pred_cb)
    print(f'  CatBoost     | RMSE={rmse_val:.4f}, R²={r2_val:.4f}')
    all_results.append({
        'Phase': c_name.split('_')[0],
        'Combo': c_name, 'Model': 'CatBoost', 'N_Features': len(feats),
        'RMSE': rmse_val, 'R2': r2_val, 'MAPE': mape_val, 'MAE': mae_val
    })

    # FLAML (30s)
    flaml_automl = AutoML()
    flaml_automl.fit(
        X_train=X_tr, y_train=y_train,
        time_budget=30, metric='r2', task='regression',
        eval_method='cv',
        log_file_name=f'flaml_combo_{c_name}.log',
        verbose=0
    )
    pred_fl = flaml_automl.predict(X_te)
    r2_val = r2_score(y_test, pred_fl)
    rmse_val = np.sqrt(mean_squared_error(y_test, pred_fl))
    mae_val = mean_absolute_error(y_test, pred_fl)
    mape_val = mean_absolute_percentage_error(y_test, pred_fl)
    best_est = flaml_automl.best_estimator
    print(f'  FLAML({best_est:15s}) | RMSE={rmse_val:.4f}, R²={r2_val:.4f}')
    all_results.append({
        'Phase': c_name.split('_')[0],
        'Combo': c_name, 'Model': f'FLAML({best_est})', 'N_Features': len(feats),
        'RMSE': rmse_val, 'R2': r2_val, 'MAPE': mape_val, 'MAE': mae_val
    })

# ============================================================
# 6. 종합 결과 분석
# ============================================================
print('\n' + '=' * 70)
print('📊 종합 결과 분석')
print('=' * 70)

results_df = pd.DataFrame(all_results)

# 조합별 최고 성능
print('\n[조합별 최고 성능 — RMSE 기준]')
print(f'{"조합":25s} | {"피처수":>4s} | {"최고모델":20s} | {"RMSE":>8s} | {"R²":>8s} | {"MAPE":>8s}')
print('-' * 85)

combo_bests = []
for c_name in combos.keys():
    subset = results_df[results_df['Combo'] == c_name]
    best = subset.loc[subset['RMSE'].idxmin()]
    combo_bests.append(best)
    print(f'{c_name:25s} | {int(best["N_Features"]):4d} | {best["Model"]:20s} | {best["RMSE"]:.4f}  | {best["R2"]:.4f}  | {best["MAPE"]*100:.2f}%')

# Phase별 최고
print('\n[Phase별 최고 성능]')
for phase in ['P1', 'P2', 'P3', 'REF']:
    subset = results_df[results_df['Phase'] == phase]
    if len(subset) == 0:
        continue
    best = subset.loc[subset['RMSE'].idxmin()]
    print(f'  {phase}: {best["Combo"]} + {best["Model"]} → RMSE={best["RMSE"]:.4f}, R²={best["R2"]:.4f}')

# 전체 Top 10
print('\n[전체 Top 10 — RMSE 순]')
top10 = results_df.nsmallest(10, 'RMSE')
print(f'{"순위":>4s} | {"조합":25s} | {"모델":20s} | {"피처수":>4s} | {"RMSE":>8s} | {"R²":>8s} | {"MAPE":>8s}')
print('-' * 100)
for i, (_, row) in enumerate(top10.iterrows(), 1):
    print(f'{i:4d} | {row["Combo"]:25s} | {row["Model"]:20s} | {int(row["N_Features"]):4d} | {row["RMSE"]:.4f}  | {row["R2"]:.4f}  | {row["MAPE"]*100:.2f}%')

# ============================================================
# 7. 핵심 비교: 측정 공정 효율성 분석
# ============================================================
print('\n' + '=' * 70)
print('📈 측정 공정 효율성 분석')
print('=' * 70)

combo_best_df = pd.DataFrame([{
    'Combo': b['Combo'], 'Model': b['Model'],
    'N_Features': int(b['N_Features']),
    'RMSE': b['RMSE'], 'R2': b['R2'], 'MAPE': b['MAPE']
} for b in combo_bests]).sort_values('RMSE')

# 측정 스텝 수 매핑
step_map = {
    'P1_S1+v42': '2단계(초기+풀충전)',
    'P1_S1+v25': '2단계(초기+완전방전)',
    'P1_S1+v36': '2단계(초기+방전중간)',
    'P2_v42_only': '1단계(풀충전)',
    'P2_v25_only': '1단계(완전방전)',
    'P2_v36_only': '1단계(방전중간)',
    'P3_all_voltage': '4단계(전압만)',
    'P3_all_impedance': '4단계(임피던스만)',
    'P3_S1v25_volt_only': '2단계(전압만)',
    'P3_S1v25_imp_only': '2단계(임피던스만)',
    'P3_v25_single_volt': '1단계(v25전압1개)',
    'REF_S1_only': '1단계(초기만)',
    'REF_S4_ALL': '4단계(전체)',
}

print(f'\n{"조합":25s} | {"측정공정":20s} | {"피처수":>4s} | {"R²":>8s} | {"RMSE":>8s} | S4대비')
print('-' * 95)

ref_s4 = combo_best_df[combo_best_df['Combo'] == 'REF_S4_ALL']
r2_s4 = ref_s4['R2'].values[0] if len(ref_s4) > 0 else 1.0

for _, row in combo_best_df.iterrows():
    step = step_map.get(row['Combo'], '?')
    pct = (row['R2'] / r2_s4 * 100) if r2_s4 > 0 else 0
    bar = '█' * int(pct / 5)
    print(f'{row["Combo"]:25s} | {step:20s} | {int(row["N_Features"]):4d} | {row["R2"]:.4f}  | {row["RMSE"]:.4f}  | {pct:5.1f}% {bar}')

# ============================================================
# 8. Phase 4: 최적 조합 FLAML 강화 학습 (60s)
# ============================================================
print('\n' + '=' * 70)
print('🔥 Phase 4: 최적 조합 FLAML 강화 학습 (60s budget)')
print('=' * 70)

# 상위 3개 조합에 대해 FLAML 강화
top3_combos = combo_best_df.head(5)['Combo'].tolist()
# 반드시 P1_S1+v25 포함
if 'P1_S1+v25' not in top3_combos:
    top3_combos.append('P1_S1+v25')
# 반드시 P2_v25_only 포함
if 'P2_v25_only' not in top3_combos:
    top3_combos.append('P2_v25_only')

enhanced_results = []
for c_name in top3_combos:
    feats = combos[c_name]
    print(f'\n▶️ {c_name} ({len(feats)}개 피처) — FLAML 60s 강화')

    X_tr = X_train[feats]
    X_te = X_test[feats]

    flaml_enhanced = AutoML()
    flaml_enhanced.fit(
        X_train=X_tr, y_train=y_train,
        time_budget=60, metric='r2', task='regression',
        eval_method='cv', n_splits=10,
        log_file_name=f'flaml_enhanced_{c_name}.log',
        verbose=0
    )
    pred = flaml_enhanced.predict(X_te)
    r2_val = r2_score(y_test, pred)
    rmse_val = np.sqrt(mean_squared_error(y_test, pred))
    mae_val = mean_absolute_error(y_test, pred)
    mape_val = mean_absolute_percentage_error(y_test, pred)
    best_est = flaml_enhanced.best_estimator

    print(f'  Best: {best_est} | RMSE={rmse_val:.4f}, R²={r2_val:.4f}, MAPE={mape_val*100:.2f}%')

    enhanced_results.append({
        'Combo': c_name, 'Model': f'FLAML_Enhanced({best_est})',
        'N_Features': len(feats), 'Features': feats,
        'RMSE': rmse_val, 'R2': r2_val, 'MAPE': mape_val, 'MAE': mae_val
    })

enh_df = pd.DataFrame(enhanced_results).sort_values('RMSE')

print('\n[Phase 4 강화 학습 결과]')
for _, row in enh_df.iterrows():
    print(f'  {row["Combo"]:25s} | {row["Model"]:30s} | RMSE={row["RMSE"]:.4f} | R²={row["R2"]:.4f} | MAPE={row["MAPE"]*100:.2f}%')

# ============================================================
# 9. 최종 결론
# ============================================================
print('\n' + '=' * 70)
print('🏆 최종 결론')
print('=' * 70)

# 전체 최고
overall_best = results_df.loc[results_df['RMSE'].idxmin()]
print(f'\n[전체 최고 성능]')
print(f'  조합: {overall_best["Combo"]}')
print(f'  모델: {overall_best["Model"]}')
print(f'  RMSE={overall_best["RMSE"]:.6f}, R²={overall_best["R2"]:.4f}, MAPE={overall_best["MAPE"]*100:.2f}%')

# S1+v25 최고
s1v25_best = results_df[results_df['Combo'] == 'P1_S1+v25']
if len(s1v25_best) > 0:
    s1v25_top = s1v25_best.loc[s1v25_best['RMSE'].idxmin()]
    print(f'\n[S1+v25 (초기+완전방전) 최고 성능]')
    print(f'  모델: {s1v25_top["Model"]}')
    print(f'  RMSE={s1v25_top["RMSE"]:.6f}, R²={s1v25_top["R2"]:.4f}, MAPE={s1v25_top["MAPE"]*100:.2f}%')
    print(f'  → S4 대비 R² 달성률: {s1v25_top["R2"]/overall_best["R2"]*100:.1f}%')

# v25 only 최고
v25_best = results_df[results_df['Combo'] == 'P2_v25_only']
if len(v25_best) > 0:
    v25_top = v25_best.loc[v25_best['RMSE'].idxmin()]
    print(f'\n[v25만 (완전방전만) 최고 성능]')
    print(f'  모델: {v25_top["Model"]}')
    print(f'  RMSE={v25_top["RMSE"]:.6f}, R²={v25_top["R2"]:.4f}, MAPE={v25_top["MAPE"]*100:.2f}%')

# 실용적 추천
print(f'\n[실용적 추천]')
print(f'  ✅ 최고 성능:          S4_ALL (8피처, 4단계 측정)')
print(f'  ✅ 최적 효율:          S1+v25 (4피처, 2단계 측정) — 초기 + 완전방전만')
print(f'  ✅ 최소 측정:          v25 only (2피처, 1단계 측정) — 완전방전만')
print('=' * 70)
