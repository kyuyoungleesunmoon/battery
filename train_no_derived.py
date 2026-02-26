"""
배터리 Capacity 예측 — 파생변수 없이 원본 피처만 사용 (S1~S4 시나리오)
=====================================================================
기존 노트북과 동일한 학습 파이프라인을 파생변수 없이 적용:
  - S1_INITIAL:        initial_voltage, initial_impedance (2개)
  - S2_INITIAL_42V:    + v42_voltage, v42_impedance (4개)
  - S3_INITIAL_42V_36V:+ v36_voltage, v36_impedance (6개)
  - S4_ALL:            + v25_voltage, v25_impedance (8개)

Phase 5: LinearReg, SVR, RandomForest, XGBoost, LightGBM × 4시나리오
Phase 6: CatBoost, SVR(RBF), FLAML, AutoGluon × 4시나리오
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
print('📦 데이터 로드 및 전처리 (파생변수 없음, S1~S4 시나리오)')
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

FEATURES = [
    'initial_voltage', 'initial_impedance',
    'v42_voltage', 'v42_impedance',
    'v25_voltage', 'v25_impedance',
    'v36_voltage', 'v36_impedance'
]
TARGET = 'capacity'

print(f'데이터 Shape: {df.shape}')
print(f'전체 피처 ({len(FEATURES)}개): {FEATURES}')
print(f'타겟: {TARGET}')

# ============================================================
# 2. 극단 이상치 제거 (기존 노트북과 동일 기준)
# ============================================================
extreme_mask = (df['v36_impedance'] - df['initial_impedance']).abs() > 100
n_extreme = extreme_mask.sum()
df_clean = df[~extreme_mask].copy()
print(f'\n극단 이상치 제거: {len(df)}개 → {len(df_clean)}개 (제거: {n_extreme}개)')

# ============================================================
# 3. 시나리오 정의 (파생변수 없이 원본 피처만)
# ============================================================
scenarios = {
    'S1_INITIAL': ['initial_voltage', 'initial_impedance'],
    'S2_INITIAL_42V': ['initial_voltage', 'initial_impedance',
                       'v42_voltage', 'v42_impedance'],
    'S3_INITIAL_42V_36V': ['initial_voltage', 'initial_impedance',
                           'v42_voltage', 'v42_impedance',
                           'v36_voltage', 'v36_impedance'],
    'S4_ALL': FEATURES,  # 8개 전체
}

print('\n📋 시나리오별 피처 구성 (파생변수 없음)')
print('=' * 70)
for name, feats in scenarios.items():
    corrs = [(f, df_clean[f].corr(df_clean[TARGET])) for f in feats]
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    max_r = max(abs(r) for _, r in corrs)
    print(f'\n{name} ({len(feats)}개 피처, 최대|r|={max_r:.3f}):')
    for f, r in corrs:
        strength = '🟢' if abs(r) > 0.5 else ('🟡' if abs(r) > 0.2 else '🔴')
        print(f'  {strength} {f:25s}: r={r:+.4f}')

# ============================================================
# 4. Train/Test 분할 및 시나리오별 데이터셋 구성
# ============================================================
X = df_clean[FEATURES]
y = df_clean[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'\n📊 Train/Test 분할')
print(f'  Train: {len(X_train)}개, Test: {len(X_test)}개')
print(f'  Train Capacity: mean={y_train.mean():.4f}, std={y_train.std():.4f}')
print(f'  Test  Capacity: mean={y_test.mean():.4f}, std={y_test.std():.4f}')

# 시나리오별 데이터셋
scenario_datasets = {}
for name, feats in scenarios.items():
    scenario_datasets[name] = {
        'X_train': X_train[feats], 'X_test': X_test[feats],
        'y_train': y_train, 'y_test': y_test,
        'features': feats
    }

# ============================================================
# 5. Phase 5: 기본 모델 학습 (5모델 × 4시나리오)
# ============================================================
print('\n' + '=' * 70)
print('🚀 Phase 5: 기본 모델 학습 (파생변수 없음, 5모델 × 4시나리오)')
print('=' * 70)

scale_needed = ['LinearReg', 'SVR']
results_p5 = []
trained_models = {}

for s_name, feats in scenarios.items():
    print(f'\n[{s_name}] (Features: {len(feats)}개)')
    print('-' * 60)

    X_tr_curr = X_train[feats]
    X_te_curr = X_test[feats]

    scaler = RobustScaler()
    X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr_curr), columns=feats, index=X_tr_curr.index)
    X_te_scaled = pd.DataFrame(scaler.transform(X_te_curr), columns=feats, index=X_te_curr.index)

    models = {
        'LinearReg': LinearRegression(),
        'SVR': SVR(C=1.0, epsilon=0.01),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    }

    for m_name, model in models.items():
        if m_name in scale_needed:
            X_train_final, X_test_final = X_tr_scaled, X_te_scaled
        else:
            X_train_final, X_test_final = X_tr_curr, X_te_curr

        start = time.time()
        model.fit(X_train_final, y_train)
        elapsed = time.time() - start

        trained_models[f'{s_name}_{m_name}'] = model

        y_pred = model.predict(X_test_final)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f'  {m_name:15s} | RMSE={rmse:.4f}, R²={r2:.4f}, MAPE={mape:.4f}, MAE={mae:.4f} ({elapsed:.2f}s)')

        results_p5.append({
            'Scenario': s_name, 'Model': m_name,
            'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'MAE': mae
        })

p5_df = pd.DataFrame(results_p5)

print('\n' + '=' * 70)
print('🏆 Phase 5 시나리오별 최고 성능 모델:')
for s in scenarios.keys():
    subset = p5_df[p5_df['Scenario'] == s]
    best = subset.loc[subset['RMSE'].idxmin()]
    print(f'  {s}: {best["Model"]} (RMSE={best["RMSE"]:.4f}, R²={best["R2"]:.4f})')

# ============================================================
# 6. Phase 6: Multi-AutoML 벤치마킹 (4트랙 × 4시나리오)
# ============================================================
print('\n' + '=' * 70)
print('🚀 Phase 6: Multi-AutoML 벤치마킹 (파생변수 없음, 4트랙 × 4시나리오)')
print('비교 트랙: 1) CatBoost, 2) SVR, 3) FLAML, 4) AutoGluon')
print('=' * 70)

benchmark_results = []

for name, feats in scenarios.items():
    print(f'\n▶️ 시나리오: {name} ({len(feats)}개 피처)')

    X_train_s = scenario_datasets[name]['X_train']
    X_test_s = scenario_datasets[name]['X_test']

    # === Track 1: CatBoost ===
    print('  - Track 1. CatBoost')
    cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_seed=42)
    cb.fit(X_train_s, y_train)
    pred_cb = cb.predict(X_test_s)
    r2_val = r2_score(y_test, pred_cb)
    rmse_val = np.sqrt(mean_squared_error(y_test, pred_cb))
    mae_val = mean_absolute_error(y_test, pred_cb)
    mape_val = mean_absolute_percentage_error(y_test, pred_cb)
    print(f'    RMSE={rmse_val:.4f}, R²={r2_val:.4f}')
    benchmark_results.append({
        'Scenario': name, 'Model_Track': '1_CatBoost',
        'R2': r2_val, 'RMSE': rmse_val, 'MAE': mae_val, 'MAPE': mape_val
    })

    # === Track 2: SVR (RBF) ===
    print('  - Track 2. SVR (RBF)')
    scaler_svr = RobustScaler()
    X_tr_sc = scaler_svr.fit_transform(X_train_s)
    X_te_sc = scaler_svr.transform(X_test_s)
    svr = SVR(C=1.0, epsilon=0.1, kernel='rbf')
    svr.fit(X_tr_sc, y_train)
    pred_svr = svr.predict(X_te_sc)
    r2_val = r2_score(y_test, pred_svr)
    rmse_val = np.sqrt(mean_squared_error(y_test, pred_svr))
    mae_val = mean_absolute_error(y_test, pred_svr)
    mape_val = mean_absolute_percentage_error(y_test, pred_svr)
    print(f'    RMSE={rmse_val:.4f}, R²={r2_val:.4f}')
    benchmark_results.append({
        'Scenario': name, 'Model_Track': '2_SVR',
        'R2': r2_val, 'RMSE': rmse_val, 'MAE': mae_val, 'MAPE': mape_val
    })

    # === Track 3: FLAML ===
    print('  - Track 3. FLAML (30s budget)')
    flaml_automl = AutoML()
    flaml_settings = {
        'time_budget': 30,
        'metric': 'r2',
        'task': 'regression',
        'eval_method': 'cv',
        'log_file_name': f'flaml_noDerived_{name}.log',
        'verbose': 0
    }
    flaml_automl.fit(X_train=X_train_s, y_train=y_train, **flaml_settings)
    pred_flaml = flaml_automl.predict(X_test_s)
    r2_val = r2_score(y_test, pred_flaml)
    rmse_val = np.sqrt(mean_squared_error(y_test, pred_flaml))
    mae_val = mean_absolute_error(y_test, pred_flaml)
    mape_val = mean_absolute_percentage_error(y_test, pred_flaml)
    print(f'    Best: {flaml_automl.best_estimator} | RMSE={rmse_val:.4f}, R²={r2_val:.4f}')
    benchmark_results.append({
        'Scenario': name, 'Model_Track': f'3_FLAML ({flaml_automl.best_estimator})',
        'R2': r2_val, 'RMSE': rmse_val, 'MAE': mae_val, 'MAPE': mape_val
    })

    # === Track 4: AutoGluon ===
    print('  - Track 4. AutoGluon (60s limit)')
    train_data_ag = pd.concat([X_train_s, y_train], axis=1)
    test_data_ag = pd.concat([X_test_s, y_test], axis=1)
    ag_path = f'AutogluonModels_NoDerived_{name}'
    predictor_ag = TabularPredictor(
        label=TARGET, path=ag_path, eval_metric='r2'
    ).fit(train_data_ag, presets='medium_quality', time_limit=60, verbosity=0)
    pred_ag = predictor_ag.predict(test_data_ag.drop(columns=[TARGET]))
    r2_val = r2_score(y_test, pred_ag)
    rmse_val = np.sqrt(mean_squared_error(y_test, pred_ag))
    mae_val = mean_absolute_error(y_test, pred_ag)
    mape_val = mean_absolute_percentage_error(y_test, pred_ag)
    print(f'    RMSE={rmse_val:.4f}, R²={r2_val:.4f}')
    benchmark_results.append({
        'Scenario': name, 'Model_Track': '4_AutoGluon (Ensemble)',
        'R2': r2_val, 'RMSE': rmse_val, 'MAE': mae_val, 'MAPE': mape_val
    })
    print('  ✓ 완료')

bench_df = pd.DataFrame(benchmark_results)

# ============================================================
# 7. 종합 결과 출력
# ============================================================
print('\n' + '=' * 70)
print('📊 종합 결과 (파생변수 없음, 원본 피처만)')
print('=' * 70)

print('\n[Phase 5 — 기본 모델 (5모델 × 4시나리오)]')
for s in scenarios.keys():
    subset = p5_df[p5_df['Scenario'] == s].sort_values('RMSE')
    best = subset.iloc[0]
    print(f'  {s:25s} | 최고: {best["Model"]:12s} | RMSE={best["RMSE"]:.4f} | R²={best["R2"]:.4f}')

print('\n[Phase 6 — AutoML (4트랙 × 4시나리오)]')
for s in scenarios.keys():
    subset = bench_df[bench_df['Scenario'] == s].sort_values('RMSE')
    best = subset.iloc[0]
    print(f'  {s:25s} | 최고: {best["Model_Track"]:35s} | RMSE={best["RMSE"]:.4f} | R²={best["R2"]:.4f}')

# 전체 최고
all_results = []
for _, row in p5_df.iterrows():
    all_results.append({'Scenario': row['Scenario'], 'Model': row['Model'], 'RMSE': row['RMSE'], 'R2': row['R2'], 'MAPE': row['MAPE'], 'MAE': row['MAE']})
for _, row in bench_df.iterrows():
    all_results.append({'Scenario': row['Scenario'], 'Model': row['Model_Track'], 'RMSE': row['RMSE'], 'R2': row['R2'], 'MAPE': row['MAPE'], 'MAE': row['MAE']})

all_df = pd.DataFrame(all_results).sort_values('RMSE')
best = all_df.iloc[0]

print(f'\n🏆 전체 최고 성능:')
print(f'   시나리오: {best["Scenario"]}')
print(f'   모델: {best["Model"]}')
print(f'   RMSE  = {best["RMSE"]:.6f} Ah')
print(f'   R²    = {best["R2"]:.4f}')
print(f'   MAPE  = {best["MAPE"]*100:.2f}%')
print(f'   MAE   = {best["MAE"]:.6f} Ah')

print('\n📊 전체 결과 테이블 (RMSE 오름차순 Top 10):')
print(all_df.head(10).to_string(index=False))
print('=' * 70)
