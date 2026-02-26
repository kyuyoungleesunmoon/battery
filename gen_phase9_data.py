"""Phase 9 결과를 CSV로 미리 생성하는 스크립트."""
import warnings
warnings.filterwarnings('ignore')

import time, sys
import numpy as np
import pandas as pd

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
    HAS_FLAML = False

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

# ── 1. 데이터 로드 ──
raw = pd.read_csv('./data.csv')
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
print(f'데이터: {len(df_clean)}개 (이상치 {extreme_mask.sum()}개 제거)')

# ── 2. 분할 ──
X_orig = df_clean[ORIG_FEATURES]
y_orig = df_clean[TARGET]
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_orig, y_orig, test_size=0.2, random_state=42
)

# ── 3. 기본 모델 + CatBoost (9-2, 9-3) ──
base_models_spec = {
    'LinearReg': lambda: LinearRegression(),
    'SVR': lambda: SVR(C=1.0, epsilon=0.01),
    'RandomForest': lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost': lambda: xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1),
    'LightGBM': lambda: lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
    'CatBoost': lambda: CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_seed=42),
}

scale_needed = {'LinearReg', 'SVR'}
results = []
total = len(COMBOS) * len(base_models_spec)
done = 0

for combo_name, feats in COMBOS.items():
    X_tr = X_train_o[feats]
    X_te = X_test_o[feats]
    scaler = RobustScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=feats, index=X_tr.index)
    X_te_sc = pd.DataFrame(scaler.transform(X_te), columns=feats, index=X_te.index)

    for model_name, model_fn in base_models_spec.items():
        model = model_fn()
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
            'Step': STEP_MAP[combo_name],
            'Model': model_name,
            'N_Features': len(feats),
            'RMSE': float(np.sqrt(mean_squared_error(y_test_o, y_pred))),
            'R2': float(r2_score(y_test_o, y_pred)),
            'MAPE': float(mean_absolute_percentage_error(y_test_o, y_pred)),
            'MAE': float(mean_absolute_error(y_test_o, y_pred)),
            'TrainSec': round(time.time() - start, 2),
            'Source': 'base',
        })
        done += 1
        print(f'  [{done}/{total}] {combo_name} / {model_name} done')

# ── 4. FLAML 강화 (9-5) ──
if HAS_FLAML:
    p9_df_temp = pd.DataFrame(results)
    combo_best_temp = p9_df_temp.sort_values('RMSE').groupby('Combo', as_index=False).first()
    top5 = combo_best_temp.sort_values('RMSE').head(5)['Combo'].tolist()
    for must in ['P1_S1+v25', 'P2_v25_only', 'REF_S4_ALL']:
        if must not in top5:
            top5.append(must)

    for i, combo_name in enumerate(top5, 1):
        feats = COMBOS[combo_name]
        X_tr = X_train_o[feats]
        X_te = X_test_o[feats]
        print(f'  FLAML [{i}/{len(top5)}] {combo_name} ...')

        automl = AutoML()
        automl.fit(
            X_train=X_tr, y_train=y_train_o,
            time_budget=30, metric='r2', task='regression',
            eval_method='cv', n_splits=5, verbose=0,
            log_file_name=f'flaml_gen_{combo_name}.log',
        )
        pred = np.asarray(automl.predict(X_te), dtype=float)
        results.append({
            'Phase': combo_name.split('_')[0],
            'Combo': combo_name,
            'Step': STEP_MAP[combo_name],
            'Model': f'FLAML({automl.best_estimator})',
            'N_Features': len(feats),
            'RMSE': float(np.sqrt(mean_squared_error(y_test_o, pred))),
            'R2': float(r2_score(y_test_o, pred)),
            'MAPE': float(mean_absolute_percentage_error(y_test_o, pred)),
            'MAE': float(mean_absolute_error(y_test_o, pred)),
            'TrainSec': 30,
            'Source': 'flaml_enhanced',
        })

# ── 5. 피처 상관 ──
corr_rows = []
for f in ORIG_FEATURES:
    corr_rows.append({'Feature': f, 'Correlation': float(df_clean[f].corr(df_clean[TARGET]))})
corr_df = pd.DataFrame(corr_rows)

# ── 6. 저장 ──
final_df = pd.DataFrame(results)
final_df.to_csv('phase9_results.csv', index=False, encoding='utf-8-sig')
corr_df.to_csv('phase9_corr.csv', index=False, encoding='utf-8-sig')

print(f'\n저장 완료: phase9_results.csv ({len(final_df)}행), phase9_corr.csv ({len(corr_df)}행)')
print('총 소요 시간은 터미널에서 확인하세요.')
