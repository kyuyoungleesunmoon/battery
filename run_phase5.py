"""
Phase 5: 모델 학습 및 평가
- 베이스라인: Linear Regression
- 앙상블: Random Forest, XGBoost, LightGBM
- SVR
- 시나리오별(S1~S4) 성능 비교 (RMSE, MAE, R2, MAPE)
"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
import warnings
import time
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 데이터 로드 및 전처리 (Phase 2~4 재현)
# ============================================================
print("데이터 로드 및 전처리 중...")
raw = pd.read_csv('data.csv')
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
for col in [c for c in df.columns if c != 'cell_id']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

FEATURES = [c for c in df.columns if c not in ['cell_id', 'capacity']]
TARGET = 'capacity'

# 파생변수 생성
z_initial, z_42, z_25, z_36 = 'initial_impedance', 'v42_impedance', 'v25_impedance', 'v36_impedance'
v_initial, v_42, v_25, v_36 = 'initial_voltage', 'v42_voltage', 'v25_voltage', 'v36_voltage'
impedance_cols_list = [z_initial, z_42, z_25, z_36]

df['impedance_delta_42'] = df[z_42] - df[z_initial]
df['impedance_delta_25'] = df[z_25] - df[z_initial]
df['impedance_delta_36'] = df[z_36] - df[z_initial]
df['impedance_ratio_42'] = df[z_42] / df[z_initial]
df['impedance_ratio_25'] = df[z_25] / df[z_initial]
df['impedance_range'] = df[impedance_cols_list].max(axis=1) - df[impedance_cols_list].min(axis=1)
df['impedance_mean'] = df[impedance_cols_list].mean(axis=1)
df['impedance_std'] = df[impedance_cols_list].std(axis=1)
df['voltage_delta_initial_42'] = df[v_42] - df[v_initial]
df['voltage_delta_42_25'] = df[v_42] - df[v_25]
df['voltage_delta_42_36'] = df[v_42] - df[v_36]
df['voltage_sag'] = df[v_initial] - df[v_25]
df['ocv_deviation'] = df[v_initial] - 3.6
df['impedance_voltage_product'] = df[z_initial] * df[v_initial]
df['power_loss_indicator'] = df[z_42] * (df[v_42] - df[v_25])
df['health_index'] = df[v_42] / df[z_42]

ALL_DERIVED = ['impedance_delta_42','impedance_delta_25','impedance_delta_36',
               'impedance_ratio_42','impedance_ratio_25','impedance_range',
               'impedance_mean','impedance_std','voltage_delta_initial_42',
               'voltage_delta_42_25','voltage_delta_42_36','voltage_sag',
               'ocv_deviation','impedance_voltage_product','power_loss_indicator','health_index']

# 이상치 제거
extreme_mask = df['impedance_delta_36'].abs() > 100
df_clean = df[~extreme_mask].copy()

# 중복 피처 제거
remove_features = ['impedance_delta_42','impedance_delta_25','impedance_delta_36',
                   'impedance_range','impedance_std','voltage_delta_42_25','voltage_delta_42_36']
remaining_derived = [f for f in ALL_DERIVED if f not in remove_features]
remaining_all = FEATURES + remaining_derived

# 시나리오 정의
scenario1 = ['initial_voltage', 'initial_impedance', 'ocv_deviation']
scenario2 = ['initial_voltage', 'initial_impedance', 'v42_voltage', 'v42_impedance',
             'impedance_ratio_42', 'voltage_delta_initial_42',
             'impedance_voltage_product', 'health_index', 'ocv_deviation']
scenario3 = scenario2 + ['v36_voltage', 'v36_impedance', 'impedance_mean']
scenario4 = remaining_all

scenarios = {
    'S1': scenario1,
    'S2': scenario2,
    'S3': scenario3,
    'S4': scenario4,
}

# Train/Test 분할
X = df_clean[remaining_all]
y = df_clean[TARGET]
X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# 모델 정의
# ============================================================
models = {
    'LinearReg': LinearRegression(),
    'SVR': SVR(C=1.0, epsilon=0.01),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
}

# 스케일링이 필요한 모델
scale_needed = ['LinearReg', 'SVR']

# ============================================================
# 학습 및 평가 루프
# ============================================================
results_list = []
results_txt = open('phase5_results.txt', 'w', encoding='utf-8')

results_txt.write("="*80 + "\nPhase 5: 모델 학습 및 평가 결과\n" + "="*80 + "\n")

start_time_total = time.time()

for s_name, feats in scenarios.items():
    results_txt.write(f"\n[{s_name}] (피처 수: {len(feats)}개)\n")
    results_txt.write("-" * 60 + "\n")
    
    # 데이터 준비
    X_tr = X_train_full[feats]
    X_te = X_test_full[feats]
    
    # 스케일러 학습 (RobustScaler)
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    X_tr_scaled_df = pd.DataFrame(X_tr_scaled, columns=feats, index=X_tr.index)
    X_te_scaled_df = pd.DataFrame(X_te_scaled, columns=feats, index=X_te.index)
    
    for m_name, model in models.items():
        # 스케일링 적용 여부 결정
        if m_name in scale_needed:
            X_train_curr = X_tr_scaled_df
            X_test_curr = X_te_scaled_df
        else:
            X_train_curr = X_tr
            X_test_curr = X_te
            
        # 학습
        start_time = time.time()
        model.fit(X_train_curr, y_train)
        train_time = time.time() - start_time
        
        # 예측
        y_pred = model.predict(X_test_curr)
        
        # 평가
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        results_txt.write(f"{m_name:12s} | RMSE={rmse:.6f}, R2={r2:.4f}, MAE={mae:.6f}, MAPE={mape:.6f} ({train_time:.2f}s)\n")
        
        results_list.append({
            'Scenario': s_name,
            'Model': m_name,
            'RMSE': rmse,
            'R2': r2,
            'MAE': mae,
            'MAPE': mape
        })

results_df = pd.DataFrame(results_list)

# ============================================================
# 결과 분석 및 시각화
# ============================================================
# 1. 시나리오별 Best Model 선정
results_txt.write("\n" + "="*80 + "\n시나리오별 최적 모델 (RMSE 기준)\n" + "="*80 + "\n")
best_models = results_df.loc[results_df.groupby('Scenario')['RMSE'].idxmin()]
for _, row in best_models.iterrows():
    results_txt.write(f"{row['Scenario']}: {row['Model']} (RMSE={row['RMSE']:.6f}, R2={row['R2']:.4f})\n")

# 2. 성능 비교 히트맵 (R2 Score)
metrics_pivot = results_df.pivot(index='Model', columns='Scenario', values='R2')
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5)
plt.title('R2 Score Models vs Scenarios')
plt.tight_layout()
plt.savefig('phase5_r2_heatmap.png')
results_txt.write("\n시각화 저장: phase5_r2_heatmap.png\n")

# 3. RMSE 비교 바 차트
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='Scenario', y='RMSE', hue='Model')
plt.title('RMSE Comparison by Scenario and Model')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('phase5_rmse_barplot.png')
results_txt.write("시각화 저장: phase5_rmse_barplot.png\n")

# 4. Feature Importance (Scenario 4 Best Model 기준)
best_model_s4_name = best_models[best_models['Scenario']=='S4']['Model'].values[0]
if best_model_s4_name in ['RandomForest', 'XGBoost', 'LightGBM']:
    model = models[best_model_s4_name] # 이미 S4 데이터로 마지막에 학습됨 (loop 순서상 S4가 마지막)
    # loop 마지막이라 model 객체는 S4로 학습된 상태임
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features_s4 = scenarios['S4']
    
    plt.figure(figsize=(10, 8))
    plt.title(f'Feature Importance (Scenario 4 - {best_model_s4_name})')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [features_s4[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('phase5_feature_importance.png')
    results_txt.write(f"시각화 저장: phase5_feature_importance.png ({best_model_s4_name})\n")
    
    results_txt.write(f"\n[{best_model_s4_name} Feature Importance Top 5]\n")
    for i in range(5):
        results_txt.write(f"  {i+1}. {features_s4[indices[i]]} ({importances[indices[i]]:.4f})\n")

results_txt.close()
print("Phase 5 실행 완료. 결과 -> phase5_results.txt")
