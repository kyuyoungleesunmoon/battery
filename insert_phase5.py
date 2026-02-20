"""
Phase 5 결과를 train.ipynb에 삽입합니다.
"""
import json
import pandas as pd

# 결과 파일 로드 (텍스트 파싱 대신 하드코딩된 요약 정보 사용 또는 파싱)
# 여기서는 텍스트 파일 내용을 참조하여 직접 셀 내용을 구성합니다.

with open('train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def make_md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}

def make_code(lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in lines]}

cells = []

# Phase 5 헤더
cells.append(make_md([
    "# Phase 5: 모델 학습 및 평가",
    "",
    "Phase 4에서 준비한 4가지 시나리오 데이터셋에 대해 다음 모델들을 학습하고 평가합니다.",
    "",
    "1. **Linear Regression** (Baseline, RobustScaler)",
    "2. **SVR** (Support Vector Regression, RobustScaler)",
    "3. **Random Forest** (Ensemble, No Scaler)",
    "4. **XGBoost** (Boosting, No Scaler)",
    "5. **LightGBM** (Boosting, No Scaler)",
    "",
    "평가지표: **RMSE** (Root Mean Squared Error), **R²** (Coefficient of Determination), **MAPE**"
]))

# 코드 셀 (모델링)
cells.append(make_code([
    "# ============================================================",
    "# Phase 5. 모델 학습 및 평가 Loop",
    "# ============================================================",
    "import time",
    "from sklearn.linear_model import LinearRegression",
    "from sklearn.svm import SVR",
    "from sklearn.ensemble import RandomForestRegressor",
    "import xgboost as xgb",
    "import lightgbm as lgb",
    "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error",
    "",
    "# 모델 정의",
    "models = {",
    "    'LinearReg': LinearRegression(),",
    "    'SVR': SVR(C=1.0, epsilon=0.01),",
    "    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),",
    "    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),",
    "    'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)",
    "}",
    "",
    "results_list = []",
    "",
    "print('🚀 모델 학습 및 평가 시작...')",
    "print('=' * 80)",
    "",
    "for s_name, data in scenario_datasets.items():",
    "    X_tr = data['X_train']",
    "    X_te = data['X_test']",
    "    feats = data['features']",
    "    ",
    "    # RobustScaler (선형 모델용)",
    "    scaler = RobustScaler()",
    "    X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), columns=feats, index=X_tr.index)",
    "    X_te_scaled = pd.DataFrame(scaler.transform(X_te), columns=feats, index=X_te.index)",
    "    ",
    "    print(f'\\n[{s_name}] (피처 {len(feats)}개)')",
    "    print('-' * 60)",
    "    ",
    "    for m_name, model in models.items():",
    "        # 스케일링 적용 여부",
    "        if m_name in ['LinearReg', 'SVR']:",
    "            X_train_curr, X_test_curr = X_tr_scaled, X_te_scaled",
    "        else:",
    "            X_train_curr, X_test_curr = X_tr, X_te",
    "            ",
    "        start = time.time()",
    "        model.fit(X_train_curr, y_train)",
    "        train_time = time.time() - start",
    "        ",
    "        y_pred = model.predict(X_test_curr)",
    "        rmse = mean_squared_error(y_test, y_pred, squared=False)",
    "        r2 = r2_score(y_test, y_pred)",
    "        mape = mean_absolute_percentage_error(y_test, y_pred)",
    "        ",
    "        print(f'{m_name:12s} | RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.4f} ({train_time:.2f}s)')",
    "        ",
    "        results_list.append({",
    "            'Scenario': s_name, 'Model': m_name,",
    "            'RMSE': rmse, 'R2': r2, 'MAPE': mape",
    "        })",
    "",
    "results_df = pd.DataFrame(results_list)"
]))

# 결과 요약 마크다운
cells.append(make_md([
    "## 5-2. 결과 분석 및 시각화",
    "",
    "### 📊 시나리오별 성능 요약",
    "",
    "| 시나리오 | Best Model | RMSE | R² | 비고 |",
    "|---|---|---|---|---|",
    "| **S1 (Initial)** | LinearReg | 0.0355 | **0.0548** | 예측력 거의 없음 |",
    "| **S2 (+4.2V)** | SVR | 0.0348 | **0.0948** | 여전히 낮음 |",
    "| **S3 (+3.6V)** | RandomForest | 0.0350 | **0.0838** | 개선 없음 |",
    "| **S4 (Full)** | **XGBoost** | **0.0246** | **0.5454** | **유일하게 유의미한 예측 성능** |",
    "",
    "> **결론**: 방전 종지 전압(2.5V) 관련 데이터(`voltage_sag`, `v25_voltage`) 없이는 배터리 용량 예측이 불가능함."
]))

# 시각화 코드
cells.append(make_code([
    "# ============================================================",
    "# 5-2. 성능 비교 시각화",
    "# ============================================================",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "",
    "plt.figure(figsize=(14, 6))",
    "sns.barplot(data=results_df, x='Scenario', y='R2', hue='Model')",
    "plt.title('R2 Score Comparison by Scenario and Model', fontsize=14, fontweight='bold')",
    "plt.axhline(0, color='black', linewidth=0.5)",
    "plt.axhline(0.5, color='red', linestyle='--', label='R2=0.5')",
    "plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')",
    "plt.tight_layout()",
    "plt.show()"
]))

# 피처 중요도 코드
cells.append(make_code([
    "# ============================================================",
    "# 5-3. Feature Importance (Scenario 4 - XGBoost)",
    "# ============================================================",
    "# S4 데이터로 XGBoost 재학습 및 피처 중요도 추출",
    "xgb_best = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)",
    "xgb_best.fit(scenario_datasets['S4_ALL']['X_train'], y_train)",
    "",
    "importances = xgb_best.feature_importances_",
    "feats = scenario_datasets['S4_ALL']['features']",
    "indices = np.argsort(importances)[::-1]",
    "",
    "plt.figure(figsize=(10, 8))",
    "plt.title('Feature Importance (Scenario 4 - XGBoost)', fontsize=14, fontweight='bold')",
    "plt.barh(range(len(indices)), importances[indices], align='center', color='teal')",
    "plt.yticks(range(len(indices)), [feats[i] for i in indices])",
    "plt.xlabel('Relative Importance')",
    "plt.gca().invert_yaxis()  # 상위 피처가 위로 오게",
    "plt.tight_layout()",
    "plt.show()",
    "",
    "print('🏆 Top 5 Features:')",
    "for i in range(5):",
    "    print(f'  {i+1}. {feats[indices[i]]:25s}: {importances[indices[i]]:.4f}')"
]))

cells.append(make_md([
    "### 🔑 핵심 피처 분석",
    "",
    "1. **`voltage_sag` (중요도 0.39)**: 초기 전압과 2.5V 도달 시점의 전압 차이. 압도적인 중요도 1위.",
    "2. **`v25_voltage` (중요도 0.13)**: 방전 종지 전압 자체도 중요.",
    "3. **`impedance_mean` (중요도 0.10)**: 내부저항 평균도 기여함.",
    "",
    "> **최종 결론**: 배터리 용량 예측을 위해서는 **완전 방전 테스트(2.5V)**가 필수적이며, 이 과정에서 얻어지는 전압 강하(`voltage_sag`)가 가장 강력한 예측 인자입니다."
]))

# 노트북에 추가
for cell in cells:
    nb['cells'].append(cell)

with open('train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"완료! 총 {len(nb['cells'])}개 셀 ({len(cells)}개 Phase 5 셀 추가)")
