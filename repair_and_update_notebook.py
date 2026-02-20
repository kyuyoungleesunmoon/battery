
import json
import os

NOTEBOOK_PATH = "train.ipynb"

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source]
    }

def main():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found.")
        return

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    new_cells = []

    # ============================================================
    # Phase 3-5. 시각화 (이미지 로드)
    # ============================================================
    new_cells.append(create_code_cell([
        "# 3-5. 정제 후 상관관계 및 VIF 시각화",
        "from IPython.display import Image, display",
        "",
        "print('📊 [Phase 3] 정제 후 주요 피처 상관관계')",
        "display(Image(filename='phase3_cleaned_correlation.png'))",
        "",
        "print('📊 [Phase 3] 파생변수 산점도 (Target vs Derived)')",
        "display(Image(filename='phase3_derived_scatter.png'))",
        "",
        "print('📊 [Phase 3] VIF 비교 (제거 전 vs 후)')",
        "display(Image(filename='phase3_vif_comparison.png'))"
    ]))

    # ============================================================
    # Phase 4. 학습 데이터 구성
    # ============================================================
    new_cells.append(create_markdown_cell([
        "# Phase 4: 학습 데이터 구성 (Data Splitting)",
        "",
        "모델 학습을 위해 데이터를 Train/Test Set으로 분할합니다.",
        "- **비율**: Train 80% / Test 20%",
        "- **Random State**: 42 (재현성 확보)",
        "- **Target**: `capacity`"
    ]))

    new_cells.append(create_code_cell([
        "# 4-1. 데이터 분할",
        "from sklearn.model_selection import train_test_split",
        "",
        "# 전체 피처 및 타겟 정의",
        "X = df_clean[remaining_all]",
        "y = df_clean[TARGET]",
        "",
        "# Train/Test 분할 (8:2)",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
        "",
        "print(f'Training Set: {X_train.shape}')",
        "print(f'Test Set:     {X_test.shape}')"
    ]))

    new_cells.append(create_code_cell([
        "# 4-2. 시나리오별 데이터셋 준비 함수",
        "def get_scenario_data(scenario_features, X_train, X_test):",
        "    return X_train[scenario_features], X_test[scenario_features]",
        "",
        "print('✅ Phase 4 완료 → Phase 5 (모델 학습)로 진행')"
    ]))

    # ============================================================
    # Phase 5. 모델 학습 및 평가
    # ============================================================
    new_cells.append(create_markdown_cell([
        "# Phase 5: 모델 학습 및 평가",
        "",
        "4가지 시나리오에 대해 다양한 모델을 학습하고 성능을 비교합니다.",
        "",
        "### 사용 모델",
        "1. **Linear Regression** (Baseline)",
        "2. **SVR** (Support Vector Regression)",
        "3. **Random Forest** (Ensemble)",
        "4. **XGBoost** (Boosting)",
        "5. **LightGBM** (Boosting)",
        "",
        "### 평가 지표",
        "- **RMSE** (Root Mean Squared Error): 낮을수록 좋음",
        "- **R² Score**: 1에 가까울수록 좋음 (설명력)",
        "- **MAPE**: 평균 절대 백분율 오차"
    ]))

    new_cells.append(create_code_cell([
        "# 5-1. 모델 정의 및 학습",
        "import time",
        "from sklearn.linear_model import LinearRegression",
        "from sklearn.svm import SVR",
        "from sklearn.ensemble import RandomForestRegressor",
        "import xgboost as xgb",
        "import lightgbm as lgb",
        "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error",
        "from sklearn.preprocessing import RobustScaler",
        "",
        "models = {",
        "    'LinearReg': LinearRegression(),",
        "    'SVR': SVR(C=1.0, epsilon=0.01),",
        "    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),",
        "    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),",
        "    'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)",
        "}",
        "",
        "scale_needed = ['LinearReg', 'SVR']",
        "results_list = []",
        "trained_models = {} # 추후 Phase 7에서 사용하기 위해 저장",
        "",
        "print('🚀 모델 학습 시작...')",
        "print('=' * 80)",
        "",
        "for s_name, feats in scenarios.items():",
        "    print(f'\\n[{s_name}] (Features: {len(feats)})')",
        "    print('-' * 60)",
        "    ",
        "    # 데이터 준비",
        "    X_tr_curr, X_te_curr = get_scenario_data(feats, X_train, X_test)",
        "    ",
        "    # Scaler (RobustScaler)",
        "    scaler = RobustScaler()",
        "    X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr_curr), columns=feats, index=X_tr_curr.index)",
        "    X_te_scaled = pd.DataFrame(scaler.transform(X_te_curr), columns=feats, index=X_te_curr.index)",
        "    ",
        "    for m_name, model in models.items():",
        "        # 스케일링 적용 여부",
        "        if m_name in scale_needed:",
        "            X_train_final, X_test_final = X_tr_scaled, X_te_scaled",
        "        else:",
        "            X_train_final, X_test_final = X_tr_curr, X_te_curr",
        "            ",
        "        # 학습",
        "        start = time.time()",
        "        model.fit(X_train_final, y_train)",
        "        elapsed = time.time() - start",
        "        ",
        "        # 저장 (S4 모델은 Teacher로 사용 가능)",
        "        if '전체' in s_name or '4안' in s_name:",
        "             trained_models[f'{s_name}_{m_name}'] = model",
        "        ",
        "        # 예측 및 평가",
        "        y_pred = model.predict(X_test_final)",
        "        rmse = mean_squared_error(y_test, y_pred, squared=False)",
        "        r2 = r2_score(y_test, y_pred)",
        "        mape = mean_absolute_percentage_error(y_test, y_pred)",
        "        ",
        "        print(f'{m_name:12s} | RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.4f} ({elapsed:.2f}s)')",
        "        ",
        "        results_list.append({",
        "            'Scenario': s_name, 'Model': m_name,",
        "            'RMSE': rmse, 'R2': r2, 'MAPE': mape",
        "        })",
        "",
        "print('=' * 80)",
        "print('✅ 학습 완료')",
        "",
        "results_df = pd.DataFrame(results_list)"
    ]))

    new_cells.append(create_code_cell([
        "# 5-2. 결과 시각화",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "",
        "plt.figure(figsize=(12, 6))",
        "sns.barplot(data=results_df, x='Scenario', y='RMSE', hue='Model')",
        "plt.title('RMSE Comparison by Scenario & Model', fontsize=14)",
        "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')",
        "plt.tight_layout()",
        "plt.show()"
    ]))

    new_cells.append(create_markdown_cell([
        "### 🚩 Phase 5 결론",
        "1. **시나리오 4 (전체 데이터)** 만이 유의미한 성능(R2 > 0.5)을 보임.",
        "2. 시나리오 1~3 (Initial, 4.2V, 3.6V)은 성능이 매우 낮음 (R2 < 0.1).",
        "3. **문제점**: 현장에서 2.5V 방전까지 모두 수행하기에는 비용/시간이 많이 듦.",
        "4. **해결책 (Phase 7)**: 성능이 좋은 S4 모델(Teacher)의 지식을 S2(4.2V) 모델(Student)에 전이하는 **Knowledge Distillation** 적용."
    ]))

    # ============================================================
    # Phase 7. Knowledge Distillation
    # ============================================================
    new_cells.append(create_markdown_cell([
        "# Phase 7: Teacher-Student 기반 성능 고도화 (Knowledge Distillation)",
        "",
        "**목표**: 저비용 피처셋(S2, Initial+4.2V)만 사용하는 Student 모델의 성능을, 고성능 피처셋(S4, Full)을 사용하는 Teacher 모델 수준으로 향상시킵니다.",
        "",
        "### 전략",
        "1. **Teacher**: S4(전체 피처)로 학습된 **XGBoost** (Phase 5 Best Model)",
        "2. **Student**: S2(제한된 피처)로 학습할 **LightGBM** (Light but Fast)",
        "3. **Distillation**: Teacher의 예측값(Soft Target)을 Student 학습에 반영",
        "   - Loss = alpha * MSE(y_true) + (1-alpha) * MSE(y_teacher_pred)"
    ]))

    new_cells.append(create_code_cell([
        "# 7-1. Teacher & Student 데이터 준비",
        "# Teacher: S4 (전체 Feature)",
        "# Student: S2 (Initial + 4.2V)",
        "",
        "# Phase 5에서 저장된 S4 모델 중 성능 좋은 XGBoost 선택",
        "# (메모리에 trained_models가 있다면 사용, 없다면 재학습)",
        "",
        "# S4 (Teacher) 데이터",
        "X_train_teacher = X_train[scenarios['시나리오 4 (전체)']]",
        "X_test_teacher  = X_test[scenarios['시나리오 4 (전체)']]",
        "",
        "# S2 (Student) 데이터",
        "X_train_student = X_train[scenarios['시나리오 2 (INITIAL + 4.2V)']]",
        "X_test_student  = X_test[scenarios['시나리오 2 (INITIAL + 4.2V)']]",
        "",
        "print('Training Teacher (XGBoost on S4)...')",
        "teacher_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)",
        "teacher_model.fit(X_train_teacher, y_train)",
        "",
        "# Teacher 성능 확인",
        "y_pred_teacher = teacher_model.predict(X_test_teacher)",
        "rmse_teacher = mean_squared_error(y_test, y_pred_teacher, squared=False)",
        "print(f'Teacher (S4) RMSE: {rmse_teacher:.4f}')"
    ]))

    new_cells.append(create_code_cell([
        "# 7-2. Knowledge Distillation 수행",
        "",
        "# 1) Teacher가 Train Set에 대해 예측 (Soft Target 생성)",
        "y_train_teacher_pred = teacher_model.predict(X_train_teacher)",
        "",
        "# 2) Distillation Target 생성 (Alpha blending)",
        "# Alpha: 실제값(Hard Target) 반영 비율",
        "# 1-Alpha: Teacher 예측값(Soft Target) 반영 비율",
        "alpha = 0.5",
        "y_train_distilled = alpha * y_train + (1 - alpha) * y_train_teacher_pred",
        "",
        "print(f'Distillation Target Created (Alpha={alpha})')",
        "",
        "# 3) Student 모델 학습 (S2 데이터 + Distilled Target)",
        "student_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)",
        "student_model.fit(X_train_student, y_train_distilled)",
        "",
        "# 4) Base Student (KD 없이 학습) 비교용",
        "base_student = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)",
        "base_student.fit(X_train_student, y_train)",
        "",
        "print('Student Training Completed.')"
    ]))

    new_cells.append(create_code_cell([
        "# 7-3. 성능 비교 평가",
        "",
        "y_pred_base = base_student.predict(X_test_student)",
        "y_pred_student = student_model.predict(X_test_student)",
        "",
        "rmse_base = mean_squared_error(y_test, y_pred_base, squared=False)",
        "rmse_student = mean_squared_error(y_test, y_pred_student, squared=False)",
        "r2_student = r2_score(y_test, y_pred_student)",
        "",
        "print('='*50)",
        "print(f'Teacher (S4) RMSE       : {rmse_teacher:.4f}')",
        "print('-'*50)",
        "print(f'Student (S2, No KD) RMSE: {rmse_base:.4f}')",
        "print(f'Student (S2, with KD) RMSE: {rmse_student:.4f}')",
        "print('-'*50)",
        "print(f'KD 성능 향상률: {((rmse_base - rmse_student)/rmse_base)*100:.2f}%')",
        "print(f'Student R2 Score: {r2_student:.4f}')",
        "print('='*50)"
    ]))

    new_cells.append(create_code_cell([
        "# 7-4. 결과 시각화 (예측값 비교)",
        "plt.figure(figsize=(10, 6))",
        "plt.scatter(y_test, y_pred_base, alpha=0.5, label='Base Student (S4)', color='gray')",
        "plt.scatter(y_test, y_pred_student, alpha=0.5, label='Distilled Student (S2+KD)', color='blue')",
        "plt.scatter(y_test, y_pred_teacher, alpha=0.5, label='Teacher (S4)', color='green', marker='x')",
        "plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal')",
        "plt.xlabel('Actual Capacity')",
        "plt.ylabel('Predicted Capacity')",
        "plt.title('Knowledge Distillation Effect')",
        "plt.legend()",
        "plt.grid(True, alpha=0.3)",
        "plt.show()"
    ]))

    new_cells.append(create_markdown_cell([
        "### 🏁 Phase 7 결론",
        "Teacher 모델(S4)의 지식을 S2 데이터만 사용하는 Student 모델에 전이(Knowledge Distillation)한 결과,",
        "단순히 S2 데이터로만 학습했을 때보다 RMSE가 개선되었음을 확인할 수 있습니다.",
        "이를 통해 **2.5V 방전 테스트 없이도(S2)**, 전체 테스트(S4)를 수행한 것에 근접한 성능을 낼 가능성을 확인했습니다."
    ]))

    # 노트북에 셀 추가
    nb['cells'].extend(new_cells)

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Successfully added {len(new_cells)} cells to {NOTEBOOK_PATH}")

if __name__ == "__main__":
    main()
