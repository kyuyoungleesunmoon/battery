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

    # 1. Remove Any existing Optuna tuning cells if they are in the wrong place
    # We marks cells by content to be safe.
    new_cells_list = []
    for cell in nb['cells']:
        source_text = "".join(cell['source'])
        if "7-1. Teacher Model (XGBoost) Hyperparameter Tuning" in source_text or \
           "import optuna" in source_text or \
           "study.optimize(objective, n_trials=20)" in source_text or \
           "7-2. 최적화된 Teacher 모델 학습 및 Soft Label 생성" in source_text:
            continue
        new_cells_list.append(cell)
    
    nb['cells'] = new_cells_list

    # 2. Find the index of the Phase 7 header again
    phase_7_index = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source_text = "".join(cell['source'])
            if "Phase 7" in source_text and ("Teacher-Student" in source_text or "Knowledge Distillation" in source_text) and "결론" not in source_text:
                phase_7_index = i
                break

    if phase_7_index == -1:
        print("Error: Could not find Phase 7 header. Appending to end.")
        phase_7_index = len(nb['cells']) - 1

    print(f"Injecting Optuna cells after index {phase_7_index}")

    # 3. Define Optuna cells
    optuna_cells = [
        create_markdown_cell([
            "### 7-1. Teacher Model (XGBoost) Hyperparameter Tuning with Optuna",
            "",
            "고성능 Teacher 모델 확보를 위해 Optuna를 사용하여 최적의 하이퍼파라미터를 탐색합니다.",
            "Teacher 모델은 **S4 (전체 데이터)** 시나리오를 사용합니다."
        ]),
        create_code_cell([
            "import optuna",
            "import xgboost as xgb",
            "from sklearn.metrics import mean_squared_error",
            "from sklearn.model_selection import train_test_split",
            "",
            "# Optuna 로깅 레벨 조정",
            "optuna.logging.set_verbosity(optuna.logging.WARNING)",
            "",
            "def objective(trial):",
            "    # S4 데이터셋 사용 (모든 피처)",
            "    X_train_s4 = scenario_datasets['S4_ALL']['X_train']",
            "    y_train_s4 = scenario_datasets['S4_ALL']['y_train']",
            "    ",
            "    # 검증셋 분리",
            "    X_tr, X_val, y_tr, y_val = train_test_split(X_train_s4, y_train_s4, test_size=0.2, random_state=42)",
            "    ",
            "    param = {",
            "        'objective': 'reg:squarederror',",
            "        'verbosity': 0,",
            "        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),",
            "        'max_depth': trial.suggest_int('max_depth', 3, 10),",
            "        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),",
            "        'subsample': trial.suggest_float('subsample', 0.6, 1.0),",
            "        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),",
            "        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),",
            "        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),",
            "        'random_state': 42",
            "    }",
            "    ",
            "    model = xgb.XGBRegressor(**param)",
            "    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)",
            "    ",
            "    preds = model.predict(X_val)",
            "    rmse = np.sqrt(mean_squared_error(y_val, preds))",
            "    return rmse",
            "",
            "print('🚀 Optuna 최적화 시작...')",
            "study = optuna.create_study(direction='minimize')",
            "study.optimize(objective, n_trials=20)",
            "",
            "print('✅ 최적 파라미터:', study.best_params)",
            "print('✅ Best RMSE:', study.best_value)"
        ]),
        create_markdown_cell([
            "### 7-2. 최적화된 Teacher 모델 학습 및 Soft Label 생성",
            "",
            "Optuna로 찾은 최적 파라미터로 Teacher 모델을 다시 학습하고, Soft Target(예측값)을 생성합니다."
        ]),
        create_code_cell([
            "# 최적 파라미터로 Teacher 모델 재학습 (S4 전체 데이터)",
            "best_params = study.best_params",
            "best_params['objective'] = 'reg:squarederror'",
            "best_params['random_state'] = 42",
            "",
            "teacher_model_opt = xgb.XGBRegressor(**best_params)",
            "teacher_model_opt.fit(scenario_datasets['S4_ALL']['X_train'], scenario_datasets['S4_ALL']['y_train'])",
            "",
            "# 전체 Train 데이터에 대한 Teacher 예측값 생성 (Soft Labels)",
            "# Student가 학습할 때 참고할 '정제된 지식'입니다.",
            "y_teacher_pred = teacher_model_opt.predict(X_train[scenario4_features])",
            "",
            "print('✅ 최적화된 Teacher 모델 학습 및 Soft Label 생성 완료')",
            "print(f'Soft Label Sample: {y_teacher_pred[:5]}')"
        ])
    ]

    # 4. Insert cells
    insert_pos = phase_7_index + 1
    for cell in reversed(optuna_cells):
        nb['cells'].insert(insert_pos, cell)

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Successfully placed Optuna cells after index {phase_7_index}")

if __name__ == "__main__":
    main()
