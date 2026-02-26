import json

notebook_path = r'c:\6.1 밧데리_학습\train.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Identify the start and end of Phase 6 Python code cells
    phase_6_start_idx = -1
    for i, cell in enumerate(nb.get('cells', [])):
        source = "".join(cell.get('source', []))
        if "Phase 6" in source and ("AutoML 적용" in source or "PyCaret" in source or "FLAML" in source or "AutoGluon" in source):
            phase_6_start_idx = i
            break
            
    if phase_6_start_idx == -1:
        print("Could not find Phase 6 setup cell. Please check the notebook.")
    else:
        phase_6_end_idx = phase_6_start_idx
        for i in range(phase_6_start_idx, len(nb['cells'])):
            source = "".join(nb['cells'][i].get('source', []))
            if "Phase 7" in source or "Phase 8" in source:
                break
            phase_6_end_idx = i

        new_cells = [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 패키지 설치 (환경 구축 안 된 라이브러리 추가)\n",
                    "!pip install catboost flaml[automl] autogluon.tabular\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ============================================================\n",
                    "# Phase 6. 시나리오별 다중 모델 벤치마킹 (CatBoost/SVR, FLAML, AutoGluon)\n",
                    "# ============================================================\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import time\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error\n\n",
                    "from catboost import CatBoostRegressor\n",
                    "from sklearn.svm import SVR\n",
                    "from flaml import AutoML\n",
                    "from autogluon.tabular import TabularPredictor\n\n",
                    "print('Phase 6: 시나리오(S1~S4)별 다중 벤치마킹을 수행합니다.')\n",
                    "print('비교 트랙: 1) CatBoost, 2) SVR, 3) FLAML, 4) AutoGluon')\n",
                    "print('============================================================')\n\n",
                    "# 전체 결과를 저장할 리스트\n",
                    "benchmark_results = []\n\n",
                    "for name, feats in scenarios.items():\n",
                    "    print(f'\\n▶️ 진행 중인 시나리오: {name}')\n",
                    "    \n",
                    "    # 데이터셋 준비 (Phase 4의 scenario_datasets 구조 활용)\n",
                    "    # 만약 메모리 절약을 위해 df_clean에서 직접 분리한다면:\n",
                    "    X_train_s = scenario_datasets[name]['X_train']\n",
                    "    X_test_s = scenario_datasets[name]['X_test']\n",
                    "    y_train_s = scenario_datasets[name]['y_train']\n",
                    "    y_test_s = scenario_datasets[name]['y_test']\n",
                    "    \n",
                    "    # === Track 1: CatBoost ===\n",
                    "    print('  - 학습 중: Track 1. CatBoost (Battery Domain 최적 트리 앙상블)')\n",
                    "    cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_seed=42)\n",
                    "    cb.fit(X_train_s, y_train_s)\n",
                    "    pred_cb = cb.predict(X_test_s)\n",
                    "    benchmark_results.append({\n",
                    "        'Scenario': name, 'Model_Track': '1_CatBoost', \n",
                    "        'R2': r2_score(y_test_s, pred_cb), 'RMSE': np.sqrt(mean_squared_error(y_test_s, pred_cb)),\n",
                    "        'MAE': mean_absolute_error(y_test_s, pred_cb)\n",
                    "    })\n",
                    "    \n",
                    "    # === Track 2: SVR ===\n",
                    "    print('  - 학습 중: Track 2. Support Vector Regressor (비선형 커널)')\n",
                    "    # SVR은 스케일링이 민감하므로 기존 Phase4의 Robust 스케일러 적용 고려 (여기선 파이프라인 무시하고 직접 학습)\n",
                    "    from sklearn.preprocessing import RobustScaler\n",
                    "    scaler_svr = RobustScaler()\n",
                    "    X_tr_sc = scaler_svr.fit_transform(X_train_s)\n",
                    "    X_te_sc = scaler_svr.transform(X_test_s)\n",
                    "    svr = SVR(C=1.0, epsilon=0.1, kernel='rbf')\n",
                    "    svr.fit(X_tr_sc, y_train_s)\n",
                    "    pred_svr = svr.predict(X_te_sc)\n",
                    "    benchmark_results.append({\n",
                    "        'Scenario': name, 'Model_Track': '2_SVR', \n",
                    "        'R2': r2_score(y_test_s, pred_svr), 'RMSE': np.sqrt(mean_squared_error(y_test_s, pred_svr)),\n",
                    "        'MAE': mean_absolute_error(y_test_s, pred_svr)\n",
                    "    })\n",
                    "    \n",
                    "    # === Track 3: FLAML AutoML ===\n",
                    "    print('  - 학습 중: Track 3. Microsoft FLAML (Fast AutoML, 30sec budget)')\n",
                    "    flaml_automl = AutoML()\n",
                    "    # 예산 30초, Lgbm, rf, xgboost 등 자동 탐색\n",
                    "    flaml_settings = {\n",
                    "        \"time_budget\": 30,  # 30초\n",
                    "        \"metric\": 'r2',\n",
                    "        \"task\": 'regression',\n",
                    "        \"eval_method\": 'cv',\n",
                    "        \"log_file_name\": f\"flaml_{name}.log\",\n",
                    "        \"verbose\": 0\n",
                    "    }\n",
                    "    flaml_automl.fit(X_train=X_train_s, y_train=y_train_s, **flaml_settings)\n",
                    "    pred_flaml = flaml_automl.predict(X_test_s)\n",
                    "    benchmark_results.append({\n",
                    "        'Scenario': name, 'Model_Track': f'3_FLAML ({flaml_automl.best_estimator})', \n",
                    "        'R2': r2_score(y_test_s, pred_flaml), 'RMSE': np.sqrt(mean_squared_error(y_test_s, pred_flaml)),\n",
                    "        'MAE': mean_absolute_error(y_test_s, pred_flaml)\n",
                    "    })\n",
                    "    \n",
                    "    # === Track 4: AutoGluon ===\n",
                    "    print('  - 학습 중: Track 4. AWS AutoGluon (Ensemble Tabular)')\n",
                    "    # AutoGluon은 Pandas DataFrame 전체를 입력받고 label 컬럼을 명시해야 함.\n",
                    "    train_data_ag = pd.concat([X_train_s, y_train_s], axis=1)\n",
                    "    test_data_ag = pd.concat([X_test_s, y_test_s], axis=1)\n",
                    "    # presets='medium_quality' or 'best_quality' (여기선 속도 타협)\n",
                    "    predictor_ag = TabularPredictor(label=TARGET, path=f'AutogluonModels_{name}', eval_metric='r2').fit(\n",
                    "        train_data_ag, \n",
                    "        presets='medium_quality', \n",
                    "        time_limit=60, \n",
                    "        verbosity=0\n",
                    "    )\n",
                    "    pred_ag = predictor_ag.predict(test_data_ag.drop(columns=[TARGET]))\n",
                    "    benchmark_results.append({\n",
                    "        'Scenario': name, 'Model_Track': '4_AutoGluon (Ensemble)', \n",
                    "        'R2': r2_score(y_test_s, pred_ag), 'RMSE': np.sqrt(mean_squared_error(y_test_s, pred_ag)),\n",
                    "        'MAE': mean_absolute_error(y_test_s, pred_ag)\n",
                    "    })\n",
                    "    print('  ✓ 완료')\n\n",
                    "print('\\n\\n============================================================')\n",
                    "print('시나리오 & 모델 트랙별 벤치마킹 성능 최종 프레임')\n",
                    "print('============================================================')\n",
                    "bench_df = pd.DataFrame(benchmark_results)\n",
                    "display(bench_df.sort_values(by=['Scenario', 'R2'], ascending=[True, False]))\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 성능 시각화 (R2 기준 그룹별 막대 그래프)\n",
                    "plt.figure(figsize=(14, 7))\n",
                    "sns.barplot(data=bench_df, x='Scenario', y='R2', hue='Model_Track', palette='Set2')\n",
                    "plt.title('시나리오 및 적용 프레임워크별 R2 스코어 (높을수록 우수)', fontsize=15, fontweight='bold')\n",
                    "plt.ylabel('R2 Score', fontsize=12)\n",
                    "plt.xlabel('Scenario', fontsize=12)\n",
                    "plt.ylim(0, 1.05)\n",
                    "plt.legend(title='Model Track', loc='lower right')\n",
                    "plt.grid(axis='y', linestyle='--', alpha=0.6)\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n"
                ]
            }
        ]
        
        # We replace the chunks
        nb['cells'] = nb['cells'][:phase_6_start_idx] + new_cells + nb['cells'][phase_6_end_idx+1:]
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1, separators=(',', ': '))
            
        print(f"REPLACED_SUCCESSFULLY: Replaced old Phase 6 Pycaret cells with New Multi-Benchmark cells.")

except Exception as e:
    print(f"Error: {e}")
