import json

notebook_path = r'c:\6.1 밧데리_학습\train.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # 1. Find Phase 7 start
    phase_7_start = -1
    for i, cell in enumerate(nb.get('cells', [])):
        source = "".join(cell.get('source', []))
        if "Phase 7" in source and "모델 해석" in source:
            phase_7_start = i
            break
            
    if phase_7_start == -1:
        # If not found, find where to insert (after Phase 6)
        # We know Phase 6 bench_df code was added. Let's find end of Phase 6.
        for i, cell in enumerate(nb.get('cells', [])):
            source = "".join(cell.get('source', []))
            if "시나리오 및 적용 프레임워크별 R2 스코어" in source: # End of phase 6
                phase_7_start = i + 1
                break

    # 2. Add Phase 7 Cells
    if phase_7_start != -1:
        # Delete old Phase 7 cells until Phase 8
        phase_7_end = phase_7_start
        for i in range(phase_7_start, len(nb['cells'])):
            source = "".join(nb['cells'][i].get('source', []))
            if "Phase 8" in source or "결론" in source:
                break
            phase_7_end = i
            
        new_cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Phase 7. 최고 성능 모델 해석 (Feature Importance & SHAP)\n",
                    "\n",
                    "앞선 Phase 6의 벤치마킹 결과(`bench_df`)를 바탕으로, 가장 R2 성능이 높았던 **최고 성능 모델(Best Model)**과 **해당 시나리오 데이터셋**을 자동으로 식별하여 변수 중요도 및 모델 해석(SHAP 등)을 수행합니다.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. 벤치마크 결과 기준 1위 모델 식별\n",
                    "best_row = bench_df.sort_values(by='R2', ascending=False).iloc[0]\n",
                    "best_scenario_name = best_row['Scenario']\n",
                    "best_track_name = best_row['Model_Track']\n",
                    "best_r2 = best_row['R2']\n\n",
                    "print('============================================================')\n",
                    "print(f'🏆 벤치마크 1위 최고 성능 모델 식별 🏆')\n",
                    "print(f'   - 시나리오 데이터: {best_scenario_name}')\n",
                    "    - 프레임워크 트랙: {best_track_name}')\n",
                    "print(f'   - Test R2 Score: {best_r2:.4f}')\n",
                    "print('============================================================')\n\n",
                    "# 데이터 준비\n",
                    "X_train_best = scenario_datasets[best_scenario_name]['X_train']\n",
                    "X_test_best = scenario_datasets[best_scenario_name]['X_test']\n",
                    "y_train_best = scenario_datasets[best_scenario_name]['y_train']\n",
                    "y_test_best = scenario_datasets[best_scenario_name]['y_test']\n",
                    "best_feature_names = scenario_datasets[best_scenario_name]['features']\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. 모델 재지정 및 모델 해석(Feature Importance / SHAP) 수행\n",
                    "import shap\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n\n",
                    "plt.figure(figsize=(10, 6))\n\n",
                    "if 'CatBoost' in best_track_name:\n",
                    "    print('▶️ CatBoost 모델 SHAP 분석 (TreeExplainer) 진행')\n",
                    "    best_model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_seed=42).fit(X_train_best, y_train_best)\n",
                    "    explainer = shap.TreeExplainer(best_model)\n",
                    "    shap_values = explainer.shap_values(X_test_best)\n",
                    "    shap.summary_plot(shap_values, X_test_best, feature_names=best_feature_names)\n\n",
                    "elif 'SVR' in best_track_name:\n",
                    "    print('▶️ SVR 모델 SHAP 분석 (KernelExplainer) 진행 - 샘플링 사용')\n",
                    "    from sklearn.preprocessing import RobustScaler\n",
                    "    scaler_svr = RobustScaler()\n",
                    "    X_tr_sc = scaler_svr.fit_transform(X_train_best)\n",
                    "    X_te_sc = scaler_svr.transform(X_test_best)\n",
                    "    best_model = SVR(C=1.0, epsilon=0.1, kernel='rbf').fit(X_tr_sc, y_train_best)\n",
                    "    # 시간 관계상 K-means로 기준 샘플 요약\n",
                    "    X_train_summary = shap.kmeans(X_tr_sc, 20)\n",
                    "    explainer = shap.KernelExplainer(best_model.predict, X_train_summary)\n",
                    "    shap_values = explainer.shap_values(X_te_sc[:100]) # 처음 100개만 빠른 확인\n",
                    "    shap.summary_plot(shap_values, pd.DataFrame(X_te_sc[:100], columns=best_feature_names), feature_names=best_feature_names)\n\n",
                    "elif 'FLAML' in best_track_name:\n",
                    "    print('▶️ FLAML 최고 도출 모델 SHAP 분석 (TreeExplainer) 진행')\n",
                    "    # 다시 훈련시키지 않고 (아까 튜닝이 오래 걸릴 수 있으므로, FLAML 결과 재현)\n",
                    "    # 여기선 빠른 코드상 재학습으로 흉내. 실제론 상단 객체를 가져오는게 좋으나 안정성을 위해 재학습\n",
                    "    flaml_automl = AutoML()\n",
                    "    flaml_automl.fit(X_train=X_train_best, y_train=y_train_best, time_budget=30, metric='r2', task='regression', verbose=0)\n",
                    "    best_model = flaml_automl.model.estimator\n",
                    "    try:\n",
                    "        explainer = shap.TreeExplainer(best_model)\n",
                    "        shap_values = explainer.shap_values(X_test_best)\n",
                    "        shap.summary_plot(shap_values, X_test_best, feature_names=best_feature_names)\n",
                    "    except:\n",
                    "        print('FLAML 트리 모델 외의 경우입니다. 변수 중요도(Feature Importances)만 출력합니다.')\n",
                    "        pd.Series(best_model.feature_importances_, index=best_feature_names).sort_values().plot.barh(color='skyblue')\n",
                    "        plt.title('FLAML Feature Importances')\n",
                    "        plt.show()\n\n",
                    "elif 'AutoGluon' in best_track_name:\n",
                    "    print('▶️ AutoGluon 앙상블 프레임워크 변수 중요도 추출 (Permutation 기반) 진행')\n",
                    "    predictor_path = f'AutogluonModels_{best_scenario_name}'\n",
                    "    predictor_best = TabularPredictor.load(predictor_path)\n",
                    "    \n",
                    "    train_data_ag = pd.concat([X_train_best, y_train_best], axis=1)\n",
                    "    # AutoGluon자체 기능으로 중요도 산출\n",
                    "    feature_importance_df = predictor_best.feature_importance(train_data_ag)\n",
                    "    \n",
                    "    feature_importance_df['importance'].sort_values().plot.barh(color='coral')\n",
                    "    plt.title('AutoGluon Feature Importances (Permutation)')\n",
                    "    plt.xlabel('Importance')\n",
                    "    plt.ylabel('Features')\n",
                    "    plt.show()\n\n",
                    "else:\n",
                    "    print('지원하지 않는 모델 계열입니다.')\n"
                ]
            }
        ]
        
        # 교체 작업
        nb['cells'] = nb['cells'][:phase_7_start] + new_cells + nb['cells'][phase_7_end+1:]
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1, separators=(',', ': '))
            
        print("REPLACED_SUCCESSFULLY: Replaced Phase 7 cells with dynamic BEST MODEL interpretation.")

except Exception as e:
    print(f"Error: {e}")
