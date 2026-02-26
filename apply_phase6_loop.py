import json

notebook_path = r'c:\6.1 밧데리_학습\train.ipynb'
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Identify the start of Phase 6 Python code cells
    phase_6_start_idx = -1
    for i, cell in enumerate(nb.get('cells', [])):
        source = "".join(cell.get('source', []))
        if "PyCaret Setup" in source and "setup(data=" in source:
            phase_6_start_idx = i
            break
            
    if phase_6_start_idx == -1:
        print("Could not find PyCaret setup cell.")
    else:
        # We will replace the Phase 6 cells with a new comprehensive cell
        # Let's find how many cells relate to Phase 6 (up to Phase 7 or end)
        phase_6_end_idx = phase_6_start_idx
        for i in range(phase_6_start_idx, len(nb['cells'])):
            source = "".join(nb['cells'][i].get('source', []))
            if "Phase 7" in source or "Phase 8" in source:
                break
            phase_6_end_idx = i
            
        new_source = [
            "# ============================================================\n",
            "# Phase 6. 시나리오별 AutoML 적용 (PyCaret)\n",
            "# ============================================================\n",
            "from pycaret.regression import setup, compare_models, pull, tune_model\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n\n",
            "print('Phase 6: 시나리오(S1~S4)별 PyCaret AutoML을 수행합니다.')\n",
            "print('============================================================')\n\n",
            "# 결과를 저장할 딕셔너리\n",
            "automl_results = {}\n\n",
            "for name, feats in scenarios.items():\n",
            "    print(f'\\n▶️ 진행 중인 시나리오: {name}')\n",
            "    print(f'   - 사용 변수 개수: {len(feats)}개')\n",
            "    \n",
            "    # 시나리오별 데이터셋 구성 (df_clean 활용)\n",
            "    # TARGET은 앞서 정의된 'capacity'\n",
            "    df_scenario = df_clean[feats + [TARGET]]\n",
            "    \n",
            "    # PyCaret Setup\n",
            "    # 환경 초기화, session_id 고정, normalize=True로 스케일링\n",
            "    setup(data=df_scenario, target=TARGET, session_id=42, fold=5, normalize=True, verbose=False)\n",
            "    \n",
            "    # 모델 비교 (R2 기준 상위 3개 채택)\n",
            "    best_models = compare_models(sort='R2', n_select=3, verbose=False)\n",
            "    \n",
            "    # Compare Models 의 결과 성능표 가져오기\n",
            "    leaderboard = pull()\n",
            "    \n",
            "    # 상위 1위 모델 식별\n",
            "    top_model = best_models[0]\n",
            "    top_model_name = leaderboard.index[0]\n",
            "    top_r2 = leaderboard.iloc[0]['R2']\n",
            "    top_rmse = leaderboard.iloc[0]['RMSE']\n",
            "    \n",
            "    print(f'   - 1위 모델: {top_model_name} (R2: {top_r2:.4f}, RMSE: {top_rmse:.4f})')\n",
            "    \n",
            "    # 결과 저장\n",
            "    automl_results[name] = {\n",
            "        'Top_Model': top_model_name,\n",
            "        'R2': top_r2,\n",
            "        'RMSE': top_rmse,\n",
            "        'MAE': leaderboard.iloc[0]['MAE'],\n",
            "        'Object': top_model\n",
            "    }\n\n",
            "print('\\n\\n============================================================')\n",
            "print('시나리오별 AutoML 성능 비교 점수 요약')\n",
            "print('============================================================')\n",
            "results_df = pd.DataFrame(automl_results).T\n",
            "display(results_df[['Top_Model', 'R2', 'RMSE', 'MAE']])\n\n",
            "# 성능 시각화 (R2 기준)\n",
            "plt.figure(figsize=(10, 6))\n",
            "sns.barplot(x=results_df.index, y=results_df['R2'], palette='viridis')\n",
            "plt.title('시나리오별 AutoML 1위 모델의 R2 스코어 비교', fontsize=14)\n",
            "plt.ylabel('R2 Score (Higher is Better)', fontsize=12)\n",
            "plt.xlabel('Scenario', fontsize=12)\n",
            "plt.ylim(0, 1.0)\n",
            "plt.xticks(rotation=15)\n",
            "for i, v in enumerate(results_df['R2']):\n",
            "    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
        
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": new_source
        }
        
        # We replace the chunks
        nb['cells'] = nb['cells'][:phase_6_start_idx] + [new_cell] + nb['cells'][phase_6_end_idx+1:]
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1, separators=(',', ': '))
            
        print(f"REPLACED_SUCCESSFULLY: Replaced {phase_6_end_idx - phase_6_start_idx + 1} Phase 6 cells with the scenario-based AutoML loop.")

except Exception as e:
    print(f"Error: {e}")
