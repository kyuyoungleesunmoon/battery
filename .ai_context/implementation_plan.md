# Phase 7: 최고 성능 모델 대상 해석 (SHAP) 전략 수정 계획

사용자님의 요청에 따라, 단일 모델에 고정되어 있던 기존 Phase 7 (SHAP 모델 해석 등) 코드를 **Phase 5 (이상치 제거 등 기초 모델링) 및 Phase 6 (다중 AutoML 벤치마킹) 중 가장 성능이 좋았던 최고 성능 모델(Best Model)과 그에 맞는 데이터셋(Best Scenario)** 하나만을 선택하여 적용하도록 동적으로 수정하는 계획입니다.

## User Review Required
> [!NOTE]
> Phase 6 코드 이후에, 전체 `benchmark_results` 리스트에서 R2가 가장 높았던 모델 구조체(`model 객체`)와 해당 시나리오 데이터셋(`X_train`, `X_test` 등)을 추출하여 Phase 7 코드의 입력으로 바로 활용할 예정입니다.
> - 다만 **AutoGluon**의 모델은 내부 앙상블 구조로 인해 일반적인 Tree SHAP으로 호환되지 않을 수 있습니다. 따라서 최고 성능 모델이 만약 1) CatBoost, 2) SVR, 3) FLAML(LGBM/XGB 등) 중 하나라면 기존 `shap.TreeExplainer` 또는 `KernelExplainer`를 적용하고, 4) AutoGluon이 1위라면 Feature Importance 함수나 플롯으로 SHAP을 대체하는 분기 처리를 넣겠습니다.

## Proposed Changes

### 노트북 변경 사항 (`train.ipynb`)

1. **최고 성능 모델 자동 추출 코드 삽입 (Phase 6 마지막)**
   - Phase 6가 완료된 직후 산출된 `bench_df` (벤치마킹 데이터프레임)를 R2 스코어 기준으로 내림차순 정렬.
   - 1위(Top-1) 시나리오 이름(`Best_Scenario`)과 프레임워크(`Best_Track`) 식별.
   - 해당 모델 객체(`best_model_obj`)와 훈련/테스트 데이터셋(`X_train_best`, `X_test_best`) 변수 매핑(저장) 코드 추가.

2. **Phase 7: 모델 해석 및 중요 변수 도출 (Best Model 대상) 전면 수정**
   - **기존 PyCaret 의존성 함수 제거**: 기존 노트북에 적혀 있던 `interpret_model()` 등 PyCaret 종속 코드를 삭제합니다.
   - **SHAP (Shapley Additive exPlanations) 동적 파이프라인 구성**:
     - 타겟 최고 모델(`best_model_obj`)이 Tree 계열(LGBM, XGBoost, CatBoost, RandomForest 등)인지 비선형(SVR)인지 파악.
     - Tree 계열 학습기인 경우 ➔ `shap.TreeExplainer` 사용.
     - SVR 등인 경우 ➔ 속도를 위해 `shap.KernelExplainer` 생성 (데이터 일부 샘플링 적용).
     - AutoGluon 등 앙상블 Predictor인 경우 ➔ AutoGluon 내부 내장 함수인 `.feature_importance(dataset)` 활용 플롯 구현.
   - **SHAP Summary Plot 생성**:
     - `shap.summary_plot(shap_values, X_test_best)` 를 호출하여, 실제로 가장 성능이 좋았던 모델 입장에서 배터리 수명(`capacity`)에 어떤 파생 변수가 긍정적/부정적 영향을 크게 미쳤는지 시각화합니다.

## Verification Plan
1. 새로운 파이썬 스크립트를 작성하여 Phase 7 파트 코드를 파싱하고 교체.
2. 덮어씌워진 노트북을 통해, Best Model 변수를 제대로 받아 SHAP 플롯이 크래시 없이 그려지는지 검증.
