# Phase 7: Teacher-Student 기반 성능 고도화 (Knowledge Distillation)

## [Goal Description]
사용자 제안에 따라 **Teacher-Student 학습 (Knowledge Distillation)** 기법을 도입하여, **모든 데이터(S4)**를 학습한 **Teacher 모델**의 지식을 **일부 데이터(S1~S3)**만 사용하는 **Student 모델**에게 전달합니다. 이를 통해 저비용 시나리오(S1~S3)의 예측 성능을 극대화하는 것이 목표입니다.

## User Review Required
> [!NOTE]
> **Teacher-Student 전략 핵심**
> - **Teacher Model**: S4 데이터(2.5V 방전 포함)로 학습된 고성능 모델 (XGBoost).
> - **Student Model**: S1/S2/S3 데이터(제한된 정보)만 사용하는 경량 모델 (LightGBM/DNN).
> - **Learning Strategy**: Student는 정답($y_{true}$)뿐만 아니라 Teacher의 예측($\hat{y}_{teacher}$)도 함께 학습하여 일반화 성능을 높임.

## Proposed Changes

### [New Script] `run_phase7.py`
다음 단계로 구성된 지식 증류 스크립트를 작성합니다.

1.  **Teacher 모델 학습 (S4) - 최적화 (Optuna 활용)**:
    - **도구**: **Optuna** (하이퍼파라미터 최적화 프레임워크) 도입.
    - **목표**: XGBoost/LightGBM의 최적 하이퍼파라미터(`learning_rate`, `depth`, `subsample`, `lambda` 등)를 자동으로 탐색하여 **최고 성능(R²)**을 달성하는 Teacher 모델 확보.
    - **배포 관점**: **Teacher 모델은 배포되지 않고 학습 단계에서만 사용**되므로, 모델이 복잡하거나 라이브러리 의존성이 있어도 최종 서비스 배포에는 문제가 없습니다.
2.  **Soft Target 생성**:
    - Optuna로 찾은 Best Teacher 모델로 훈련 데이터에 대한 예측값($\hat{y}_{teacher}$) 생성.
    - 이 "고품질의 지식"을 Student에게 전수.
3.  **Student 모델 학습 (S1, S2, S3)**:
    - **입력**: 시나리오별 제한된 피처 (S1: 3개, S2: 9개 등).
    - **타겟**:
        - **Hard Target**: 실제 Capacity ($y_{true}$)
        - **Soft Target**: Teacher의 예측값 ($\hat{y}_{teacher}$)
    - **손실 함수 (Loss Function)**:
        - $Loss = \alpha \cdot MSE(y_{pred}, y_{true}) + (1-\alpha) \cdot MSE(y_{pred}, \hat{y}_{teacher})$
        - $\alpha$: Hard Target과 Soft Target의 반영 비율 (예: 0.5).
4.  **검증 및 비교**:
    - Student 모델을 Test Set(S1~S3)으로 평가.
    - 기존 Phase 5의 단독 학습 모델(Baseline)과 성능(R²) 비교.

## Verification Plan

### Automated Tests
1.  `run_phase7.py` 실행.
2.  `phase7_distillation_results.txt` 확인.
3.  **성능 목표**: S2, S3 시나리오의 Student 모델 R²가 Baseline보다 향상되었는지 확인.

### Manual Verification
- `phase7_teacher_student_compare.png`: Teacher와 Student의 예측 분포 비교 시각화 확인.
