# 배터리 Capacity 예측 모델 구축 - 작업 체크리스트

## Phase 1: 데이터 탐색 및 분석 (EDA)
- [x] 기본 통계 분석 (분포, 결측치, 이상치) → train.ipynb 작성 완료
- [x] 상관분석 (Pearson/Spearman) → train.ipynb 작성 완료
- [x] 분포 시각화 (히스토그램, 박스플롯, KDE) → train.ipynb 작성 완료
- [x] 피처 vs Capacity 산점도 → train.ipynb 작성 완료
- [x] 이상치 탐지 (IQR, Z-score) → train.ipynb 작성 완료
- [x] 각 셀 실행 결과 분석 마크다운 삽입 완료

## Phase 2: 파생변수 생성
- [x] 임피던스 기반 파생변수 8개
- [x] 전압 기반 파생변수 5개
- [x] 복합 파생변수 3개
- [x] 파생변수 vs Capacity 상관분석 및 시각화
- [x] 각 셀 실행 결과 분석 마크다운 6개 삽입 완료

### Phase 2 핵심 발견
- 최강 파생변수: `voltage_sag` (r=+0.600)
- 다중공선성: |r| > 0.9인 쌍 9개

## Phase 3: 데이터 전처리 및 시각화
- [x] 극단 이상치 처리: 1개 제거 (v36_impedance 오류)
- [x] VIF 분석 및 중복 피처 7개 제거
- [x] 유효 파생변수 9개 확정 (총 17개 피처)
- [x] 시나리오별 피처 리스트 확정 (3/9/12/17개)
- [x] 각 셀 분석 마크다운 삽입 완료

## Phase 4: 최종 학습 데이터 구성
- [x] Train/Test 분할 (831:208, 80:20, random_state=42)
- [x] 분포 동질성 검증 (KS test p=0.4561, 통과)
- [x] 시나리오별 데이터셋 준비 완료 (결측치/무한값 없음)
- [x] 스케일링 준비 (Robust=선형용, None=트리용)
- [x] Train/Test 분포 시각화 및 셀 삽입 완료

## Phase 5: 모델 학습 및 평가
- [x] 베이스라인 모델 (Linear Regression) 학습 및 평가
- [x] 앙상블 모델 (Random Forest, XGBoost, LightGBM) 학습
- [x] SVR (Support Vector Regression) 학습
- [x] 시나리오별 성능 비교 (RMSE, MAE, R², MAPE)
- [x] 피처 중요도 분석 (SHAP 대용: Feature Importance)
- [x] 노트북 셀 삽입 완료

## Phase 6: 결과 보고
- [x] 시나리오별 성능 비교 차트 (walkthrough.md 및 노트북에 포함)
- [x] 최적 모델 선정 및 근거 정리 (XGBoost @ S4)
- [x] 최종 결과 보고서 작성 (walkthrough.md)

# ✅ 프로젝트 완료 (1차)
- 최종 `train.ipynb` 구성 완료 (총 셀 92개)
- 결과 보고서 `walkthrough.md` 생성 완료

## Phase 7: 비용 절감 시나리오 성능 고도화 (Knowledge Distillation)
- [x] Teacher 모델(S4 XGBoost) 재학습 및 Soft Label 생성 → train.ipynb (위치 최적화 완료)
- [x] **[Upgrade]** Teacher 모델 하이퍼파라미터 튜닝 (Optuna) → train.ipynb (Phase 7-1에 추가)
- [x] Student 모델(S2 LightGBM) 정의 및 학습 → train.ipynb Phase 7
- [x] Knowledge Distillation 적용 및 성능 비교 (RMSE 개선 확인) → train.ipynb Phase 7
- [x] KD 효과 시각화 (예측값 산점도 등) → train.ipynb Phase 7
