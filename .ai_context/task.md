# 배터리 Capacity 예측 프로젝트 — 작업 내역

## 완료된 작업

### Phase 1~8: 기본 ML 파이프라인 (파생변수 포함)
- [x] Phase 1: EDA (1,040셀, 8피처, 결측치/이상치/상관분석)
- [x] Phase 2: 파생변수 16개 생성 (임피던스/전압/복합)
- [x] Phase 3: 이상치 제거(1개), VIF 분석, 시나리오 S1~S4 확정
- [x] Phase 4: Train/Test 분할 (831/208, KS 검정 통과)
- [x] Phase 5: 기본 모델 벤치마크 (LR/SVR/RF/XGBoost/LightGBM)
- [x] Phase 6: Multi-AutoML (CatBoost/SVR/FLAML/AutoGluon) × S1~S4
- [x] Phase 7: 최고 모델 SHAP/Feature Importance 분석
- [x] Phase 8: 최종 테스트 성능 검증 및 시각화

### Phase 9: 파생변수 제외 최적 측정 조합 탐색
- [x] `train_no_derived.py`: 파생변수 없이 S1~S4 학습 파이프라인 구현 및 실행
- [x] 노트북에 Phase 9 코드 추가 (9-1 ~ 9-6)
  - 9-1: 13개 측정 조합 정의 (P1:S1+단일, P2:단일만, P3:전압/임피던스 분리, REF)
  - 9-2: 기본 모델 6종 × 13조합 = 78 실험
  - 9-3: CatBoost + FLAML × 13조합 AutoML 벤치마킹
  - 9-4: 종합 결과 분석 및 측정 공정 효율성 평가
  - 9-5: 상위 조합 FLAML 강화 학습 (30s budget)
  - 9-6: 시각화 및 최종 결론
- [x] `gen_phase9_data.py`: Phase 9 결과를 CSV로 미리 생성하는 스크립트
- [x] `phase9_results.csv` / `phase9_corr.csv`: 83행 실험 결과 데이터

### Streamlit 보고서
- [x] `app_phase9_report.py`: Phase 9 리포트 (CSV 기반 즉시 로딩, 실행 버튼 없음)
  - 9-1: 조합 정의 + 상관분석 차트
  - 9-2/9-3: 전체 결과 테이블 / RMSE 히트맵 / 모델별 R² 막대 (탭)
  - 9-4: 조합별 최고 성능 + Top10 + R² 수평 막대 + 효율성 프론티어
  - 9-5: FLAML 강화 결과 + 기본모델 대비 비교
  - 9-6: 결론 카드 + 데이터 조합 설명 + 핵심 인사이트
- [x] `app_phase9_summary.py`: 초기 요약형 앱 (이후 report로 대체)
- [x] `app.py`: 기존 종합 보고서 앱 (변경 없음)

### 인프라
- [x] Git 커밋 & 원격 push
- [x] `.gitignore` 업데이트 (AutogluonModels, catboost_info, logs, temp 제외)
- [x] `.ai_context/` 작업 내역 업데이트

## 핵심 결과
- **최고 성능**: REF_S4_ALL + FLAML(extra_tree) — 파생변수 없이도 최고 정확도
- **최적 효율**: P1_S1+v25 (초기+완전방전 2단계) — 피처 절반으로 유사 성능
- **최소 측정**: P2_v25_only (완전방전만 1단계) — 실용적 빠른 스크리닝
- **핵심 인사이트**: v25(완전방전) 측정이 capacity 예측의 핵심
