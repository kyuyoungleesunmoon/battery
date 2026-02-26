# 배터리 Capacity 예측 — 구현 계획 및 아키텍처

## 프로젝트 구조

```
c:\2.배터리\
├── data.csv                          # 원본 데이터 (1,040셀 × 15컬럼)
├── battery_capacity_prediction.ipynb  # 메인 분석 노트북 (Phase 1~9, 10 초안)
├── train_no_derived.py               # 파생변수 없이 S1~S4 학습 스크립트
├── gen_phase9_data.py                # Phase 9 결과 CSV 생성 스크립트
├── phase9_results.csv                # Phase 9 실험 결과 (83행)
├── phase9_corr.csv                   # 피처-Capacity 상관계수 (8행)
├── app.py                           # Streamlit 종합 보고서 (Phase 1~8)
├── app_phase9_report.py              # Streamlit Phase 9 리포트 (CSV 즉시 로딩)
├── app_phase9_summary.py             # Streamlit Phase 9 요약 (초기 버전)
├── .gitignore
├── requirements.txt
└── .ai_context/
    ├── task.md                       # 작업 완료 내역
    └── implementation_plan.md        # 이 파일
```

## 기술 스택
- **환경**: conda `battery`, Python 3.11.14, Windows
- **ML**: scikit-learn, xgboost, lightgbm, catboost, flaml, autogluon
- **시각화**: matplotlib, seaborn
- **보고서**: Streamlit
- **데이터**: pandas, numpy

## 데이터 파이프라인
1. `data.csv` → 컬럼 재정의 → 라벨/빈 컬럼 제거 → 수치형 변환
2. 이상치 제거: `|v36_impedance - initial_impedance| > 100` (1개)
3. Train/Test 분할: 80:20, random_state=42

## 시나리오 설계

### Phase 1~8 (파생변수 포함)
| 시나리오 | 피처 | 설명 |
|---------|------|------|
| S1_INITIAL | 3개 | 초기 OCV/임피던스 + ocv_deviation |
| S2_INITIAL_42V | 9개 | S1 + 4.2V 측정 + 파생비율 |
| S3_INITIAL_42V_36V | 12개 | S2 + 3.6V 측정 |
| S4_ALL | 17개 | 전체 원본 + 선별 파생변수 |

### Phase 9 (파생변수 제외)
| 조합 | 피처 수 | 측정 공정 |
|------|--------|----------|
| P1_S1+v42 | 4 | 2단계(초기+풀충전) |
| P1_S1+v25 | 4 | 2단계(초기+완전방전) |
| P1_S1+v36 | 4 | 2단계(초기+방전중간) |
| P2_v42_only | 2 | 1단계(풀충전만) |
| P2_v25_only | 2 | 1단계(완전방전만) |
| P2_v36_only | 2 | 1단계(방전중간만) |
| P3_all_voltage | 4 | 4단계(전압만) |
| P3_all_impedance | 4 | 4단계(임피던스만) |
| P3_S1v25_volt | 2 | 2단계(전압만) |
| P3_S1v25_imp | 2 | 2단계(임피던스만) |
| P3_v25_volt_1 | 1 | 1단계(v25전압1개) |
| REF_S1_only | 2 | 1단계(초기만) |
| REF_S4_ALL | 8 | 4단계(전체) |

## Streamlit 보고서 실행
```bash
# Phase 9 리포트 (CSV 기반 즉시 로딩)
streamlit run app_phase9_report.py --server.port 8504

# 종합 보고서
streamlit run app.py --server.port 8501
```

## 핵심 발견
- v25(완전방전) 측정이 capacity 예측의 핵심
- 파생변수 없이도 FLAML extra_tree가 최고 성능 달성
- 트리 기반 모델(CatBoost/XGBoost)이 배터리 데이터에 가장 적합
- 전압이 임피던스보다 예측력이 높음
