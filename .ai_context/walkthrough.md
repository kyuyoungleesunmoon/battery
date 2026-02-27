# Phase 10 및 Phase 11 시각화 에러 연쇄 픽스 리포트

## 1차 에러: KeyError: 'ocv_deviation' (Phase 10)
`battery_capacity_prediction.ipynb` 파일의 `Phase 10` (현재 10-3-1, 구 Cell 84) 코드 실행 중 파생 변수인 `ocv_deviation`이 `data`에서 유실되어 발생한 에러입니다.

**해결 방안:**
- 단절된 데이터프레임 구조를 복원하기 위해 원본 `data_df`에서 값을 가져오거나, 수식으로 즉석 재계산하여 빈 공간을 메우는 안전장치(Fallback) 로직을 추가했습니다.

## 2차 에러: KeyError: 'Capacity' (Phase 10)
위 방어 로직을 적용한 후 산점도(regplot) 및 다음 셀의 박스플롯(boxplot)에서 연달아 발생한 에러입니다.
- **원인:** 종속(타겟) 변수 컬럼명이 이전 과정에서 소문자 `'capacity'`(혹은 'capacity [Ah]')로 전처리되었으나, 시각화 코드들에서는 하드코딩으로 알파벳 대문자인 `Capacity` 컬럼을 맵핑하려 시도했습니다.
- **해결 방안:** 노트북 전체를 스캔하여 오타를 일괄 치환(Patch)하였고, DataFrame에 존재하는 실제 타겟 컬럼 이름을 동적으로 런타임에 찾아내는 로직을 편입시켰습니다.

## 3차 에러: KeyError: 'Cluster_Pred' (Phase 11-4)
- **원인:** Phase 11-3(예측 기반 클러스터링) 단계에서 K-Means 예측 결과인 `Cluster_Pred` 변수가 백업본인 `data_df` 테이블에만 저장되고, 실제로 모델 검증 및 시각화에 쓰이는 `data` 변수로는 전달(동기화)되지 않은 상태로 Phase 11-4 (ARI 스코어 단계)를 호출하여 발생한 구문 논리 에러입니다.
- **해결 방안:** ARI 스코어 측정 셀(11-4) 및 최종 PCA 스캐터 플롯 차트 셀 상단에, `data` 객체의 컬럼 유실 여부를 검사하고 즉시 `data_df` 로부터 누락된 `Cluster_GT`와 `Cluster_Pred` 변수를 끌어와 복구하는(Synchronization) 데이터 방어 코드를 삽입했습니다.
