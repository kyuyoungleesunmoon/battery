# Teacher 모델 NameError 및 셀 구조 해결 계획

`train.ipynb`에서 Teacher 모델 학습 중 발생하는 `NameError: name 'study' is not defined` 오류를 해결하고, 꼬여 있는 Phase 7 셀 구조를 정리합니다.

## Proposed Changes

### [train.ipynb](file:///c:/6.1%20밧데리_학습/train.ipynb)

- **[MODIFY]** Phase 5 결론 이후에 잘못 삽입된 Phase 7 관련 코드 셀(라인 3560-3582 부근)을 제거합니다.
- **[MODIFY]** 기존에 중복되거나 정의가 빠진 Phase 7 영역(라인 3588-3806 부근)을 아래와 같은 올바른 순서로 재구성합니다:
    1. **Phase 7 헤더 마크다운**: Teacher-Student 기반 성능 고도화 설명.
    2. **7-1. Teacher Model Optuna Tuning 마크다운 & 코드 셀**: `import optuna` 및 `study` 변수를 생성하는 최적화 로직 포함.
    3. **7-2. 최적화된 Teacher 모델 학습 마크다운 & 코드 셀**: `study.best_params`를 사용하여 최종 모델 학습 및 Soft Label 생성.
    4. **나머지 Student 학습 및 비교 로직**: 기존 코드 중 유효한 부분을 유지하며 변수명(`scenario4_features` 등) 일관성 유지.

## Verification Plan

### Automated Tests
- 수정된 `train.ipynb`의 Phase 7 셀들을 순차적으로 실행하여 오류 없이 동작하는지 확인합니다.
- 특히 `best_params = study.best_params` 라인에서 `NameError`가 발생하지 않는지 검증합니다.

### Manual Verification
- `train.ipynb`를 열어 Phase 5 결론 이후와 Phase 7 섹션의 셀들이 논리적으로 올바른 순서로 배치되었는지 육안으로 확인합니다.
- `study` 변수가 사용되기 전에 `optuna.create_study()`와 `study.optimize()`가 포함된 셀이 존재하는지 확인합니다.
