# 클러스터링 ARI 성능 획기적 개선 계획

## 1. 근본 원인 분석 (Root Cause)

> [!CAUTION]
> 현재 ARI가 낮은 **가장 핵심적인 원인**은 알고리즘 문제가 아니라, **입력 차원의 극단적 불일치**입니다.

| 구분 | Ground Truth (11-2) | Prediction (11-3) |
|------|--------------------|--------------------|
| **사용 피처** | S4_ALL (17개: 8개 원본 + 9개 파생변수) + capacity | **capacity 1개** (또는 pred_capacity 1개) |
| **차원 수** | **17~18차원** | **1차원** |
| **클러스터링** | KMeans (k=4) | KMeans (k=4) |

Ground Truth는 전압, 임피던스, 파생변수 등 **17차원 공간**에서 배터리의 복합적 특성을 반영하여 군집을 나누지만, Prediction은 **capacity 1차원**으로만 군집을 나눕니다. 1차원 직선 위의 4등분과 17차원 공간의 4등분이 일치할 수 없는 것은 **수학적으로 당연한 결과**입니다.

## 2. 개선 전략 (5단계)

### 전략 1. 🎯 다변량 예측 (Multi-Output Prediction) — **가장 핵심**
> 현재: capacity 1개만 예측 → 개선: **원본 8개 피처 전체를 예측**

Phase 9의 최고 성능 모델(예: CatBoost, FLAML)을 활용하여, `S1+v25`(초기측정 + 완전방전) 4개 피처만으로 **나머지 4개 피처(v42_voltage, v42_impedance, v36_voltage, v36_impedance)를 동시에 예측**합니다.

```python
from sklearn.multioutput import MultiOutputRegressor

# 입력: 초기측정 + 완전방전 (4개)
input_features = ['initial_voltage', 'initial_impedance', 'v25_voltage', 'v25_impedance']
# 출력: 나머지 + capacity (5개)
output_features = ['v42_voltage', 'v42_impedance', 'v36_voltage', 'v36_impedance', 'capacity']

model = MultiOutputRegressor(CatBoostRegressor(...))
model.fit(X_train[input_features], X_train[output_features])
pred_all = model.predict(X_test[input_features])

# 예측된 8개 + capacity로 클러스터링 → ARI 비교
```

**기대 효과:** 17차원 → 9차원(8개 예측 피처 + capacity)으로 차원 격차를 **대폭 축소**. ARI 0.4~0.7 이상 기대.

---

### 전략 2. 📐 차원 축소 후 클러스터링 (UMAP / PCA 기반)
> GT와 Pred 모두 **동일한 저차원 공간**에서 비교

```python
import umap

# GT 피처를 2~3차원으로 축소
reducer = umap.UMAP(n_components=3, random_state=42)
X_gt_umap = reducer.fit_transform(X_gt_scaled)

# Pred 피처도 동일 변환기로 축소
X_pred_umap = reducer.transform(X_pred_scaled)  

# 동일 공간에서 클러스터링 비교
```

**기대 효과:** 고차원의 복잡한 구조를 보존하면서 비교 가능. 특히 UMAP은 지역적 구조(local structure)를 잘 보존하므로 ARI 향상에 유리.

---

### 전략 3. 🔄 클러스터링 알고리즘 다양화
> KMeans의 구형(spherical) 가정이 배터리 데이터에 부적합할 수 있음

| 알고리즘 | 장점 | 적합한 경우 |
|----------|------|------------|
| **GMM** (Gaussian Mixture) | 타원형 클러스터, 확률적 소속 | 클러스터가 겹치는 경우 |
| **Spectral Clustering** | 비볼록(non-convex) 형태 처리 | 복잡한 데이터 구조 |
| **HDBSCAN** | 밀도 기반, 자동 k 결정, 노이즈 감지 | 이상치가 많은 경우 |
| **Agglomerative** | 계층적 관계 파악 | 클러스터 간 계층 존재 시 |

```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
labels_gmm = gmm.fit_predict(X_scaled)
```

---

### 전략 4. 🧠 딥 클러스터링 (Deep Embedded Clustering)
> Autoencoder로 클러스터 친화적 잠재 공간(latent space)을 학습

```python
# Variational Autoencoder (VAE) + KMeans
# 1. VAE로 원본 17차원 → 잠재 공간 3차원 압축
# 2. GT: 원본 17차원 → VAE 인코더 → 3차원 → KMeans
# 3. Pred: 예측된 피처 → 동일 VAE 인코더 → 3차원 → KMeans
# 4. ARI 비교
```

**기대 효과:** 클러스터 구조를 보존하는 방향으로 차원 축소가 학습되므로 가장 높은 ARI 달성 가능. 단, 구현 복잡도가 높음.

---

### 전략 5. 🎲 헝가리안 매칭 (Hungarian Algorithm) 적용
> 클러스터 라벨 번호 불일치(permutation) 문제 해결

KMeans는 클러스터 번호를 임의로 할당하므로, GT의 클러스터 0이 Pred의 클러스터 2와 실제로는 같은 그룹일 수 있습니다. **헝가리안 알고리즘**으로 최적 매칭 후 ARI를 재계산합니다.

> [!NOTE]
> ARI 자체는 라벨 순서에 불변(permutation-invariant)이므로 이 전략은 **시각화 시 색상 매칭 개선**에 더 유용합니다.

## 3. 추천 실행 순서

| 우선순위 | 전략 | 예상 ARI 개선폭 | 구현 난이도 |
|---------|------|----------------|-----------|
| ⭐⭐⭐ | **전략 1** (다변량 예측) | +0.3 ~ +0.5 | 중간 |
| ⭐⭐ | **전략 2** (UMAP 축소) | +0.1 ~ +0.3 | 낮음 |
| ⭐⭐ | **전략 3** (GMM 등) | +0.05 ~ +0.15 | 낮음 |
| ⭐ | **전략 4** (딥 클러스터링) | +0.2 ~ +0.4 | 높음 |
| 보조 | **전략 5** (매칭) | 시각화 개선 | 매우 낮음 |

## 4. 검증 계획

1. 각 전략 적용 후 ARI, Silhouette Score, NMI(Normalized Mutual Info) 3가지 지표로 비교
2. PCA 2D 산점도로 GT vs Pred 클러스터 시각적 비교
3. 전략 1+2+3 조합 시 시너지 효과 측정

## 참고 자료
- [MDPI: Clustering for Battery Degradation Forecasting (2024)](https://www.mdpi.com/)
- [ResearchGate: Deep Neural Networks for SOH Estimation (2024)](https://www.researchgate.net/)
- [BatteryML: Feature Manipulation Platform (arXiv, 2024)](https://arxiv.org/)
- [Deep Variational Autoencoder for Battery Feature Reduction (2025)](https://www.researchgate.net/)
