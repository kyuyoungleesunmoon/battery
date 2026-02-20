"""
🔋 중고 배터리 셀 용량(Capacity) 예측 모델 개발 — Streamlit 대시보드
"""
import streamlit as st
from pathlib import Path
from PIL import Image

# ──────────────────────────── 페이지 설정 ────────────────────────────
st.set_page_config(
    page_title="🔋 배터리 용량 예측 분석 보고서",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent

# ──────────────────────────── 유틸 함수 ────────────────────────────
def load_img(name: str):
    """이미지 파일을 읽어 반환. 없으면 None."""
    p = BASE / name
    if p.exists():
        return Image.open(p)
    return None


def show_img(name: str, caption: str = "", use_width: bool = True):
    """이미지를 표시. 없으면 경고."""
    img = load_img(name)
    if img is not None:
        st.image(img, caption=caption, width="stretch" if use_width else "content")
    else:
        st.warning(f"이미지 파일을 찾을 수 없습니다: {name}")


# ──────────────────────────── 사이드바 ────────────────────────────
st.sidebar.title("📑 목차")
page = st.sidebar.radio(
    "분석 단계 선택",
    [
        "🏠 개요",
        "Phase 1: EDA",
        "Phase 2: 파생변수",
        "Phase 3: 전처리",
        "Phase 4: 데이터 분할",
        "Phase 5: 모델 학습",
        "Phase 7: Knowledge Distillation",
        "Phase 8: 성능 고도화",
        "Phase 9: S1 집중 개선",
        "Phase 10: KD 정보채널 보완",
        "📊 종합 결론",
    ],
    index=0,
)

# ════════════════════════════════════════════════════════════════════
#  🏠 개요
# ════════════════════════════════════════════════════════════════════
if page == "🏠 개요":
    st.title("🔋 중고 배터리 셀 용량(Capacity) 예측 모델 개발")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("배터리 셀 수", "1,040개")
    col2.metric("원본 피처", "8개")
    col3.metric("파생 피처", "9개")
    col4.metric("최종 피처", "17개")

    st.markdown("""
    > **프로젝트**: 중고 배터리 셀의 초기 측정 데이터 기반 실제 방전 용량 예측  
    > **데이터**: BS-LSBAT-S240629 배치, 1,040개 배터리 셀  
    > **분석 환경**: Python 3.14.2 · XGBoost 3.2.0 · LightGBM 4.6.0 · Optuna 4.7.0 · scikit-learn 1.8.0
    """)

    st.markdown("### 📋 분석 파이프라인")
    st.markdown("""
    | Phase | 내용 | 핵심 결과 |
    |------|------|----------|
    | **Phase 1** | 탐색적 데이터 분석 (EDA) | `v25_voltage` 최강 예측자 (r=-0.595) |
    | **Phase 2** | 파생변수 생성 (16개) | `voltage_sag` r=+0.600 달성 |
    | **Phase 3** | 전처리 및 피처 선택 | 17개 최종 피처, 4개 시나리오 |
    | **Phase 4** | Train/Test 분할 | 831/208 (80:20), KS 검정 통과 |
    | **Phase 5** | 모델 학습 및 평가 | S4+XGBoost 최고 (RMSE=0.0246) |
    | **Phase 7** | Knowledge Distillation | KD 효과 0.13% (미미) |
    | **Phase 8** | 성능 고도화 실험 (5개) | S3 Base 최선 (RMSE=0.0348) |
    | **Phase 9** | S1 집중 개선 (6개 실험) | **S1 Stacking RMSE=0.0346** — S3 추월 |
    | **Phase 10** | KD 정보채널 보완 (RKD, LambdaKD, Bridge) | **S3 LambdaKD RMSE=0.0347**, Spearman +24% |
    """)

    st.markdown("### 🏆 최종 핵심 결과")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best RMSE (S4)", "0.0245 Ah", help="S4 + Optuna XGBoost")
    c2.metric("Best RMSE (S1)", "0.0346 Ah", help="S1 Stacking 앙상블")
    c3.metric("Best KD", "0.0347 Ah", help="S3 LambdaKD (Phase 10)")
    c4.metric("Spearman 향상", "+24%", help="0.26→0.32 (LambdaKD)")

    st.info("**핵심 발견**: Phase 9에서 S1 Stacking으로 S3 추월(0.0346<0.0348). Phase 10에서 LambdaKD(순위 기반 KD)로 정보채널 부재 보완 — S1 +5.6%, Spearman +24% 달성.")

# ════════════════════════════════════════════════════════════════════
#  Phase 1: EDA
# ════════════════════════════════════════════════════════════════════
elif page == "Phase 1: EDA":
    st.title("Phase 1: 데이터 탐색 및 분석 (EDA)")
    st.markdown("중고 배터리 셀의 초기 측정 데이터(전압, AC 임피던스)로부터 실제 방전 용량(Capacity)을 예측하기 위한 탐색적 데이터 분석입니다.")
    st.markdown("---")

    # ── 1-1. 환경 설정 ──
    st.header("1-1. 라이브러리 로드 및 환경 설정")
    st.code("""import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False""", language="python")
    st.markdown("""
    | 라이브러리 | 용도 |
    |---|---|
    | `pandas` | 데이터프레임 조작 |
    | `numpy` | 수치 연산 |
    | `matplotlib` / `seaborn` | 시각화 |
    | `scipy.stats` | 통계 검정 (Z-score 등) |
    """)

    # ── 1-2. 데이터 로드 ──
    st.header("1-2. 데이터 로드 및 컬럼 정리")
    st.markdown("""
    - **데이터 크기**: 1,040개 셀 × 10개 컬럼 (cell_id + 8개 피처 + 1개 타겟)
    - **피처 구성**: 4개 측정 구간 × 2개 측정값 = **8개 원본 피처**
    - **셀 ID 패턴**: `BS-LSBAT-S240629-XXXX`
    """)

    st.markdown("""
    | 구간 | 의미 | 전압 대표값 | 임피던스 대표값 |
    |---|---|---|---|
    | Initial (OCV) | 초기 개방회로전압 | 3.455~3.457V | 10.93~11.61mΩ |
    | 4.2V (Full) | 만충 상태 | 4.177~4.178V | 10.31~10.56mΩ |
    | 2.5V (Cut-off) | 방전 종지 | 2.961~2.977V | 10.55~11.12mΩ |
    | 3.6V (Nominal) | 공칭 전압 부근 | 3.59V | 10.62~11.08mΩ |
    """)

    # ── 1-3. 기본 통계 ──
    st.header("1-3. 기본 통계 분석")
    import pandas as pd
    stats_data = {
        "피처": ["initial_voltage", "initial_impedance", "v42_voltage", "v42_impedance",
                 "v25_voltage", "v25_impedance", "v36_voltage", "v36_impedance", "**capacity**"],
        "mean": [3.4560, 11.40, 4.1780, 10.67, 2.9700, 10.81, 3.5899, 11.89, 5.026],
        "std": [0.0018, 0.82, 0.0030, 0.40, 0.0216, 0.32, 0.0007, 31.59, 0.033],
        "min": [3.4512, 9.47, 4.1705, 9.79, 2.8922, 9.88, 3.5852, 10.14, 4.792],
        "max": [3.4622, 14.14, 4.1940, 12.16, 3.0282, 11.86, 3.5930, 1032.0, 5.097],
        "missing": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "skew": [0.44, 0.23, 2.65, 0.69, -0.17, 0.05, -2.42, 32.24, -2.16],
        "kurtosis": [0.19, -0.28, 8.41, 0.25, -0.44, -0.10, 27.07, 1040.0, 7.84],
    }
    st.dataframe(pd.DataFrame(stats_data), width="stretch", hide_index=True)

    st.markdown("""
    **핵심 해석:**
    1. **결측치**: 전체 0개 → 별도의 결측값 처리 불필요
    2. **v42_voltage / v36_voltage**: 표준편차가 매우 작음 → 사실상 상수에 가까움
    3. **v25_voltage**: 상대적으로 변동성이 높아 **셀 간 차이를 가장 잘 반영**
    4. **v36_impedance**: min 10.14 / max 1032.0 → **극단 이상치 1개 존재**
    5. **Capacity (타겟)**: 평균 5.026 Ah, 표준편차 0.033 Ah → **약 6% 이내의 매우 좁은 변동**
    6. Capacity의 음의 왜도(skew=-2.16): **왼쪽 꼬리 분포**
    """)

    # ── 1-4. 분포 시각화 ──
    st.header("1-4. 분포 시각화 (히스토그램 + KDE)")
    show_img("eda_01_distributions.png", "9개 피처 히스토그램 + KDE 분포")
    with st.expander("📖 각 피처 분포 특성 상세"):
        st.markdown("""
        | 피처 | 분포 형태 | 특이사항 |
        |---|---|---|
        | `initial_voltage` | 정규분포에 가까움 | 평균 3.456V 부근 집중 |
        | `initial_impedance` | 오른쪽 꼬리 분포 | 일부 높은 임피던스 값 존재 |
        | `v42_voltage` | 매우 좁은 범위 집중 | 4.178V 부근에 밀집, 사실상 상수 |
        | `v42_impedance` | 오른쪽 꼬리 | 대부분 10~11mΩ, 일부 높은 값 |
        | `v25_voltage` | **가장 넓은 분포** | **셀 열화 차이를 가장 잘 반영** |
        | `v25_impedance` | 정규분포에 가까움 | 10~11mΩ 범위 |
        | `v36_voltage` | 좁은 범위 | 3.59V 부근 밀집 |
        | `v36_impedance` | 정규분포에 가까움 | 10~11mΩ (극단 이상치 1개 제외) |
        | `capacity` | 왼쪽 꼬리 | 대부분 5.0Ah 이상, 일부 열화 셀 4.8Ah 이하 |
        """)
    st.info("**핵심**: `v25_voltage`는 가장 넓은 분포로 예측 변수로서 가장 유망합니다.")

    # ── 1-5. 박스플롯 ──
    st.header("1-5. 박스플롯 (이상치 확인)")
    show_img("eda_02_boxplots.png", "9개 피처 박스플롯 (IQR 이상치)")
    st.markdown("""
    - **`v42_voltage`**: 박스가 매우 좁고 위스커 밖에 다수의 이상치
    - **`v36_voltage`**: 가장 많은 이상치 → 분포가 좁아 IQR이 작기 때문에 과도 검출
    - **`capacity`**: 아래쪽 이상치 → 공칭 용량보다 현저히 낮은 열화 셀
    """)

    # ── 1-6. 상관분석 ──
    st.header("1-6. 상관분석 (Pearson / Spearman)")
    show_img("eda_03_correlation.png", "Pearson(좌) & Spearman(우) 상관계수 히트맵")

    st.subheader("Capacity와의 상관계수 순위")
    corr_data = {
        "순위": [1, 2, 3, 4, 5, 6, 7, 8],
        "피처": ["v25_voltage", "v25_impedance", "v42_impedance", "initial_impedance",
                  "initial_voltage", "v36_impedance", "v42_voltage", "v36_voltage"],
        "Pearson r": [-0.5952, -0.2130, -0.1773, -0.1499, 0.1365, -0.0211, -0.0173, 0.0137],
        "|Pearson|": [0.5952, 0.2130, 0.1773, 0.1499, 0.1365, 0.0211, 0.0173, 0.0137],
        "평가": ["🟢 가장 강한 상관", "🟡 보통", "🟡 보통", "🔴 약함", "🔴 약함", "🔴 무시 가능", "🔴 무시 가능", "🔴 무시 가능"],
    }
    st.dataframe(pd.DataFrame(corr_data), width="stretch", hide_index=True)
    st.warning("**핵심**: `v25_voltage`가 Capacity와 **유일하게 중간 이상의 상관관계**(|r| > 0.5). 대부분의 원본 피처가 약한 상관이므로 **파생변수 생성이 필수**입니다.")

    # ── 1-7. 산점도 ──
    st.header("1-7. 피처 vs Capacity 산점도")
    show_img("eda_04_scatter.png", "8개 피처 vs Capacity 산점도 (추세선 포함)")
    st.markdown("""
    - **`v25_voltage` (r=-0.595)**: 가장 뚜렷한 **음의 선형 관계**
    - **임피던스 피처들 (r ≈ -0.15 ~ -0.21)**: 약한 음의 경향
    - **`v42_voltage`, `v36_voltage`**: 거의 수직 분포 → 예측 기여 미미
    """)

    # ── 1-8. 이상치 탐지 ──
    st.header("1-8. 이상치 탐지 (IQR + Z-score)")
    outlier_data = {
        "피처": ["initial_voltage", "initial_impedance", "v42_voltage", "v42_impedance",
                 "v25_voltage", "v25_impedance", "v36_voltage", "v36_impedance", "capacity"],
        "IQR 이상치 수": [12, 6, 57, 73, 17, 2, 138, 1, 20],
        "IQR 비율(%)": [1.15, 0.58, 5.48, 7.02, 1.63, 0.19, 13.27, 0.10, 1.92],
        "Z>3 수": [0, 0, 0, 0, 0, 0, 10, 1, 0],
        "Z>3 비율(%)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.96, 0.10, 0.0],
    }
    st.dataframe(pd.DataFrame(outlier_data), width="stretch", hide_index=True)
    st.metric("이상치 포함 셀", "379개 / 1,040개 (36.4%)")

    # ── 1-9. 이상치 하이라이트 ──
    st.header("1-9. 이상치 하이라이트 시각화")
    show_img("eda_05_outliers.png", "이상치(빨간 x) vs 정상(파란 ●) 산점도")
    st.markdown("이상치가 특정 Capacity 구간에 집중되지 않고 전반적으로 분포 → **자연 편차**이지 측정 오류가 아님")

    # ── 1-10. 피처 그룹별 ──
    st.header("1-10. 피처 그룹별 패턴 분석")
    show_img("eda_06_feature_groups.png", "측정 구간별 전압(좌) / 임피던스(우) 박스플롯")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **전압 분포:**
        - Initial ≈ 3.456V (SOC 반영)
        - 4.2V ≈ 4.178V (충전 프로토콜 일정)
        - **2.5V ≈ 2.97V (가장 넓은 분포)**
        - 3.6V ≈ 3.59V (변동성 낮음)
        """)
    with c2:
        st.markdown("""
        **AC 임피던스 분포:**
        - 4개 측정 구간 모두 10~12mΩ
        - Initial에서 중앙값 높음
        - 4.2V에서 약간 감소 → 충전 시 내부저항 감소
        """)

    # ── 1-11. Pairplot ──
    st.header("1-11. 임피던스 Pairplot")
    show_img("eda_07_pairplot.png", "임피던스 피처 간 관계 Pairplot")
    st.markdown("""
    - 임피던스 피처 간 **강한 양의 선형 관계** → 다중공선성 경고
    - 임피던스 → Capacity: 약한 음의 경향, 산점도 넓게 퍼져 예측력 제한
    """)

    # ── 1-12. EDA 요약 ──
    st.header("1-12. Phase 1 EDA 요약")
    summary_data = {
        "항목": ["데이터 품질", "타겟 분포", "최강 예측자", "이상치", "다중공선성"],
        "결과": ["결측치 0, 1040개 셀", "5.026±0.033 Ah", "v25_voltage (r=-0.595)", "36.4% (379/1040)", "임피던스 피처 간 높은 상관"],
        "의미": ["전처리 부담 낮음", "매우 좁은 범위 → 정밀 모델 필요", "방전 종지 전압이 유일한 중간 상관", "IQR 기준 과도 검출, 자연 편차", "VIF 분석 및 피처 선택 필요"],
    }
    st.dataframe(pd.DataFrame(summary_data), width="stretch", hide_index=True)

# ════════════════════════════════════════════════════════════════════
#  Phase 2: 파생변수
# ════════════════════════════════════════════════════════════════════
elif page == "Phase 2: 파생변수":
    st.title("Phase 2: 파생변수 생성 (Feature Engineering)")
    st.markdown("원본 8개 피처의 단독 예측력이 대부분 약하므로, 배터리 도메인 지식에 기반한 **16개 파생변수**를 설계했습니다.")
    st.markdown("---")

    # ── 2-1. 임피던스 ──
    st.header("2-1. 임피던스 기반 파생변수 (8개)")
    st.code("""df['impedance_delta_42'] = df[z_42] - df[z_initial]
df['impedance_delta_25'] = df[z_25] - df[z_initial]
df['impedance_delta_36'] = df[z_36] - df[z_initial]
df['impedance_ratio_42'] = df[z_42] / df[z_initial]
df['impedance_ratio_25'] = df[z_25] / df[z_initial]
df['impedance_range'] = df[impedance_cols].max(axis=1) - df[impedance_cols].min(axis=1)
df['impedance_mean'] = df[impedance_cols].mean(axis=1)
df['impedance_std'] = df[impedance_cols].std(axis=1)""", language="python")

    import pandas as pd
    imp_data = {
        "파생변수": ["impedance_delta_42", "impedance_delta_25", "impedance_delta_36",
                    "impedance_ratio_42", "impedance_ratio_25", "impedance_range",
                    "impedance_mean", "impedance_std"],
        "수식": ["Z₄.₂V − Z_init", "Z₂.₅V − Z_init", "Z₃.₆V − Z_init",
                "Z₄.₂V / Z_init", "Z₂.₅V / Z_init", "Z_max − Z_min",
                "mean(Z_all)", "std(Z_all)"],
        "평균": [-0.680, -0.391, 0.389, 0.941, 0.966, 1.839, 11.217, 0.882],
        "Capacity r": [-0.005, -0.034, 0.019, -0.010, -0.037, 0.020, 0.013, 0.020],
    }
    st.dataframe(pd.DataFrame(imp_data), width="stretch", hide_index=True)
    st.error("임피던스 기반 파생변수는 **모든 항목에서 |r| < 0.04** → 단독 선형 예측력 거의 없음")

    # ── 2-2. 전압 ──
    st.header("2-2. 전압 기반 파생변수 (5개)")
    st.code("""df['voltage_delta_initial_42'] = df[v_42] - df[v_initial]
df['voltage_delta_42_25'] = df[v_42] - df[v_25]
df['voltage_delta_42_36'] = df[v_42] - df[v_36]
df['voltage_sag'] = df[v_initial] - df[v_25]
df['ocv_deviation'] = df[v_initial] - 3.6""", language="python")

    volt_data = {
        "파생변수": ["voltage_delta_initial_42", "voltage_delta_42_25", "voltage_delta_42_36", "voltage_sag", "ocv_deviation"],
        "수식": ["V₄.₂V − V_init", "V₄.₂V − V₂.₅V", "V₄.₂V − V₃.₆V", "V_init − V₂.₅V", "V_init − 3.6"],
        "평균": ["0.721V", "1.230V", "0.589V", "0.509V", "-0.143V"],
        "Capacity r": [-0.138, 0.594, -0.048, 0.600, 0.137],
        "평가": ["🔴 약함", "🟢 강함", "🔴 약함", "🟢 최강 ✨", "🔴 약함"],
    }
    st.dataframe(pd.DataFrame(volt_data), width="stretch", hide_index=True)
    st.success("**`voltage_sag`(r=+0.600)** 와 **`voltage_delta_42_25`(r=+0.594)** 가 원본 최강 피처를 미세하게 능가!")

    # ── 2-3. 복합 ──
    st.header("2-3. 복합 파생변수 (3개)")
    comp_data = {
        "파생변수": ["impedance_voltage_product", "power_loss_indicator", "health_index"],
        "수식": ["Z_init × V_init", "Z₄.₂V × (V₄.₂V − V₂.₅V)", "V₄.₂V / Z₄.₂V"],
        "Capacity r": [-0.146, 0.250, 0.171],
        "평가": ["🔴 약함", "🟡 보통", "🔴 약함"],
    }
    st.dataframe(pd.DataFrame(comp_data), width="stretch", hide_index=True)

    # ── 2-4. 산점도 ──
    st.header("2-4. 파생변수 vs Capacity 산점도")
    show_img("phase2_derived_vs_capacity.png", "16개 파생변수 vs Capacity 산점도 (추세선 포함)")

    # ── 2-5. 상관 히트맵 ──
    st.header("2-5. 파생변수 간 상관관계 히트맵")
    show_img("phase2_correlation_heatmap.png", "파생변수 간 Pearson 상관계수 히트맵")
    st.warning("**다중공선성 경고**: |r| > 0.9인 쌍 9개 발견 → 중복 피처 제거 필요")

    # ── 2-6. 요약 ──
    st.header("2-6. Phase 2 요약")
    c1, c2, c3 = st.columns(3)
    c1.metric("총 파생변수", "16개", "임피던스 8 + 전압 5 + 복합 3")
    c2.metric("원본 최강 |r|", "0.595", "v25_voltage")
    c3.metric("파생 최강 |r|", "0.600", "voltage_sag ✨")

# ════════════════════════════════════════════════════════════════════
#  Phase 3: 전처리
# ════════════════════════════════════════════════════════════════════
elif page == "Phase 3: 전처리":
    st.title("Phase 3: 데이터 전처리 및 피처 선택")
    st.markdown("---")

    # ── 3-1 ──
    st.header("3-1. 극단 이상치 제거")
    import pandas as pd
    st.markdown("""
    **제거 대상**: `BS-LSBAT-S240629-0126`
    - `v36_impedance` = **1032.0 mΩ** (정상 범위: 10~12 mΩ)
    - 명백한 측정 오류 (1000배 이상 이탈)
    """)
    c1, c2 = st.columns(2)
    c1.metric("제거 전", "1,040개")
    c2.metric("제거 후", "1,039개", "-1개")
    st.info("단 1개의 극단 이상치가 `impedance_mean`의 상관계수를 +0.013 → **-0.095**로 정상화")

    # ── 3-2 ──
    st.header("3-2. VIF(분산팽창인수) 분석")
    show_img("phase3_vif_comparison.png", "VIF 비교 (피처 제거 전/후)")
    st.markdown("""
    - 모든 피처의 VIF > 10 → 파생변수가 원본의 선형 변환이므로 **구조적 불가피**
    - 트리 기반 모델은 VIF와 무관 → **극단적 중복만 제거** 전략
    """)

    # ── 3-3 ──
    st.header("3-3. 중복 피처 제거")
    remove_data = {
        "제거 피처": ["impedance_delta_42/25", "impedance_delta_36, range, std",
                    "voltage_delta_42_25", "voltage_delta_42_36"],
        "유지 피처": ["impedance_ratio_42/25", "impedance_mean", "voltage_sag", "(제거)"],
        "근거": ["r=0.999, ratio가 스케일 독립적", "이상치 왜곡, mean이 해석 용이",
                 "r=0.991, sag가 r=+0.600 우위", "r=-0.048, 예측 기여 없음"],
    }
    st.dataframe(pd.DataFrame(remove_data), width="stretch", hide_index=True)
    st.markdown("**최종**: 원본 8개 + 파생 9개 = **17개 피처**")

    # ── 3-4 ──
    st.header("3-4. 시나리오별 피처셋")
    scenario_data = {
        "시나리오": ["S1 (Initial)", "S2 (+4.2V)", "S3 (+3.6V)", "S4 (전체)"],
        "피처 수": [3, 9, 12, 17],
        "최대 |r|": [0.150, 0.177, 0.177, 0.600],
        "측정 비용": ["최소", "낮음", "중간", "높음"],
        "소요 시간": ["~1분", "~30분", "~1시간", "~2시간+"],
    }
    st.dataframe(pd.DataFrame(scenario_data), width="stretch", hide_index=True)
    st.warning("**S1~S3는 v25_voltage 정보 부재** → S4에서만 핵심 예측 피처 사용 가능")

    # ── 3-5 ──
    st.header("3-5. 정제 후 상관관계 시각화")
    show_img("phase3_cleaned_correlation.png", "S4 피처 상관 히트맵(좌) + Capacity 상관계수 바 차트(우)")

    # ── 3-6 ──
    st.header("3-6. 유효 파생변수 vs Capacity")
    show_img("phase3_derived_scatter.png", "정제 후 유효 파생변수 vs Capacity 산점도")

# ════════════════════════════════════════════════════════════════════
#  Phase 4: 데이터 분할
# ════════════════════════════════════════════════════════════════════
elif page == "Phase 4: 데이터 분할":
    st.title("Phase 4: 최종 학습 데이터 구성")
    st.markdown("---")

    import pandas as pd

    st.header("4-1. Train/Test 분할")
    c1, c2, c3 = st.columns(3)
    c1.metric("전체", "1,039개")
    c2.metric("Train", "831개 (80%)")
    c3.metric("Test", "208개 (20%)")

    split_data = {
        "항목": ["샘플 수", "Capacity 평균", "Capacity 표준편차", "KS 검정 p-value"],
        "Train": ["831 (80%)", "5.0259", "0.0333", ""],
        "Test": ["208 (20%)", "5.0273", "0.0315", ""],
    }
    st.dataframe(pd.DataFrame(split_data), width="stretch", hide_index=True)
    st.success("KS 검정 p=0.4561 > 0.05 → **Train/Test 분포 동질성 확인** ✅")

    st.header("4-2. 시나리오별 데이터셋")
    ds_data = {
        "시나리오": ["S1_INITIAL", "S2_INITIAL_42V", "S3_INITIAL_42V_36V", "S4_ALL"],
        "피처 수": [3, 9, 12, 17],
        "Train shape": ["(831, 3)", "(831, 9)", "(831, 12)", "(831, 17)"],
        "Test shape": ["(208, 3)", "(208, 9)", "(208, 12)", "(208, 17)"],
        "결측치/무한값": ["✅ 0", "✅ 0", "✅ 0", "✅ 0"],
    }
    st.dataframe(pd.DataFrame(ds_data), width="stretch", hide_index=True)

    st.header("4-3. 스케일링 전략")
    st.markdown("""
    | 모델 | 스케일링 | 방법 |
    |------|---------|------|
    | RF / XGBoost / LightGBM | 없음 | 트리 기반 → 스케일 무관 |
    | Linear Regression / SVR | RobustScaler | 중앙값 + IQR 기준 (이상치 강건) |
    """)

    st.header("4-4. Train/Test 분포 시각화")
    show_img("phase4_data_split.png", "Train/Test 분포 (Capacity, 시나리오별 피처 수, 스케일링 비교, 박스플롯)")

# ════════════════════════════════════════════════════════════════════
#  Phase 5: 모델 학습
# ════════════════════════════════════════════════════════════════════
elif page == "Phase 5: 모델 학습":
    st.title("Phase 5: 모델 학습 및 평가")
    st.markdown("---")

    import pandas as pd

    st.header("5-1. 사용 모델 및 학습 방법")
    model_info = {
        "모델": ["Linear Regression", "SVR", "Random Forest", "XGBoost", "LightGBM"],
        "유형": ["선형", "커널(RBF)", "앙상블(배깅)", "앙상블(부스팅)", "앙상블(부스팅)"],
        "주요 하이퍼파라미터": ["기본", "C=1.0, ε=0.01", "n_estimators=100", "n=100, lr=0.1", "n=100, lr=0.1"],
        "스케일링": ["RobustScaler", "RobustScaler", "없음", "없음", "없음"],
    }
    st.dataframe(pd.DataFrame(model_info), width="stretch", hide_index=True)

    st.header("5-2. 전체 학습 결과 (4×5 = 20조합)")

    results_all = {
        "시나리오": (["S1"]*5 + ["S2"]*5 + ["S3"]*5 + ["S4"]*5),
        "모델": ["LinearReg", "SVR", "RandomForest", "XGBoost", "LightGBM"] * 4,
        "RMSE": [0.0355, 0.0361, 0.0387, 0.0368, 0.0361,
                 0.0358, 0.0348, 0.0368, 0.0386, 0.0358,
                 0.0358, 0.0360, 0.0350, 0.0384, 0.0363,
                 0.0262, 0.0260, 0.0249, 0.0246, 0.0271],
        "R²": [0.0548, 0.0227, -0.1238, -0.0159, 0.0222,
               0.0384, 0.0948, -0.0144, -0.1154, 0.0396,
               0.0396, 0.0303, 0.0838, -0.1040, 0.0127,
               0.4871, 0.4956, 0.5350, 0.5454, 0.4507],
        "MAPE": [0.0042, 0.0044, 0.0048, 0.0044, 0.0042,
                 0.0044, 0.0040, 0.0045, 0.0047, 0.0043,
                 0.0043, 0.0043, 0.0043, 0.0047, 0.0043,
                 0.0029, 0.0027, 0.0024, 0.0025, 0.0027],
    }
    df_results = pd.DataFrame(results_all)

    # 시나리오별 탭
    tabs = st.tabs(["S1 (Initial)", "S2 (+4.2V)", "S3 (+3.6V)", "S4 (전체)", "전체 비교"])
    for i, sc in enumerate(["S1", "S2", "S3", "S4"]):
        with tabs[i]:
            sub = df_results[df_results["시나리오"] == sc].copy()
            best_idx = sub["RMSE"].idxmin()
            st.dataframe(
                sub.style.highlight_min(subset=["RMSE"], color="#d4edda"),
                width="stretch", hide_index=True,
            )
            best_row = sub.loc[best_idx]
            st.metric(f"{sc} Best", f"{best_row['모델']} — RMSE {best_row['RMSE']:.4f}")

    with tabs[4]:
        st.dataframe(
            df_results.sort_values("RMSE").style.highlight_min(subset=["RMSE"], color="#d4edda"),
            width="stretch", hide_index=True,
        )

    st.header("5-3. 시나리오별 최고 성능 요약")
    best_data = {
        "시나리오": ["S1 (Initial)", "S2 (+4.2V)", "S3 (+3.6V)", "S4 (전체)"],
        "Best 모델": ["LinearReg", "SVR", "RandomForest", "🏆 XGBoost"],
        "RMSE": [0.0355, 0.0348, 0.0350, 0.0246],
        "R²": [0.055, 0.095, 0.084, 0.545],
        "MAPE": ["0.42%", "0.40%", "0.43%", "0.25%"],
        "평가": ["❌ 예측 불가", "❌ 거의 무의미", "❌ 미미", "✅ 유의미"],
    }
    st.dataframe(pd.DataFrame(best_data), width="stretch", hide_index=True)

    st.header("5-4. RMSE 비교 시각화")
    show_img("phase5_rmse_barplot.png", "시나리오 × 모델 RMSE 비교")

    col1, col2 = st.columns(2)
    with col1:
        show_img("phase5_r2_heatmap.png", "R² 히트맵")
    with col2:
        show_img("phase5_feature_importance.png", "피처 중요도")

    st.header("5-5. Phase 5 핵심 분석")
    st.error("""
    **피처셋이 모델보다 중요하다:**
    - S1~S3 최고 성능: RMSE ≈ 0.035 (R² < 0.10) → 모델 종류 무관
    - S4 최저 성능: RMSE = 0.027 (R² = 0.45) → 가장 약한 모델도 S1~S3보다 우수
    - **어떤 최첨단 알고리즘도, 2.5V 방전 측정 없이는 RMSE 0.035 이하로 내려가기 어렵습니다.**
    """)

# ════════════════════════════════════════════════════════════════════
#  Phase 7: KD
# ════════════════════════════════════════════════════════════════════
elif page == "Phase 7: Knowledge Distillation":
    st.title("Phase 7: Knowledge Distillation")
    st.markdown("**목표**: S4(17피처) Teacher의 지식을 S2(9피처) Student에 전이하여, 적은 측정으로도 높은 성능 달성")
    st.markdown("---")

    import pandas as pd

    st.header("7-1. Optuna Teacher 하이퍼파라미터 최적화")
    st.code("""study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)""", language="python")

    opt_data = {
        "파라미터": ["n_estimators", "max_depth", "learning_rate", "subsample",
                    "colsample_bytree", "reg_alpha (L1)", "reg_lambda (L2)"],
        "최적값": [653, 7, 0.182, 0.835, 0.810, 0.002, 3.887],
        "해석": ["충분한 앙상블 효과", "과적합-성능 균형", "비교적 높은 학습률",
                "83.5% 데이터 사용", "81% 피처 사용", "거의 없음", "과적합 방지"],
    }
    st.dataframe(pd.DataFrame(opt_data), width="stretch", hide_index=True)
    st.success("**Validation RMSE = 0.0195** (기본 XGBoost 대비 20.7% 개선)")

    st.header("7-2. Knowledge Distillation 수식")
    st.latex(r"y_{distilled} = \alpha \cdot y_{true} + (1 - \alpha) \cdot y_{teacher\_pred}")
    st.markdown("- α=0.5: Hard/Soft Label 동일 비중 (기본 설정)")

    st.header("7-3. KD 성능 비교")
    kd_data = {
        "모델": ["Teacher (XGBoost)", "Base Student (LightGBM)", "Distilled Student (LightGBM+KD)"],
        "시나리오": ["S4 (17 피처)", "S2 (9 피처)", "S2 (9 피처)"],
        "RMSE": [0.0260, 0.0363, 0.0363],
        "R²": ["~0.49", "0.013", "0.015"],
    }
    st.dataframe(pd.DataFrame(kd_data), width="stretch", hide_index=True)
    st.metric("KD 성능 향상률", "0.13%", help="거의 무의미한 수준")

    st.header("7-4. KD 효과가 미미한 원인")
    st.error("""
    **정보 격차 (Information Gap):**
    - Teacher 핵심 정보: `v25_voltage` (|r|=0.600) ← **S2에 없음!**
    - Student 최대 정보: `v42_impedance` (|r|=0.177)
    - 격차: 0.600 vs 0.177 → Teacher 지식 재구성 불가

    KD는 Teacher-Student 피처 간 **정보 채널**이 존재할 때만 유효합니다.
    """)

# ════════════════════════════════════════════════════════════════════
#  Phase 8: 성능 고도화
# ════════════════════════════════════════════════════════════════════
elif page == "Phase 8: 성능 고도화":
    st.title("Phase 8: 성능 고도화 실험")
    st.markdown("Phase 7의 한계를 극복하기 위해 **5가지 체계적 실험**을 수행했습니다.")
    st.markdown("---")

    import pandas as pd

    exp_info = {
        "실험": ["8-1", "8-2", "8-3", "8-4", "8-5"],
        "방법": ["S3 Student KD", "α 값 그리드 탐색", "Feature Augmentation",
                 "Semi-supervised Transfer", "Optuna Student + 종합 비교"],
        "목표": ["3.6V 추가 효과 검증", "최적 블렌딩 비율", "v25_voltage 간접 예측",
                 "S4 부분 측정 + KD", "Student 자체 최적화"],
    }
    st.dataframe(pd.DataFrame(exp_info), width="stretch", hide_index=True)

    # ── 8-1 ──
    st.header("8-1. S3 Student KD (3.6V 추가 효과)")
    exp1_data = {
        "모델": ["Teacher (S4)", "S2 Base (9 피처)", "S3 Base (12 피처)", "S3 + KD α=0.5"],
        "RMSE": [0.0245, 0.0363, 0.0348, 0.0351],
        "R²": [0.550, 0.013, 0.092, 0.081],
        "vs S2 Base": ["-", "Baseline", "-4.1% ✅", "-3.5%"],
    }
    st.dataframe(pd.DataFrame(exp1_data), width="stretch", hide_index=True)
    st.info("S2→S3 전환으로 **4.1% 개선** (3.6V 추가 효과). KD 적용 시 오히려 +0.64% 악화")

    # ── 8-2 ──
    st.header("8-2. α 값 그리드 탐색")
    show_img("phase8_alpha_search.png", "S2/S3에 대한 α 값(0.1~0.9) 그리드 탐색 결과")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**S2 결과:**")
        s2_alpha = {
            "α": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "RMSE": [0.0364, 0.0365, 0.0363, 0.0365, 0.0364, 0.0365, 0.0365, 0.0365, 0.0366],
        }
        st.dataframe(pd.DataFrame(s2_alpha), width="stretch", hide_index=True)
        st.metric("S2 최적", "α=0.3, RMSE=0.0363")
    with col2:
        st.markdown("**S3 결과:**")
        s3_alpha = {
            "α": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "RMSE": [0.0352, 0.0348, 0.0352, 0.0349, 0.0351, 0.0350, 0.0352, 0.0350, 0.0351],
        }
        st.dataframe(pd.DataFrame(s3_alpha), width="stretch", hide_index=True)
        st.metric("S3 최적", "α=0.2, RMSE=0.0348")
    st.warning("두 시나리오 모두 α에 대한 민감도 극히 낮음 (RMSE 변화 < 0.1%)")

    # ── 8-3 ──
    st.header("8-3. Feature Augmentation (v25_voltage 간접 예측)")
    exp3_data = {
        "모델": ["Teacher (S4)", "S2 Base (9 피처)", "Aug-S2 Base (11 피처)", "Aug-S2 + KD (11 피처)"],
        "RMSE": [0.0245, 0.0363, 0.0360, 0.0360],
        "R²": [0.550, 0.013, 0.032, 0.029],
        "효과": ["-", "Baseline", "-0.98%", "-0.80%"],
    }
    st.dataframe(pd.DataFrame(exp3_data), width="stretch", hide_index=True)
    st.error("v25_voltage 예측 R² ≈ 0 → S2 피처와 v25_voltage 사이에 **정보적 연결 없음**. Feature Augmentation 실패.")

    # ── 8-4 ──
    st.header("8-4. Semi-supervised Transfer")
    show_img("phase8_semi_supervised.png", "좌: 성능 변화, 우: 비용 효율 분석")

    semi_data = {
        "S4 비율": ["10%", "30%", "50%", "70%", "100%"],
        "Teacher RMSE": [0.0314, 0.0271, 0.0259, 0.0258, 0.0245],
        "Student RMSE": [0.0359, 0.0357, 0.0366, 0.0368, 0.0360],
        "S2 대비": ["+1.05%", "+1.59% ⭐", "-0.82%", "-1.29%", "+0.82%"],
    }
    st.dataframe(pd.DataFrame(semi_data), width="stretch", hide_index=True)
    st.info("30%에서 최적. 강한 Teacher의 Soft Label이 Student에게 오히려 혼란 야기 (비단조적)")

    # ── 8-5 ──
    st.header("8-5. 전체 실험 종합 비교")
    show_img("phase8_comparison.png", "Phase 7-8 전체 실험 RMSE(좌) / R²(우) 비교")

    final_data = {
        "모델": ["Teacher S4 (Optuna)", "S3 + KD (α=0.2)", "S3 Base (LightGBM)",
                 "Semi-S4 (30%)", "Aug-S2 Base", "Optuna Student",
                 "Aug-S2 + KD", "S2 + KD (α=0.3)", "S2 Base (LightGBM)"],
        "피처 수": [17, 12, 12, 17, 11, 11, 11, 9, 9],
        "RMSE": [0.0245, 0.0348, 0.0348, 0.0357, 0.0360, 0.0360, 0.0360, 0.0363, 0.0363],
        "R²": [0.5502, 0.0949, 0.0923, 0.0440, 0.0315, 0.0301, 0.0290, 0.0117, 0.0127],
        "개선률(%)": [32.50, 4.27, 4.11, 1.59, 0.98, 0.90, 0.80, 0.05, 0.00],
    }
    df_final = pd.DataFrame(final_data)
    st.dataframe(
        df_final.style.highlight_min(subset=["RMSE"], color="#d4edda")
                      .highlight_max(subset=["R²"], color="#d4edda"),
        width="stretch", hide_index=True,
    )

    st.success("""
    **🏆 Best Student Model: S3 + KD (α=0.2)**
    - RMSE = 0.0348, R² = 0.0949
    - S2 Base 대비 개선률: **4.27%**
    - Teacher 대비 차이: 0.0103 (42.1%)
    """)

# ════════════════════════════════════════════════════════════════════
#  Phase 9: S1 집중 개선
# ════════════════════════════════════════════════════════════════════
elif page == "Phase 9: S1 집중 개선":
    st.title("Phase 9: S1 집중 개선 — 최소 피처로 최대 성능")
    st.markdown("가장 저비용 시나리오인 **S1(initial_voltage, initial_impedance, ocv_deviation)** 3개 피처만으로 성능을 극대화합니다.")
    st.markdown("---")

    import pandas as pd

    # ── 배경 ──
    st.header("📋 실험 배경")
    col1, col2 = st.columns(2)
    col1.metric("S1 기존 최고", "RMSE 0.0355", help="Phase 5 LinearReg")
    col2.metric("S1 피처 수", "3개", help="initial_voltage, initial_impedance, ocv_deviation")
    st.warning("Phase 8까지의 모든 개선 실험은 S2/S3에 집중. **S1 전용 최적화는 한 번도 수행되지 않았습니다.**")

    # ── 실험 계획 ──
    st.header("🧪 실험 계획 (6가지)")
    exp_plan = {
        "실험": ["9-1", "9-2", "9-3", "9-4", "9-5", "9-6"],
        "방법": [
            "다항식/비선형 피처 확장 (3→15)",
            "S1 전용 Optuna 최적화 (LGB/XGB/GB/SVR)",
            "Stacking 앙상블 (7 이기종 모델)",
            "확장 피처 + Teacher KD (α+HP 동시 탐색)",
            "Teacher-Guided Binning (클러스터별 지식 매핑)",
            "전체 최적 앙상블 (Super-Stacking + Blend)",
        ],
        "기대 효과": [
            "비선형 관계 포착",
            "모델 잠재력 극대화",
            "다양성 기반 성능 향상",
            "정보 전이 채널 확대",
            "Teacher 지식 → S1 공간 매핑",
            "Best of All 결합",
        ],
    }
    st.dataframe(pd.DataFrame(exp_plan), width="stretch", hide_index=True)

    # ── 9-1: 피처 확장 ──
    st.header("9-1. 확장 피처 엔지니어링")
    st.markdown("""
    **3개 → 15개 확장 피처 선택** (Mutual Information 기준 상위 15개)
    
    변환 종류: Polynomial(degree 3), log, sqrt, inverse, square, cube, 교호작용 비율·곱·차
    """)
    top_feat = {
        "순위": list(range(1, 6)),
        "피처": ["impedance × ocv_deviation²", "voltage × impedance × ocv_deviation",
                 "v_z_o (3-way product)", "impedance × ocv_deviation", "z_times_o"],
        "MI Score": [0.0985, 0.0874, 0.0870, 0.0805, 0.0800],
        "|Pearson|": [0.2697, 0.2409, 0.2409, 0.2436, 0.2436],
    }
    st.dataframe(pd.DataFrame(top_feat), width="stretch", hide_index=True)
    st.success("원래 max |r|=0.15이던 S1이 비선형 변환으로 **|r|=0.27까지 상승** — 80% 향상")

    # ── 9-2: Optuna 결과 ──
    st.header("9-2. S1 전용 Optuna 최적화")
    optuna_res = {
        "모델": ["LGB-Ext", "XGB-Ext", "GB-Ext", "SVR-Ext"],
        "CV RMSE": [0.0300, 0.0299, 0.0301, 0.0299],
        "Test RMSE": [0.0349, 0.0350, 0.0352, 0.0352],
        "R²": [0.0869, 0.0840, 0.0747, 0.0709],
        "vs S1 Base": ["+1.6%", "+1.4%", "+0.9%", "+0.7%"],
    }
    st.dataframe(pd.DataFrame(optuna_res), width="stretch", hide_index=True)
    st.info("4개 모델 모두 CV RMSE ≈ 0.030으로 수렴 — 확장 피처 효과가 모델 종류를 불문하고 발현")

    # ── 9-3: Stacking ──
    st.header("9-3. Stacking 앙상블")
    st.markdown("""
    **7개 이기종 모델** (LGB + XGB + GB + RF + SVR + Ridge + KNN) OOF Stacking → Ridge Meta-Learner
    """)
    stack_weights = {
        "Base 모델": ["SVR", "XGB", "LGB", "GB", "Ridge", "RF", "KNN"],
        "Meta 가중치": ["+0.4915", "+0.3394", "+0.2508", "+0.2198", "+0.0905", "-0.0676", "-0.1534"],
        "Test RMSE": [0.0352, 0.0350, 0.0349, 0.0352, 0.0355, 0.0366, 0.0399],
    }
    st.dataframe(pd.DataFrame(stack_weights), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    c1.metric("Stacking RMSE", "0.0346", "-2.5% vs S1 Base")
    c2.metric("Stacking R²", "0.1028", "+87% vs S1 Base (0.055)")

    # ── 9-4: KD ──
    st.header("9-4. 확장 S1 + Teacher KD")
    st.markdown("Optuna로 α(Hard/Soft 블렌딩)와 LightGBM HP를 동시 탐색 (80 trials)")
    c1, c2, c3 = st.columns(3)
    c1.metric("최적 α", "0.693")
    c2.metric("Test RMSE", "0.0348")
    c3.metric("vs S1 Base", "+1.8%")
    st.info("확장 피처와 결합해도 KD 효과는 +1.8%에 그침 — S1 공간에서 Teacher 지식 흡수 채널 부족")

    # ── 9-5: Binning + Super-Stacking ──
    st.header("9-5. Teacher-Guided Binning")
    st.markdown("S1 피처 공간을 KMeans 클러스터링 → 각 클러스터의 Teacher 예측 평균을 새 피처로 활용")
    bin_res = {
        "K (클러스터)": [5, 10, 15, 20, 30, 50],
        "Test RMSE": [0.0349, 0.0349, 0.0354, 0.0353, 0.0350, 0.0352],
        "R²": [0.087, 0.089, 0.061, 0.067, 0.081, 0.072],
    }
    st.dataframe(pd.DataFrame(bin_res), width="stretch", hide_index=True)
    st.markdown("최적 K=10, Test RMSE=0.0349")

    # ── 9-6: 종합 ──
    st.header("📊 Phase 9 종합 결과")
    show_img("phase9_s1_comparison.png", "Phase 9: S1 기반 모델 RMSE 비교 & 예측 산점도")

    final_res = {
        "순위": ["─", 1, 2, "─", 3, 4, 5, 6, 7, 8, 9, 10, "─"],
        "모델": [
            "(비교) Teacher S4",
            "S1 Stacking (7-model)", "S1 Super-Stacking",
            "(비교) S3 Base",
            "S1 ExtFeat + KD", "S1 Teacher-Binning",
            "S1 LGB-Ext (Optuna)", "S1 XGB-Ext (Optuna)",
            "S1 Prediction Blend", "S1 GB-Ext (Optuna)",
            "S1 SVR-Ext (Optuna)", "S1 Base (LinearReg)",
            "(비교) S2 Base",
        ],
        "피처": ["17", "15", "15", "12", "15", "18", "15", "15", "-", "15", "15", "3", "9"],
        "RMSE": ["0.0245", "0.0346", "0.0346", "0.0348", "0.0348", "0.0349",
                 "0.0349", "0.0350", "0.0350", "0.0352", "0.0352", "0.0355", "0.0363"],
        "R²": ["0.5502", "0.1028", "0.1023", "0.0923", "0.0915", "0.0890",
               "0.0869", "0.0840", "0.0811", "0.0747", "0.0709", "0.0550", "0.0127"],
        "vs S1 Base": ["+30.9%", "+2.5%", "+2.4%", "+1.9%", "+1.8%", "+1.7%",
                       "+1.6%", "+1.4%", "+1.3%", "+0.9%", "+0.7%", "0.0%", "-2.3%"],
    }
    st.dataframe(pd.DataFrame(final_res), width="stretch", hide_index=True)

    # ── 핵심 발견 ──
    st.header("🏆 Phase 9 핵심 발견")
    st.success("""
    **S1(3피처) Stacking이 S3(12피처)를 추월: RMSE 0.0346 < 0.0348**
    
    - 단 3개 입고시점 데이터만으로 3.6V 측정 추가한 S3보다 우수
    - S2(9피처, 0.0363) 대비 **4.7% 우위**
    - 추가 측정 없이 입고 시점 데이터만으로 최선의 예측 가능
    """)

    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    #### 확장 피처의 힘
    - `impedance × ocv_deviation²` 핵심
    - 원래 max |r|=0.15 → 0.27 (80%↑)
    - 3개 → 15개 확장
    """)
    c2.markdown("""
    #### Stacking 효과
    - 7종 이기종 모델 결합
    - SVR+XGB+LGB 주력
    - 단일 모델 대비 일관 우수
    """)
    c3.markdown("""
    #### KD 한계 확인
    - 확장 피처와 결합해도 +1.8%
    - S1 공간의 정보 한계
    - Stacking이 더 효과적
    """)

# ════════════════════════════════════════════════════════════════════
#  Phase 10: KD 정보채널 보완
# ════════════════════════════════════════════════════════════════════
elif page == "Phase 10: KD 정보채널 보완":
    st.title("Phase 10: KD 정보채널 보완 — 관계·순위·경로 기반 KD")
    st.markdown("KD가 효과적이려면 Teacher-Student 피처 간 **정보 채널**이 필요합니다. S2/S3 피처에서 v25_voltage를 재구성할 수 없어 기존 Output KD가 한계를 보였으므로, **대안적 지식 전이 경로**를 탐색합니다.")
    st.markdown("---")

    import pandas as pd

    # ── 문제 정의 ──
    st.header("📋 문제 정의: 정보 채널 부재")
    col1, col2 = st.columns(2)
    with col1:
        st.error("""
        **기존 Output KD의 한계**
        - Teacher 핵심: `v25_voltage` (|r|=0.600)
        - Student 최대: `v42_impedance` (|r|=0.177)
        - S2/S3 피처로 v25_voltage 재구성 불가
        - Soft Label만으로는 채널 부재 시 노이즈
        """)
    with col2:
        st.success("""
        **Phase 10 보완 전략**
        - **RKD**: 샘플 간 거리/유사도 관계 전이
        - **LambdaKD**: Teacher 순위 구조 보존 전이
        - **Progressive Bridge**: 중간 피처 경로 구성
        """)

    # ── 실험 계획 ──
    st.header("🧪 실험 계획")
    exp_plan = {
        "실험": ["10-1", "10-2", "10-3", "10-4"],
        "방법": [
            "RKD (Relational KD) — Anchor 기반 거리 피처",
            "LambdaKD (Ranking-based KD) — Optuna 최적화",
            "Progressive Feature Bridge — S2→v36→v25→capacity",
            "전체 종합 비교 & 시각화",
        ],
        "핵심 아이디어": [
            "Teacher 예측의 상대적 관계 구조를 피처화",
            "순위 보존도(Spearman)를 목적함수에 통합",
            "측정 단계별 중간 피처를 체인으로 예측",
            "기존 KD vs Phase 10 방법 비교",
        ],
    }
    st.dataframe(pd.DataFrame(exp_plan), width="stretch", hide_index=True)

    # ── 10-1: RKD ──
    st.header("10-1. RKD (Relational Knowledge Distillation)")
    st.markdown("""
    **2단계 학습**: Stage 1(직접 예측) → Stage 2(Teacher 예측 기반 **Anchor 거리 피처** 추가)
    - 20개 Anchor Point (Teacher 예측의 균등 분위수)
    - 각 샘플과 Anchor 간 거리/유사도를 새 피처로 생성
    - OOF 기반 누수 방지 평가
    """)

    rkd_data = {
        "시나리오": ["S1", "S2", "S3"],
        "Base RMSE": [0.0374, 0.0371, 0.0356],
        "RKD Ensemble RMSE": [0.0368, 0.0366, 0.0355],
        "개선률": ["+1.68%", "+1.34%", "+0.30%"],
        "Ensemble R²": [0.0170, 0.0023, 0.0572],
    }
    st.dataframe(pd.DataFrame(rkd_data), width="stretch", hide_index=True)
    st.info("RKD는 소폭 개선. Stage 2가 Stage 1 예측에 의존하므로 Test 시 관계 피처 품질이 저하됩니다.")

    # ── 10-2: LambdaKD ──
    st.header("10-2. LambdaKD (Ranking-based KD)")
    st.markdown("""
    **핵심 아이디어**: Teacher의 **순위 구조**를 Student에 전이
    - 목적함수: `RMSE − α · Spearman(pred, teacher_rank)`
    - Optuna로 λ(Teacher 신호 비율) + α(순위 가중) + HP 동시 최적화 (60 trials/scenario)
    - 3-way 앙상블: Direct + Lambda + RankMap 결합
    """)

    lambda_data = {
        "시나리오": ["S1", "S2", "S3"],
        "Base RMSE": [0.0374, 0.0371, 0.0356],
        "LambdaKD RMSE": ["**0.0353**", "0.0358", "**0.0347**"],
        "개선률": ["**+5.64%**", "+3.32%", "+2.49%"],
        "Spearman ρ 변화": ["0.26 → 0.32 (+24%)", "0.28 → 0.34", "0.35 → 0.40"],
        "최적 λ": [0.70, 0.41, 0.63],
    }
    st.dataframe(pd.DataFrame(lambda_data), width="stretch", hide_index=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("S1 LambdaKD", "0.0353", "+5.64% vs Base")
    col2.metric("S3 LambdaKD", "0.0347", "+2.49% vs Base")
    col3.metric("Spearman 향상", "+24%", "0.26→0.32 (S1)")

    st.success("""
    **LambdaKD가 가장 효과적인 KD 보완 방법!**
    - 예측값이 아닌 **순위 구조**를 전이하므로 피처 채널 부재에도 유효
    - S1에서 +5.64% 개선은 Phase 7-8의 모든 KD 실험 중 최대
    - Spearman ρ +24% 향상 → 배터리 등급 판별 실무에 중요
    """)

    # ── 10-3: Progressive Bridge ──
    st.header("10-3. Progressive Feature Bridge")
    st.markdown("""
    **단계적 피처 예측 체인**: S2 피처 → v36_hat → v25_hat → capacity
    - Stage 1: S2 피처로 v36_voltage, v36_impedance 예측
    - Stage 2: S2 + v36_hat으로 v25_voltage, v25_impedance 예측
    - Stage 3: 원래 피처 + 예측 피처로 최종 capacity 예측
    """)

    bridge_data = {
        "모델": ["Bridge-LGB", "Bridge-KD", "Bridge-XGB", "S1Ext-Bridge"],
        "특성": ["S2 + Bridge 피처", "S2 + Bridge + KD α=0.5", "S2 + Bridge (XGBoost)", "S1 확장 + Bridge"],
        "RMSE": [0.0355, 0.0357, 0.0372, 0.0367],
        "R²": [0.057, 0.049, -0.036, -0.012],
        "vs S2 Base": ["+2.15%", "+1.60%", "-2.63%", "-1.18%"],
    }
    st.dataframe(pd.DataFrame(bridge_data), width="stretch", hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **v25_voltage 예측 분석:**
        - Direct 예측: r = 0.211
        - Bridge 예측: r = 0.221 (+0.01)
        - v25_impedance: r = 0.678 (잘 예측됨)
        """)
    with col2:
        st.markdown("""
        **핵심 발견:**
        - 단계적 경로가 직접 예측보다 약간 우수
        - v25_voltage 예측 정확도 제한이 병목
        - Bridge-LGB가 S2 Base 대비 +2.15% 개선
        """)

    # ── 10-4: 종합 비교 ──
    st.header("📊 Phase 10 종합 비교")
    show_img("phase10_kd_channel.png", "Phase 10: RMSE 비교, 순위 보존도, KD 유형별 개선율, 정보 채널 흐름")

    st.subheader("KD 유형별 S2 Base 대비 개선율")
    kd_compare = {
        "KD 유형": ["Output KD (Phase 7)", "RKD (Phase 10-1)", "LambdaKD (Phase 10-2)", "3-way Ensemble (Phase 10-2)", "Bridge (Phase 10-3)"],
        "S2 대비 개선율": ["+0.13%", "-0.64%", "+1.37%", "+0.75%", "**+2.15%**"],
        "전이 대상": ["예측값 (Soft Label)", "샘플 간 거리/유사도", "순위 구조 (Spearman)", "순위+직접+매핑 결합", "중간 피처 경로"],
        "효과": ["제한적", "제한적", "**효과적**", "보통", "**효과적**"],
    }
    st.dataframe(pd.DataFrame(kd_compare), width="stretch", hide_index=True)

    st.subheader("순위 보존도 비교 (Spearman ρ)")
    rank_data = {
        "모델": ["Teacher S4", "S3 LambdaKD", "S1 Stacking", "S3 RKD", "S1 LambdaKD", "S1 Base"],
        "Spearman ρ": [0.7458, 0.3980, 0.3750, 0.3530, 0.3200, 0.2600],
        "RMSE": [0.0245, 0.0347, 0.0346, 0.0355, 0.0353, 0.0374],
    }
    st.dataframe(pd.DataFrame(rank_data), width="stretch", hide_index=True)
    st.info("LambdaKD는 RMSE뿐 아니라 **순위 보존도(Spearman)에서도 일관 우수** — 배터리 등급 분류에 유리")

    # ── 핵심 발견 ──
    st.header("🏆 Phase 10 핵심 발견")

    st.markdown("""
    <div style="background:#3498db22; border:2px solid #3498db; border-radius:10px;
                padding:16px; margin-bottom:12px;">
        <b style="font-size:1.1rem;">1. LambdaKD = 최고의 KD 보완 방법</b><br>
        S1 +5.64% (0.0374→0.0353), S3 +2.49% (0.0356→0.0347)<br>
        Spearman ρ +24% (0.26→0.32) — 순위 구조 전이 성공
    </div>
    <div style="background:#27ae6022; border:2px solid #27ae60; border-radius:10px;
                padding:16px; margin-bottom:12px;">
        <b style="font-size:1.1rem;">2. Progressive Bridge = 정보 경로 확보</b><br>
        S2→v36→v25 체인으로 Bridge-LGB RMSE 0.0355 (S2 대비 +2.15%)<br>
        v25_impedance r=0.678 — 부분적 정보 채널 확인
    </div>
    <div style="background:#e74c3c22; border:2px solid #e74c3c; border-radius:10px;
                padding:16px;">
        <b style="font-size:1.1rem;">3. 정보 이론적 한계</b><br>
        Teacher S4 RMSE 0.0245 vs Phase 10 최고 RMSE 0.0347<br>
        <b>잔여 격차 41.5%</b> — v25_voltage 직접 측정 없이의 근본 한계
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  📊 종합 결론
# ════════════════════════════════════════════════════════════════════
elif page == "📊 종합 결론":
    st.title("종합 결론 및 실무 제언")
    st.markdown("---")

    import pandas as pd

    # ── 핵심 결론 ──
    st.header("🔑 핵심 결론 1: 정보량의 벽 (Information Bottleneck)")
    st.markdown("""
    배터리 용량 예측의 성능은 **"2.5V 방전 종지 전압(v25_voltage)을 측정했는가 여부"**로 결정됩니다.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("S4 (v25 포함)", "RMSE 0.024", help="17 피처")
    with col2:
        st.metric("S2/S3 (v25 미포함)", "RMSE 0.035", "~40% 격차", delta_color="inverse")

    st.error("""
    - `v25_voltage` 상관 |r|=0.600 vs 다른 피처 최대 |r|=0.177 → **정보량 격차가 압도적**
    - KD, α 튜닝, Feature Augmentation, Optuna, Semi-supervised 모두 시도 → **1~4% 미미한 개선**
    - 이는 **데이터의 물리적 한계**이며, 알고리즘으로 해결할 수 없는 문제
    """)

    st.header("🔑 핵심 결론 2: S1 확장피처 + Stacking으로 S3 추월")
    st.success("""
    **Phase 9 결과**: S1(3피처) Stacking RMSE **0.0346** < S3(12피처) RMSE **0.0348**
    
    - 비선형 확장 피처(3→15개) + 7종 이기종 앙상블로 S3 수준 달성
    - 추가 측정 없이 입고 시점 데이터만으로 최선의 예측 가능
    - S2(9피처, 0.0363) 대비 **4.7% 우위**
    """)

    st.header("🔑 핵심 결론 3: KD 정보채널 보완 (Phase 10)")
    st.warning("KD가 효과적이려면 Teacher-Student 피처 간 **정보 채널**이 필요. Phase 10에서 순위 기반(LambdaKD)·경로 기반(Bridge) KD로 보완.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **LambdaKD (순위 기반 KD):**
        - S1: 0.0374 → **0.0353** (+5.64%)
        - S3: 0.0356 → **0.0347** (+2.49%)
        - Spearman ρ: 0.26 → 0.32 (**+24%**)
        - Teacher의 **순위 구조**를 전이하여 채널 부재 문제 우회
        """)
    with col2:
        st.markdown("""
        **Progressive Bridge (경로 기반 KD):**
        - S2→v36→v25 단계적 예측 체인
        - Bridge-LGB: RMSE 0.0355 (S2 대비 **+2.15%**)
        - v25_impedance r=0.678 — 부분적 정보 경로 확인
        - **정보 이론적 한계: 41.5% 격차 잔존**
        """)

    # ── 실무 제언 ──
    st.header("📋 실무 제언: 측정 전략별 경제성 분석")
    strategy_data = {
        "전략": ["A. 정밀 예측", "B. 비용+정확도 ✨", "C. 비용+순위 보존", "D. S3+순위 KD", "E. S2+경로 보완", "F. 기본 예측"],
        "측정 범위": ["S4 (2.5V 완전 방전)", "S1 Stacking (입고 데이터만)", "S1 LambdaKD (입고 데이터)", "S3 LambdaKD (3.6V 측정)", "Bridge-LGB (4.2V 측정)", "S3 Base (3.6V 측정)"],
        "RMSE (Ah)": ["0.0245", "0.0346", "0.0353", "0.0347", "0.0355", "0.0348"],
        "Spearman ρ": ["0.746", "0.375", "0.320", "0.398", "-", "0.350"],
        "소요 시간": ["~2시간+", "입고 즉시", "입고 즉시", "~1시간", "~30분", "~1시간"],
        "추천 대상": ["고가 셀, 품질 인증", "대량 초기 분류(최적)", "등급 판별 중시", "S3 측정 가능 시 최고", "S2만으로 성능 향상 원할 시", "기본 스크리닝"],
    }
    st.dataframe(pd.DataFrame(strategy_data), width="stretch", hide_index=True)

    # ── 구체적 권고 ──
    st.header("🎯 구체적 권고사항")
    tab1, tab2, tab3 = st.tabs(["단기 적용", "중기 적용", "장기 개선"])

    with tab1:
        st.markdown("""
        ### 즉시 가능
        - **S4 기반 Optuna XGBoost 모델 배포**
        - RMSE 0.0245 (공칭 5.0Ah 대비 0.49% 오차)
        - 최적 하이퍼파라미터: n=653, depth=7, lr=0.182, λ=3.887
        - 17개 피처 입력 → 용량 예측값 출력
        """)

    with tab2:
        st.markdown("""
        ### 공정 최적화
        - **측정 프로토콜 이원화:**
          - 전수: S2(4.2V 충전) 측정으로 빠른 1차 스크리닝
          - 표본: S4(2.5V 방전)으로 정밀 2차 검증 (10~30%)
          - 1차에서 '의심 셀' (예측 용량 < 임계값)만 2차 정밀 측정
        """)

    with tab3:
        st.markdown("""
        ### 장기 개선 방향
        - **방전 프로파일 확장**: V-t 프로파일 전체를 피처로 활용
        - **EIS(전기화학 임피던스 분광법)**: 다주파수 EIS 스펙트럼 활용
        - **온도 보정**: 측정 시 온도 데이터로 임피던스/전압 보정
        - **배치 간 전이학습**: Domain Adaptation으로 모델 범용성 확보
        """)

    # ── 최종 요약 ──
    st.header("📊 최종 요약 다이어그램")

    # -- 시나리오별 카드 --
    st.markdown("##### 측정 비용 증가 →")
    s_cols = st.columns(4)
    _scenarios = [
        ("S1 · Initial", "3→15 확장", 0.0346, 0.103, "#ff6b6b"),
        ("S2 · +4.2V", "9 피처", 0.0363, 0.013, "#ff9f43"),
        ("S3 · +3.6V", "12 피처", 0.0348, 0.092, "#feca57"),
        ("S4 · +2.5V", "17 피처", 0.0245, 0.550, "#1dd1a1"),
    ]
    for col, (name, feat, rmse, r2, color) in zip(s_cols, _scenarios):
        col.markdown(
            f"""
            <div style="background:{color}22; border-left:5px solid {color};
                        border-radius:8px; padding:14px 12px; text-align:center;">
                <b style="font-size:1.05rem;">{name}</b><br>
                <span style="color:gray;">{feat}</span>
                <hr style="margin:6px 0;">
                <span style="font-size:1.3rem; font-weight:700;">RMSE {rmse}</span><br>
                <span style="font-size:0.95rem;">R² = {r2}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -- 화살표 + 구간 설명 --
    a1, a2 = st.columns([3, 1])
    a1.markdown(
        '<div style="text-align:center; margin:8px 0; font-size:1.1rem;">'
        '← ─ <b>S1≈S3 (Stacking으로 추월!)</b> ─ → ← <b>S2 약간 열세</b> →</div>',
        unsafe_allow_html=True,
    )
    a2.markdown(
        '<div style="text-align:center; margin:8px 0; font-size:1.1rem; color:#1dd1a1; font-weight:700;">'
        '⬆ 급격한<br>성능 향상</div>',
        unsafe_allow_html=True,
    )

    # -- RMSE 바 차트 --
    import plotly.graph_objects as go

    fig = go.Figure()
    labels = ["S1 Stacking", "S2 (+4.2V)", "S3 (+3.6V)", "S4 (+2.5V)"]
    rmse_vals = [0.0346, 0.0363, 0.0348, 0.0245]
    r2_vals = [0.103, 0.013, 0.092, 0.550]
    colors = ["#ff6b6b", "#ff9f43", "#feca57", "#1dd1a1"]

    fig.add_trace(go.Bar(
        x=labels, y=rmse_vals, name="RMSE",
        marker_color=colors, text=[f"{v:.3f}" for v in rmse_vals],
        textposition="outside", yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=r2_vals, name="R²",
        mode="lines+markers+text", text=[f"{v:.3f}" for v in r2_vals],
        textposition="top center", yaxis="y2",
        line=dict(color="#5f27cd", width=3), marker=dict(size=12),
    ))
    fig.update_layout(
        yaxis=dict(title="RMSE (Ah)", range=[0, 0.045], showgrid=True),
        yaxis2=dict(title="R²", overlaying="y", side="right", range=[-0.05, 0.7]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=380, margin=dict(t=50, b=30),
        title="시나리오별 RMSE & R² 비교",
    )
    st.plotly_chart(fig, width="stretch")

    # -- 핵심 메시지 --
    st.markdown(
        """
        <div style="background:#1dd1a122; border:2px solid #1dd1a1; border-radius:10px;
                    padding:16px; text-align:center; margin-top:4px;">
            <span style="font-size:1.3rem;">⭐</span>
            <b style="font-size:1.15rem;"> 핵심 발견: S1 Stacking(0.0346)이 S3 추월 | LambdaKD로 순위 보존 +24%</b><br>
            <span style="font-size:1.05rem;">확장 피처 + 앙상블로 <b>추가 측정 없이</b> 최선의 예측 | LambdaKD로 정보채널 보완 | 궁극의 성능은 S4(<code>v25_voltage</code>)에 있습니다.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -- Phase 10 KD 보완 요약 --
    st.header("📡 Phase 10: KD 정보채널 보완 요약")
    show_img("phase10_kd_channel.png", "Phase 10: RMSE 비교, 순위 보존도, KD 유형별 개선율, 정보 채널 흐름")

    p10_cols = st.columns(3)
    p10_cols[0].metric("LambdaKD (S1)", "0.0353", "+5.64% vs Base")
    p10_cols[1].metric("LambdaKD (S3)", "0.0347", "+2.49% vs Base")
    p10_cols[2].metric("Bridge-LGB (S2)", "0.0355", "+2.15% vs S2 Base")

    # ── 생성 이미지 목록 ──
    with st.expander("📂 생성된 시각화 파일 목록 (21개)"):
        img_list = {
            "파일명": [
                "eda_01_distributions.png", "eda_02_boxplots.png", "eda_03_correlation.png",
                "eda_04_scatter.png", "eda_05_outliers.png", "eda_06_feature_groups.png",
                "eda_07_pairplot.png", "phase2_derived_vs_capacity.png", "phase2_correlation_heatmap.png",
                "phase3_cleaned_correlation.png", "phase3_derived_scatter.png", "phase3_vif_comparison.png",
                "phase4_data_split.png", "phase5_rmse_barplot.png", "phase5_r2_heatmap.png",
                "phase5_feature_importance.png", "phase8_alpha_search.png", "phase8_semi_supervised.png",
                "phase8_comparison.png", "phase9_s1_comparison.png", "phase10_kd_channel.png",
            ],
            "Phase": ["1","1","1","1","1","1","1","2","2","3","3","3","4","5","5","5","8-2","8-4","8-5","9","10"],
            "내용": [
                "히스토그램+KDE", "박스플롯(IQR)", "Pearson/Spearman 히트맵",
                "피처 vs Capacity 산점도", "이상치 하이라이트", "측정 구간별 비교",
                "임피던스 Pairplot", "파생변수 산점도", "파생변수 상관 히트맵",
                "정제 후 상관 히트맵", "파생변수 산점도", "VIF 비교",
                "분포 시각화", "RMSE 막대그래프", "R² 히트맵",
                "피처 중요도", "α 탐색", "Semi-supervised", "종합 비교",
                "S1 RMSE 비교 & 예측 산점도", "KD 정보채널 보완 종합 비교",
            ],
        }
        st.dataframe(pd.DataFrame(img_list), width="stretch", hide_index=True)

    # ── 분석 환경 ──
    with st.expander("🔧 분석 환경 상세"):
        env_data = {
            "항목": ["Python", "XGBoost", "LightGBM", "Optuna", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn"],
            "버전": ["3.14.2", "3.2.0", "4.6.0", "4.7.0", "1.8.0", "2.3.0", "2.3.0", "3.10.1", "0.13.2"],
        }
        st.dataframe(pd.DataFrame(env_data), width="stretch", hide_index=True)
