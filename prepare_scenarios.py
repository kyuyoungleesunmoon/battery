"""
배터리 성능 검증 데이터 - 4가지 시나리오별 DataFrame 생성 스크립트

시나리오 설명:
  1안: INITIAL 전용 (과거 데이터만으로 예측)
  2안: INITIAL + 4.2V (만충 상태 임피던스 추가)
  3안: INITIAL + 4.2V + 3.6V (공칭 전압 구간 추가)
  4안: INITIAL + 4.2V + 3.6V + 2.5V (전 구간 데이터)

타겟(y): Capacity [Ah]
"""

import pandas as pd
import os

# ============================================================
# 1. 원본 데이터 로드
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.csv")
raw = pd.read_csv(DATA_PATH)

# 원본 컬럼 확인
print("=" * 60)
print("원본 컬럼 목록:")
for i, col in enumerate(raw.columns):
    print(f"  [{i}] '{col}'")
print(f"\n데이터 shape: {raw.shape}")
print("=" * 60)

# ============================================================
# 2. 컬럼 이름 재정의 (명확한 이름으로 매핑)
# ============================================================
# 원본 CSV 헤더 순서:
# 0: 항목
# 1: INITIAL (빈 컬럼)
# 2: voltage (INITIAL voltage)
# 3: ac impedence (INITIAL ac impedance)
# 4: 4.2 [V] (빈 컬럼)
# 5: voltage (4.2V voltage)
# 6: ac impedence (4.2V ac impedance)
# 7: 2.5 [V] (빈 컬럼)
# 8: voltage (2.5V voltage)
# 9: ac impedence (2.5V ac impedance)
# 10: 3.6 [V] (빈 컬럼)
# 11: voltage (3.6V voltage)
# 12: (빈 컬럼)
# 13: ac impedence (3.6V ac impedance)
# 14: Capacity

new_columns = [
    "cell_id",              # 0: 항목 (셀 식별자)
    "initial_label",        # 1: INITIAL (라벨, 빈 값)
    "initial_voltage",      # 2: INITIAL 전압 [V]
    "initial_impedance",    # 3: INITIAL AC 임피던스 [mΩ]
    "v42_label",            # 4: 4.2V (라벨, 빈 값)
    "v42_voltage",          # 5: 4.2V 충전 후 전압 [V]
    "v42_impedance",        # 6: 4.2V AC 임피던스 [mΩ]
    "v25_label",            # 7: 2.5V (라벨, 빈 값)
    "v25_voltage",          # 8: 2.5V 방전 후 전압 [V]
    "v25_impedance",        # 9: 2.5V AC 임피던스 [mΩ]
    "v36_label",            # 10: 3.6V (라벨, 빈 값)
    "v36_voltage",          # 11: 3.6V 전압 [V]
    "v36_empty",            # 12: (빈 컬럼)
    "v36_impedance",        # 13: 3.6V AC 임피던스 [mΩ]
    "capacity",             # 14: Capacity [Ah]
]

raw.columns = new_columns

# 라벨/빈 컬럼 제거
drop_cols = ["initial_label", "v42_label", "v25_label", "v36_label", "v36_empty"]
df = raw.drop(columns=drop_cols)

# 데이터 타입 변환 (수치 컬럼을 float으로)
numeric_cols = [c for c in df.columns if c != "cell_id"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"\n정제 후 컬럼: {list(df.columns)}")
print(f"정제 후 shape: {df.shape}")
print(f"결측치:\n{df.isnull().sum()}")
print()

# ============================================================
# 3. 시나리오별 DataFrame 생성
# ============================================================
TARGET = "capacity"

# --- 1안: INITIAL 전용 ---
scenario1_features = ["initial_voltage", "initial_impedance"]
df_scenario1 = df[["cell_id"] + scenario1_features + [TARGET]].copy()
df_scenario1.columns = ["cell_id"] + scenario1_features + [TARGET]

# --- 2안: INITIAL + 4.2V ---
scenario2_features = scenario1_features + ["v42_voltage", "v42_impedance"]
df_scenario2 = df[["cell_id"] + scenario2_features + [TARGET]].copy()

# --- 3안: INITIAL + 4.2V + 3.6V ---
scenario3_features = scenario2_features + ["v36_voltage", "v36_impedance"]
df_scenario3 = df[["cell_id"] + scenario3_features + [TARGET]].copy()

# --- 4안: INITIAL + 4.2V + 3.6V + 2.5V (전체) ---
scenario4_features = scenario3_features + ["v25_voltage", "v25_impedance"]
df_scenario4 = df[["cell_id"] + scenario4_features + [TARGET]].copy()

# ============================================================
# 4. 결과 확인
# ============================================================
scenarios = {
    "1안 (INITIAL 전용)": df_scenario1,
    "2안 (INITIAL + 4.2V)": df_scenario2,
    "3안 (INITIAL + 4.2V + 3.6V)": df_scenario3,
    "4안 (INITIAL + 4.2V + 3.6V + 2.5V)": df_scenario4,
}

for name, sdf in scenarios.items():
    print("=" * 60)
    print(f"📊 {name}")
    print(f"   Shape: {sdf.shape}")
    print(f"   피처: {[c for c in sdf.columns if c not in ['cell_id', TARGET]]}")
    print(f"   타겟: {TARGET}")
    print(f"\n{sdf.head(3).to_string(index=False)}")
    print()

# ============================================================
# 5. CSV로 저장
# ============================================================
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

save_names = {
    "scenario1_initial.csv": df_scenario1,
    "scenario2_initial_42v.csv": df_scenario2,
    "scenario3_initial_42v_36v.csv": df_scenario3,
    "scenario4_full.csv": df_scenario4,
}

print("=" * 60)
print("📁 CSV 파일 저장:")
for fname, sdf in save_names.items():
    path = os.path.join(OUTPUT_DIR, fname)
    sdf.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"   ✅ {fname} ({sdf.shape[0]}행 × {sdf.shape[1]}열)")

print("\n✨ 모든 시나리오 DataFrame 생성 완료!")

# ============================================================
# 6. 기본 통계 요약
# ============================================================
print("\n" + "=" * 60)
print("📈 시나리오별 Capacity 통계 (동일해야 정상):")
print("-" * 60)
for name, sdf in scenarios.items():
    stats = sdf[TARGET].describe()
    print(f"  {name}:")
    print(f"    평균: {stats['mean']:.4f} Ah | 표준편차: {stats['std']:.4f} Ah")
    print(f"    최소: {stats['min']:.4f} Ah | 최대: {stats['max']:.4f} Ah")
    print()
