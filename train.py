import os
import sys
import argparse
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import VotingRegressor

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    배터리 충방전 데이터 전처리 클래스
    train.ipynb에서 정의된 전처리 로직(파생변수 생성 등)을 통합 수행
    """
    def __init__(self):
        # 학습에 사용할 최종 피처(Phase 3 시나리오 4 기준)
        self.target_features = [
            'voltage_sag', 
            'v25_voltage', 
            'power_loss_indicator', 
            'v42_impedance', 
            'impedance_voltage_product', 
            'health_index',
            'temperature_variation', 
            'impedance_delta_36', 
            'ocv_deviation'
        ]
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, df):
        # df가 리스트/단일파일/전체 DataFrame 등 상황에 따라 다를 수 있으나,
        # 여기서는 이미 병합된 전체/일부 DataFrame을 입력받는다고 가정
        df_processed = df.copy()
        
        # 1. 누락 데이터 처리 (기존 코드 참고)
        missing_cols = ['i12_current', 'i36_current', 'i37_current', 'c40_capacity']
        for col in missing_cols:
            if col in df_processed.columns:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
                
        # 2. 파생 파라미터 그룹 A: Voltage & Current Dynamics
        if 'v25_voltage' in df_processed.columns and 'v37_voltage' in df_processed.columns:
            df_processed['voltage_sag'] = df_processed['v25_voltage'] - df_processed['v37_voltage']
        
        if 'v42_impedance' in df_processed.columns and 'v10_impedance' in df_processed.columns:
            df_processed['impedance_delta_36'] = df_processed['v42_impedance'] - df_processed['v10_impedance']
            
        if 'v25_voltage' in df_processed.columns and 'i25_current' in df_processed.columns:
            df_processed['i25_v25_product'] = df_processed['v25_voltage'] * df_processed['i25_current']
            
        # 3. 파생 파라미터 그룹 B: Energy & Power Proxies
        if 'v42_impedance' in df_processed.columns and 'v42_voltage' in df_processed.columns:
            df_processed['impedance_voltage_product'] = df_processed['v42_impedance'] * df_processed['v42_voltage']
            
        if 'v10_voltage' in df_processed.columns and 'v37_voltage' in df_processed.columns and 'i37_current' in df_processed.columns:
            df_processed['power_loss_indicator'] = (df_processed['v10_voltage'] - df_processed['v37_voltage']) * df_processed['i37_current']
            
        # 4. 파생 파라미터 그룹 C: Battery State Indicators
        if 't25_temperature' in df_processed.columns and 'v25_voltage' in df_processed.columns:
            df_processed['health_index'] = df_processed['t25_temperature'] / (df_processed['v25_voltage'] + 1e-6)
            
        if 't36_temperature' in df_processed.columns and 't10_temperature' in df_processed.columns:
            df_processed['temperature_variation'] = df_processed['t36_temperature'] - df_processed['t10_temperature']
            
        # OCV Deviation 근사치 추가
        if 'v42_voltage' in df_processed.columns and 'v10_voltage' in df_processed.columns:
            df_processed['ocv_deviation'] = df_processed['v42_voltage'] - df_processed['v10_voltage']
            
        return df_processed
        
    def fit_transform(self, df, y=None):
        return self.transform(df)

def main():
    parser = argparse.ArgumentParser(description='Battery Capacity Prediction - Train Model')
    parser.add_argument('--data_path', type=str, default='./dataset/train/', help='학습 데이터 폴더 경로')
    parser.add_argument('--model_path', type=str, default='./best_model.pkl', help='저장할 모델 파일 경로')
    args = parser.parse_args()

    print(f"[Info] 데이터 로딩을 시작합니다: {args.data_path}")
    
    # 1. 데이터 로딩 (모든 csv 파일 병합)
    data_list = []
    if os.path.isdir(args.data_path):
        for root, dirs, files in os.walk(args.data_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    df_temp = pd.read_csv(file_path)
                    data_list.append(df_temp)
    elif os.path.isfile(args.data_path) and args.data_path.endswith('.csv'):
        data_list.append(pd.read_csv(args.data_path))
        
    if not data_list:
        print(f"[Error] 지정된 경로에 유효한 CSV 데이터가 없습니다: {args.data_path}")
        sys.exit(1)
        
    df_raw = pd.concat(data_list, ignore_index=True)
    print(f"[Info] 총 {len(data_list)}개의 파일, {len(df_raw)}개의 데이터 샘플을 병합했습니다.")
    
    # 2. 전처리 클래스 인스턴스화 및 변환
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit_transform(df_raw)
    
    # Target 분리
    if 'Capacity' not in df_processed.columns:
        print("[Error] 데이터에 Target 변수 'Capacity'가 존재하지 않습니다.")
        sys.exit(1)
        
    y = df_processed.pop('Capacity')
    
    # Feature 선택 및 누락 처리 보장
    features_to_use = [col for col in preprocessor.target_features if col in df_processed.columns]
    print(f"[Info] 학습에 사용할 피처: {features_to_use}")
    X = df_processed[features_to_use]
    
    # 혹시 남아있을 수 있는 결측치 보완 (간편화)
    X = X.fillna(X.median())
    
    # 극단치 제거 로직 (Train 데이터 학습 안정성을 위해 전체 적용 후 분할)
    if 'impedance_delta_36' in X.columns:
        valid_idx = abs(X['impedance_delta_36']) <= 100
        X = X[valid_idx]
        y = y[valid_idx]
        print(f"[Info] impedance_delta_36 극단치 제거 완료: {len(df_raw)} -> {len(X)} 개 샘플 남음.")
        
    # 3. Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[Info] 데이터 분할 완료: Train {X_train.shape}, Test {X_test.shape}")
    
    # 4. 모델 훈련 (LGBM & XGBoost Ensemble)
    print("[Info] 모델 학습을 시작합니다...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    
    # Voting ensemble
    ensemble_model = VotingRegressor(
        estimators=[('lgb', lgb_model), ('xgb', xgb_model)],
        weights=[0.6, 0.4]
    )
    
    ensemble_model.fit(X_train, y_train)
    
    # 5. 성능 평가
    y_pred = ensemble_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n==========================")
    print("      학습 모델 성능 평가      ")
    print("==========================")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")
    print("==========================\n")
    
    # 6. 모델 및 파이프라인(전처리기 포함 정보 등) 저장
    # 전처리기도 나중에 추론 시 필요하므로 함께 묶어 저장
    bundle = {
        'model': ensemble_model,
        'features': features_to_use
    }
    
    joblib.dump(bundle, args.model_path)
    print(f"[Info] 학습된 모델과 피처 정보가 {args.model_path} 에 저장되었습니다.")
    print("[Complete] 모든 과정이 성공적으로 완료되었습니다.")

if __name__ == '__main__':
    main()
