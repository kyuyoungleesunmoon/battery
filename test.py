import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# train.py에 정의된 전처리 클래스를 불러옵니다.
try:
    from train import DataPreprocessor
except ImportError:
    print("[Error] train.py 모듈을 불러올 수 없습니다. 같은 디렉터리 내에 train.py가 있는지 확인하세요.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Battery Capacity Prediction - Test Model')
    parser.add_argument('--test_data_path', type=str, default='./dataset/test.csv', help='평가/추론할 데이터 (CSV 파일 지정)')
    parser.add_argument('--model_path', type=str, default='./best_model.pkl', help='로드할 기학습 모델 파일 경로')
    parser.add_argument('--output_path', type=str, default='./predictions.csv', help='예측 결과 저장 경로')
    args = parser.parse_args()

    # 1. 모델 로그 및 파일 검증
    if not os.path.exists(args.model_path):
        print(f"[Error] 모델 파일이 존재하지 않습니다: {args.model_path}")
        sys.exit(1)
        
    print(f"[Info] 저장된 모델을 불러옵니다: {args.model_path}")
    bundle = joblib.load(args.model_path)
    ensemble_model = bundle['model']
    features_to_use = bundle['features']
    
    # 2. 테스트 데이터 로드
    if not os.path.exists(args.test_data_path):
        print(f"[Error] 지정된 경로에 테스트 데이터가 없습니다: {args.test_data_path}")
        sys.exit(1)
        
    print(f"[Info] 평가 데이터를 로딩합니다: {args.test_data_path}")
    df_raw = pd.read_csv(args.test_data_path)
    print(f"[Info] 데이터 로드 완료 (샘플 수: {len(df_raw)})")
    
    # Target 값이 존재하는지 체크 (평가 여부 결정)
    has_target = 'Capacity' in df_raw.columns
    if has_target:
        y_true = df_raw.pop('Capacity')
    
    # Original Data 저장용 ID(필요 시 index 사용) 복사
    results_df = pd.DataFrame()
    results_df['index'] = df_raw.index
    if has_target:
        results_df['Actual_Capacity'] = y_true.values

    # 3. 데이터 전처리
    print("[Info] 테스트 데이터 전처리를 수행합니다...")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.transform(df_raw)
    
    # 모델 학습 시 사용한 피처만 추출 및 순서 보장
    for col in features_to_use:
        if col not in df_processed.columns:
            print(f"[Warning] 테스트 데이터에 누락된 피처가 있습니다 -> {col} (0으로 대체)")
            df_processed[col] = 0.0
            
    X_test = df_processed[features_to_use]
    
    # 혹시 남아있을 결측치 보완
    X_test = X_test.fillna(X_test.median().fillna(0))
    
    # 극단치 제외 로직 (테스트 시에는 평가를 위해 남기거나 선택적으로 실행)
    # 여기서는 추론 목적이므로 극단치 제거를 생략하고 모든 샘플에 대해 예측 진행
    
    # 4. 추론 수행
    print("[Info] 모델 추론을 시작합니다...")
    y_pred = ensemble_model.predict(X_test)
    results_df['Predicted_Capacity'] = y_pred
    
    # 5. 성능 평가 (종속변수가 있는 경우만)
    if has_target:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print("\n==========================")
        print("      최종 테스트 모델 성능      ")
        print("==========================")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE : {mae:.4f}")
        print(f"R2  : {r2:.4f}")
        print("==========================\n")
    else:
        print("[Info] 'Capacity' 열이 없어 성능 지표 평가는 생략합니다.")
        
    # 6. 결과 저장
    results_df.to_csv(args.output_path, index=False)
    print(f"[Info] 모든 예측 결과가 저장되었습니다: {args.output_path}")

if __name__ == '__main__':
    main()
