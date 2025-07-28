# 저장한 모델 성능 확인
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 경로 정의
data_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_for_randomForest.csv"
model_path = "C:/workspace/Project01/model_storage/rf_best_model_final.pkl"
feature_path = "C:/workspace/Project01/model_storage/rf_model_features_final.json"

# ✅ 데이터 로딩
df = pd.read_csv(data_path)

# ✅ 피처 리스트 로딩
with open(feature_path, "r") as f:
    feature_cols = json.load(f)

# ✅ 타겟 및 피처 분리
target_col = "years_until_ad"
X = df[feature_cols]
y = df[target_col]

# ✅ 결측치 제거
X = X.dropna()
y = y.loc[X.index]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

# ✅ 모델 로딩
model = joblib.load(model_path)

# ✅ 예측 및 R² 계산
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

# ✅ 결과 출력
print("✅ RandomForest 저장 모델 성능")
print(f"Train R²: {train_r2:.4f}")
print(f"Test  R²:  {test_r2:.4f}")
