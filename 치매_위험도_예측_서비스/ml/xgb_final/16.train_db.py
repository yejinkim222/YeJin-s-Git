import pandas as pd
import numpy as np
import os
import json
import joblib
from sqlalchemy import create_engine, text
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ====== DB 접속 정보 ======
HOST = "localhost"
PORT = 1521
SERVICE_NAME = "xe"
USER = "asdf"
PASSWORD = "asdf"

# ====== DB 연결 ======
db_url = f"oracle+cx_oracle://{USER}:{PASSWORD}@{HOST}:{PORT}/?service_name={SERVICE_NAME}"
engine = create_engine(db_url)

# ====== 데이터 로딩 ======
query = text("SELECT * FROM LGBM_RESULT")
df = pd.read_sql(query, engine)
print("✅ 원본 데이터 로드 완료:", df.shape)

# ====== 결측치 처리 (inf, -inf, NaN) ======
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
print("🧹 NaN/inf 제거 후:", df.shape)

# ====== 타겟/피처 분리 ======
target_col = "years_until_ad"
X = df.drop(columns=[target_col])
y = df[target_col]

# ====== 데이터 분할 ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====== XGBoost 모델 파라미터 ======
best_params = {
    "n_estimators": 496,
    "max_depth": 12,
    "learning_rate": 0.14823095794483634,
    "subsample": 0.8915652093256143,
    "colsample_bytree": 0.9265601946981163,
    "reg_alpha": 1.5238151913604103,
    "reg_lambda": 0.032926218816428134,
    "gamma": 9.125702083638783,
    "max_delta_step": 5,
    "random_state": 42,
    "objective": "reg:squarederror"
}

# ====== 모델 학습 ======
model = XGBRegressor(**best_params)
model.fit(X_train, y_train)

# ====== 모델 성능 출력 ======
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))
print(f"\n🎯 Train R²: {train_r2:.4f}")
print(f"🎯 Test  R²: {test_r2:.4f}")

# ====== 저장 경로 설정 ======
save_dir = "C:/workspace/Project01/model_storage"
os.makedirs(save_dir, exist_ok=True)

# ====== 저장 경로 정의 (final3) ======
model_path = os.path.join(save_dir, "xgb_best_model_final3.pkl")
features_path = os.path.join(save_dir, "xgb_model_features_final3.json")
params_path = os.path.join(save_dir, "xgb_best_params_final3.json")
data_path = os.path.join(save_dir, "xgb_data_final3.csv")

# ====== 모델 및 관련 정보 저장 ======
joblib.dump(model, model_path)
with open(features_path, "w") as f:
    json.dump(list(X.columns), f, indent=2)
with open(params_path, "w") as f:
    json.dump(best_params, f, indent=2)
df.to_csv(data_path, index=False)

# ====== 저장 완료 출력 ======
print("\n📦 모델 저장 완료 →", model_path)
print("📦 피처 리스트 저장 완료 →", features_path)
print("📦 하이퍼파라미터 저장 완료 →", params_path)
print("📦 학습 데이터 저장 완료 →", data_path)

# 결과
# ✅ 원본 데이터 로드 완료: (1013, 25)
# 🧹 NaN/inf 제거 후: (1013, 25)

# 🎯 Train R²: 0.9470
# 🎯 Test  R²: 0.7163

# 📦 모델 저장 완료 → C:/workspace/Project01/model_storage\xgb_best_model_final3.pkl
# 📦 피처 리스트 저장 완료 → C:/workspace/Project01/model_storage\xgb_model_features_final3.json    
# 📦 하이퍼파라미터 저장 완료 → C:/workspace/Project01/model_storage\xgb_best_params_final3.json    
# 📦 학습 데이터 저장 완료 → C:/workspace/Project01/model_storage\xgb_data_final3.csv