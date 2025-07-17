import pandas as pd
import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 1. 데이터 로딩 (지정된 파일 사용)
file_path = "C:/workspace/Project01/model_storage/xgb_data_final2.csv"
df = pd.read_csv(file_path)

# ✅ 2. 파생변수 7개 생성
df["cog_flag_interact"]     = df["AD_MCI_status"] * df["edu_yrs"]
df["hibpe_age_ratio"]       = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["age_squared"]           = df["age"] ** 2
df["has_any_risk"]          = (df[["has_db", "has_hibpe"]].sum(axis=1) > 0).astype(int)
df["risk_weighted_age2"]    = df["age"] * (df["has_db"] + df["has_hibpe"]) ** 2
df["age_plus_onset"]        = df["age"] + df["hibpe_onset_after"]
df["log_edu_yrs"]           = np.log1p(df["edu_yrs"])

# ✅ 3. 결측값 처리 (inf, NaN)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ✅ 4. 입력/타겟 정의
selected_features = [
    "cog_flag_interact", "hibpe_age_ratio", "age_squared",
    "has_any_risk", "risk_weighted_age2", "age_plus_onset", "log_edu_yrs"
]
base_features = [col for col in df.columns if col != "years_until_ad"]
X = df[selected_features + [col for col in base_features if col not in selected_features + ["years_until_ad"]]]
y = df["years_until_ad"]

# ✅ 5. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 6. 최적 파라미터 적용 (Optuna 결과: R² 0.7125)
best_params = {
    "n_estimators": 173,
    "max_depth": 49,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": None,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1
}
model = RandomForestRegressor(**best_params)
model.fit(X_train, y_train)

# ✅ 7. 저장 경로 설정
save_dir = "C:/workspace/Project01/model_storage"
os.makedirs(save_dir, exist_ok=True)

# ✅ 8. 저장 경로 정의
model_path = os.path.join(save_dir, "rf_best_model_final2.pkl")
features_path = os.path.join(save_dir, "rf_model_features_final2.json")
params_path = os.path.join(save_dir, "rf_best_params_final2.json")
data_path = os.path.join(save_dir, "rf_data_final2.csv")

# ✅ 9. 모델 및 정보 저장
joblib.dump(model, model_path)
with open(features_path, "w") as f:
    json.dump(list(X.columns), f, indent=2)
with open(params_path, "w") as f:
    json.dump(best_params, f, indent=2)
df.to_csv(data_path, index=False)

# ✅ 10. 결과 출력
print("📦 모델 저장 완료 →", model_path)
print("📦 피처 리스트 저장 완료 →", features_path)
print("📦 하이퍼파라미터 저장 완료 →", params_path)
print("📦 파생변수 포함 학습데이터 저장 완료 →", data_path)

train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))
print(f"Train R²: {train_r2:.4f}")
print(f"Test  R²: {test_r2:.4f}")

# 결과
# 📦 모델 저장 완료 → C:/workspace/Project01/model_storage\rf_best_model_final2.pkl
# 📦 피처 리스트 저장 완료 → C:/workspace/Project01/model_storage\rf_model_features_final2.json
# 📦 하이퍼파라미터 저장 완료 → C:/workspace/Project01/model_storage\rf_best_params_final2.json        
# 📦 파생변수 포함 학습데이터 저장 완료 → C:/workspace/Project01/model_storage\rf_data_final2.csv      
# Train R²: 0.9434
# Test  R²: 0.7125
