import pandas as pd
import os
import json
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 1. 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(file_path)

# ✅ 2. 파생변수 생성
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["age_edu_ratio"] = df["age"] / (df["edu_yrs"] + 1)

# ✅ 3. 타겟/특징 정의
y = df["years_until_ad"]
X = df.drop(columns=["years_until_ad", "ad_year", "hhid_year",
                     "hhid", "ad_year_missing", "year", "year_missing"])

# ✅ 4. 결측치 제거 (inf, nan 포함)
df_model = pd.concat([X, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
X = df_model.drop(columns=["years_until_ad"])
y = df_model["years_until_ad"]

# ✅ 5. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 6. 최적 파라미터 적용 (Optuna 결과)
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

model = XGBRegressor(**best_params)
model.fit(X_train, y_train)

# ✅ 7. 저장 경로 설정
save_dir = "C:/workspace/Project01/model_storage"
os.makedirs(save_dir, exist_ok=True)

# ✅ 8. 경로 정의
model_path = os.path.join(save_dir, "xgb_best_model_final2.pkl")
features_path = os.path.join(save_dir, "xgb_model_features_final2.json")
params_path = os.path.join(save_dir, "xgb_best_params_final2.json")
data_path = os.path.join(save_dir, "xgb_data_final2.csv")

# ✅ 9. 저장
joblib.dump(model, model_path)
with open(features_path, "w") as f:
    json.dump(list(X.columns), f, indent=2)
with open(params_path, "w") as f:
    json.dump(best_params, f, indent=2)
df_model.to_csv(data_path, index=False)

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
# 📦 모델 저장 완료 → C:/workspace/Project01/model_storage\xgb_best_model_final2.pkl
# 📦 피처 리스트 저장 완료 → C:/workspace/Project01/model_storage\xgb_model_features_final2.json
# 📦 하이퍼파라미터 저장 완료 → C:/workspace/Project01/model_storage\xgb_best_params_final2.json       
# 📦 파생변수 포함 학습데이터 저장 완료 → C:/workspace/Project01/model_storage\xgb_data_final2.csv     
# Train R²: 0.9467
# Test  R²: 0.7260
