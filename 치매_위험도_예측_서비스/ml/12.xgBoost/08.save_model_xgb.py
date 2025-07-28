# 데이터, 모델 저장
import pandas as pd
import os
import json
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 1. 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(file_path)

# ✅ 2. 파생변수 생성
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["edu_level_bucket"] = df["edu_level"] // 2

# ✅ 3. 타겟/특징 정의
y = df["years_until_ad"]
X = df.drop(columns=["years_until_ad", "ad_year", "hhid_year",
                     "hhid", "ad_year_missing", "year", "year_missing"])

# ✅ 4. 결측치 제거
df_model = pd.concat([X, y], axis=1).dropna()
X = df_model.drop(columns=["years_until_ad"])
y = df_model["years_until_ad"]

# ✅ 5. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 6. 모델 정의 및 학습
best_params = {
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.2,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.005,
    "reg_lambda": 0.5,
    "min_child_weight": 3,
    "gamma": 0,
    "max_delta_step": 0
}

model = XGBRegressor(**best_params)
model.fit(X_train, y_train)

# ✅ 7. 저장 경로 설정
save_dir = "C:/workspace/Project01/model_storage"
os.makedirs(save_dir, exist_ok=True)

# ✅ 8. 모델 저장
joblib.dump(model, os.path.join(save_dir, "xgb_best_model_final.pkl"))

# ✅ 9. 피처 리스트 저장
with open(os.path.join(save_dir, "xgb_model_features_final.json"), "w") as f:
    json.dump(list(X.columns), f, indent=2)

# ✅ 10. 하이퍼파라미터 저장
with open(os.path.join(save_dir, "xgb_best_params_final.json"), "w") as f:
    json.dump(best_params, f, indent=2)

print("📦 모델 저장 완료 →", os.path.join(save_dir, "xgb_best_model_final.pkl"))
print("📦 피처 리스트 저장 완료 →", os.path.join(save_dir, "xgb_model_features_final.json"))
print("📦 하이퍼파라미터 저장 완료 →", os.path.join(save_dir, "xgb_best_params_final.json"))

train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))
print(f"Train R²: {train_r2:.4f}")
print(f"Test  R²:  {test_r2:.4f}")

# 결과
# 📦 모델 저장 완료 → C:/workspace/Project01/model_storage\xgb_best_model_final.pkl
# 📦 피처 리스트 저장 완료 → C:/workspace/Project01/model_storage\xgb_model_features_final.json
# 📦 하이퍼파라미터 저장 완료 → C:/workspace/Project01/model_storage\xgb_best_params_final.json        

# 이 모델 스코어
# Train R²: 0.9709
# Test  R²:  0.6038