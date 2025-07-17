import pandas as pd
import os
import json
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 1. 데이터 로딩 (xgb_data_final2.csv 사용)
file_path = "C:/workspace/Project01/model_storage/rf_data_final2.csv"
df = pd.read_csv(file_path)

# ✅ 2. 결측값 처리 (inf, NaN 포함)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ✅ 3. 입력/타겟 정의
X = df.drop(columns=["years_until_ad"])
y = df["years_until_ad"]

# ✅ 4. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 5. 최적 파라미터 적용 (Optuna 결과: R² 0.7221 기준)
best_params = {
    "n_estimators": 776,
    "max_depth": 37,
    "learning_rate": 0.07689774287364474,
    "num_leaves": 184,
    "min_child_samples": 3,
    "subsample": 0.6401051140170831,
    "colsample_bytree": 0.939529231783635,
    "reg_alpha": 2.5388493064554645,
    "reg_lambda": 2.7580180847936497,
    "random_state": 42,
    "n_jobs": -1
}
model = lgb.LGBMRegressor(**best_params)
model.fit(X_train, y_train)

# ✅ 6. 저장 경로 정의
save_dir = "C:/workspace/Project01/model_storage"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "lgbm_best_model_final2.pkl")
features_path = os.path.join(save_dir, "lgbm_model_features_final2.json")
params_path = os.path.join(save_dir, "lgbm_best_params_final2.json")
data_path = os.path.join(save_dir, "lgbm_data_final2.csv")

# ✅ 7. 저장
joblib.dump(model, model_path)
with open(features_path, "w") as f:
    json.dump(list(X.columns), f, indent=2)
with open(params_path, "w") as f:
    json.dump(best_params, f, indent=2)
df.to_csv(data_path, index=False)

# ✅ 8. 성능 출력
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))
print("📦 모델 저장 완료 →", model_path)
print("📦 피처 리스트 저장 완료 →", features_path)
print("📦 하이퍼파라미터 저장 완료 →", params_path)
print("📦 학습 데이터 저장 완료 →", data_path)
print(f"✅ LGBM Train R²: {train_r2:.4f}")
print(f"✅ LGBM Test  R²: {test_r2:.4f}")

# 결과
# 📦 모델 저장 완료 → C:/workspace/Project01/model_storage\lgbm_best_model_final2.pkl
# 📦 피처 리스트 저장 완료 → C:/workspace/Project01/model_storage\lgbm_model_features_final2.json
# 📦 하이퍼파라미터 저장 완료 → C:/workspace/Project01/model_storage\lgbm_best_params_final2.json      
# 📦 학습 데이터 저장 완료 → C:/workspace/Project01/model_storage\lgbm_data_final2.csv
# ✅ LGBM Train R²: 0.9580
# ✅ LGBM Test  R²: 0.7221