# 모델 저장하는 코드
# 데이터, 모델 저장
import pandas as pd
import os
import json
import joblib
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 1. 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_for_lgbm.csv"
df = pd.read_csv(file_path)
df = df.drop(columns=["ad_year", "ad_year_missing", "year_missing", "cognitive_decline_flag", "years_until_ad_ratio"], errors="ignore")

# ✅ 2. 모델 학습 제외 변수 정의
target_col = "years_until_ad"
model_exclude_only = ["has_db", "edu_level"]
non_features = ["hhid", "year", "hhid_year", target_col] + model_exclude_only

# ✅ 3. 타겟 및 피처 구성
y = df[target_col]
X = df.drop(columns=[col for col in non_features if col in df.columns])

# ✅ 4. 결측치 제거
X = X.dropna()
y = y.loc[X.index]

# ✅ 5. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 6. LGBM 모델 정의 및 학습
best_params = {
    "objective": "regression",
    "random_state": 42,
    "n_estimators": 250,
    "max_depth": 7,
    "learning_rate": 0.12,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.1,
    "min_child_samples": 5,
    "boosting_type": "gbdt"
}

model = LGBMRegressor(**best_params)
model.fit(X_train, y_train)

# ✅ 7. 저장 경로 설정
save_dir = "C:/workspace/Project01/model_storage"
os.makedirs(save_dir, exist_ok=True)

# ✅ 8. 모델 저장
joblib.dump(model, os.path.join(save_dir, "lgbm_best_model_final.pkl"))

# ✅ 9. 피처 리스트 저장
with open(os.path.join(save_dir, "lgbm_model_features_final.json"), "w") as f:
    json.dump(list(X.columns), f, indent=2)

# ✅ 10. 하이퍼파라미터 저장
with open(os.path.join(save_dir, "lgbm_best_params_final.json"), "w") as f:
    json.dump(best_params, f, indent=2)

print("📦 모델 저장 완료 →", os.path.join(save_dir, "lgbm_best_model_final.pkl"))
print("📦 피처 리스트 저장 완료 →", os.path.join(save_dir, "lgbm_model_features_final.json"))
print("📦 하이퍼파라미터 저장 완료 →", os.path.join(save_dir, "lgbm_best_params_final.json"))

# 결과
# 📦 모델 저장 완료 → C:/workspace/Project01/model_storage\lgbm_best_model_final.pkl
# 📦 피처 리스트 저장 완료 → C:/workspace/Project01/model_storage\lgbm_model_features_final.json
# 📦 하이퍼파라미터 저장 완료 → C:/workspace/Project01/model_storage\lgbm_best_params_final.json     

# 이 모델 스코어
# ✅ Train R²: 0.9003
# ✅ Test  R²: 0.5838