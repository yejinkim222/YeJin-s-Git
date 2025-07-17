# 저장했던 모델 스코어 확인하기
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 1. 파일 경로 정의
model_path = "C:/workspace/Project01/model_storage/xgb_best_model_final.pkl"
feature_path = "C:/workspace/Project01/model_storage/xgb_model_features_final.json"
data_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"

# ✅ 2. 데이터 로딩
df = pd.read_csv(data_path)

# ✅ 3. 파생변수 복원
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["edu_level_bucket"] = df["edu_level"] // 2

# ✅ 4. 타겟 및 피처 정의
target_col = "years_until_ad"
y = df[target_col]

# ✅ 5. 저장된 피처 리스트 불러오기
with open(feature_path, "r") as f:
    feature_cols = json.load(f)

X = df[feature_cols].copy()

# ✅ 6. 결측치 제거 (학습 당시와 동일)
df_model = pd.concat([X, y], axis=1).dropna()
X = df_model[feature_cols]
y = df_model[target_col]

# ✅ 7. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 8. 모델 불러오기
model = joblib.load(model_path)

# ✅ 9. 예측 및 R² 평가
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))

# ✅ 10. 결과 출력
print(f"✅ XGBoost 모델 성능")
print(f"Train R²: {train_r2:.4f}")
print(f"Test  R²:  {test_r2:.4f}")
