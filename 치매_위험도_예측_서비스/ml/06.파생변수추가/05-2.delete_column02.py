# 기존 데이터 로딩
# 새 파생변수 5개 생성
# 생성된 파생변수 실험
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import early_stopping

# ✅ 데이터 로드
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/08.AD_train_selected_features.csv")

# ✅ 파생변수 생성
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["edu_x_risk"] = df["edu_yrs"] * df["risk_weighted_age"]
df["risk_to_age_ratio"] = df["risk_weighted_age"] / (df["age"] + 1)
df["edu_ratio_to_risk"] = (df["age"] / (df["edu_yrs"] + 1)) / (df["risk_weighted_age"] + 1)
df["male_age_x_edu"] = df["male_age"] * df["edu_yrs"]

# ✅ 입력 변수 / 타겟 정의
target_col = "years_until_ad"
excluded_cols = ["hhid", "year", "hhid_year", target_col]
feature_cols = [col for col in df.columns if col not in excluded_cols]

X = df[feature_cols]
y = df[target_col]

# ✅ 학습/검증 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ LightGBM 모델 설정
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=60,
    max_depth=6,
    reg_alpha=0.3,
    reg_lambda=1.2,
    subsample=0.85,
    colsample_bytree=0.9,
    random_state=42
)

# ✅ 학습 (EarlyStopping 적용)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[early_stopping(stopping_rounds=50)],
    eval_metric="rmse"
)

# ✅ 예측 및 평가
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))
print(f"\n🔹 모델: LightGBM (파생변수 추가 실험)")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")

# ✅ 중요 변수 확인
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n📌 중요 변수 Top 10:")
print(importance_df.head(10))

# 결과
# 🔹 모델: LightGBM (파생변수 추가 실험)
#   - Train R² = 0.4507
#   - Test  R² = 0.3001

# 📌 중요 변수 Top 10:
#               feature  importance
# 0                 age         445
# 29         edu_x_risk         424
# 31  edu_ratio_to_risk         408
# 24      age_edu_ratio         311
# 27      age_male_diff         308
# 19         female_age         307
# 28          age_x_edu         305
# 5           has_hibpe         239
# 32     male_age_x_edu         229
# 18           male_age         215