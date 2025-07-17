# 새로운 파생변수를 추가
# 모델 학습을 통해 성능과 중요도를 확인
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import early_stopping

# ✅ 데이터 로드
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/09.AD_train_lgbm_final_pruned.csv")

# ✅ 파생변수 추가
df["age_squared"] = df["age"] ** 2
df["edu_squared"] = df["edu_yrs"] ** 2
df["edu_to_age_ratio"] = df["edu_yrs"] / (df["age"] + 1)
df["edu_female_diff"] = df["edu_yrs"] - df["female_age"]
df["male_female_age_diff"] = df["male_age"] - df["female_age"]
df["is_young_high_risk"] = ((df["age"] < 65) & (df["risk_weighted_age"] > 80)).astype(int)

# ✅ 타겟과 제외할 기본 컬럼 지정
target_col = "years_until_ad"
exclude_cols = [
    "edu_yrs", "edu_level", "years_until_mci", "has_db", "gender",
    "hhid", "year", "hhid_year"
]

# ✅ 타겟 및 입력 변수 정의
X = df.drop(columns=[target_col] + exclude_cols, errors="ignore")
y = df[target_col]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ LightGBM 모델 정의
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

# ✅ 모델 학습
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    callbacks=[early_stopping(stopping_rounds=50)]
)

# ✅ 성능 출력
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))
print(f"\n🔹 모델: LightGBM (새 파생변수 포함)")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")

# ✅ 중요 변수 출력
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n📌 중요 변수 Top 10:")
print(importance_df.head(10))

print("\n📌 제거 후보 (중요도 < 200):")
print(importance_df[importance_df["importance"] < 200])

# 결과
# 🔹 모델: LightGBM (새 파생변수 포함)
#   - Train R² = 0.4917
#   - Test  R² = 0.3317

# 📌 중요 변수 Top 10:
#               feature  importance
# 7       age_edu_ratio         807
# 0                 age         739
# 8       edu_male_diff         653
# 3      years_until_db         648
# 12   edu_to_age_ratio         633
# 4   risk_weighted_age         620
# 9       age_male_diff         484
# 6          female_age         450
# 11        edu_squared         387
# 13    edu_female_diff         368

# 📌 제거 후보 (중요도 < 200):
#                  feature  importance
# 14  male_female_age_diff         107
# 10           age_squared          86
# 1          AD_MCI_status          85
# 15    is_young_high_risk           7