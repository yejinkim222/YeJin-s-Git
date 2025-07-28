# 기여도 낮은 파생변수 제거하고
# 새로운 파생변수 추가해보기
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import early_stopping

# ✅ 1. 데이터 로드
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 2. 파생변수 생성 (기존 + 새로 제안된 변수들)
df["age_edu_ratio"] = df["age"] / (df["edu_yrs"] + 1)
df["is_old_and_low_edu"] = ((df["age_group5"] >= 12) & (df["edu_yrs"] < 6)).astype(int)
df["has_db_and_hibpe"] = ((df["years_until_db"] != -1) & (df["years_until_hibpe"] != -1)).astype(int)
df["has_any_chronic"] = ((df["years_until_db"] != -1) | (df["years_until_hibpe"] != -1) | (df["has_hibpe"] == 1)).astype(int)
df["age_male_diff"] = df["male_age"] - df["age"]
df["edu_age_ratio"] = df["edu_yrs"] / (df["age"] + 1)
df["edu_risk_ratio"] = df["edu_yrs"] / (df["risk_weighted_age"] + 1)
df["log_age_ratio"] = np.log1p(df["age"]) / np.log1p(df["edu_yrs"] + 1)
df["age_minus_risk"] = df["age"] - df["risk_weighted_age"]
df["edu_male_diff"] = df["male_age"] - df["edu_yrs"]

# ✅ 3. 필터링: age_group5 ≥ 10
df = df[df["age_group5"] >= 10].copy()

# ✅ 4. 입력 변수/타겟 정의
target_col = "years_until_ad"
feature_cols = [
    'age', 'edu_yrs', 'male_age', 'female_age', 'risk_weighted_age',
    'years_until_db', 'years_until_hibpe', 'has_hibpe',
    'age_edu_ratio', 'is_old_and_low_edu', 'has_db_and_hibpe',
    'has_any_chronic', 'age_male_diff',
    'edu_age_ratio', 'edu_risk_ratio', 'log_age_ratio',
    'age_minus_risk', 'edu_male_diff'
]
X = df[feature_cols]
y = df[target_col]

# ✅ 5. 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 6. LightGBM 모델 정의 및 학습
model = lgb.LGBMRegressor(
    learning_rate=0.01,
    num_leaves=80,
    max_depth=7,
    reg_alpha=0.4,
    reg_lambda=1.5,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_estimators=1000
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
)

# ✅ 7. 성능 평가
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# ✅ 8. 중요 변수 출력 및 제거 후보 확인
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n🔹 모델: LightGBM (전체 파생변수 포함)")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")

print("\n📌 중요 변수 Top 5:")
for i, row in importance_df.head(5).iterrows():
    print(f"  - {row['feature']}: {row['importance']}")

print("\n📌 제거 후보 (중요도 < 200):")
print(importance_df[importance_df["importance"] < 200])

# 결과
# 🔹 모델: LightGBM (전체 파생변수 포함)
#   - Train R² = 0.4673
#   - Test  R² = 0.3141

# 📌 중요 변수 Top 5:
#   - age: 632
#   - log_age_ratio: 593
#   - risk_weighted_age: 496
#   - edu_male_diff: 487
#   - female_age: 482

# 📌 제거 후보 (중요도 < 200):
#                feature  importance
# 16      age_minus_risk          55
# 11     has_any_chronic          51
# 6    years_until_hibpe           0
# 10    has_db_and_hibpe           0
# 9   is_old_and_low_edu           0