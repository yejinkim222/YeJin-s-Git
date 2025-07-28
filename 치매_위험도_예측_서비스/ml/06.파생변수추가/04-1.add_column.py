# 성능 한계 느껴져서
# 파생변수 더 추가해보기
# age_group5 ≥ 10도 같이 적용
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import early_stopping

# ✅ 데이터 로드
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv")

# ✅ 파생변수 생성
df["age_edu_ratio"] = df["age"] / (df["edu_yrs"] + 1)
df["is_old_and_low_edu"] = ((df["age_group5"] >= 12) & (df["edu_yrs"] < 6)).astype(int)
df["has_db_and_hibpe"] = ((df["years_until_db"] != -1) & (df["years_until_hibpe"] != -1)).astype(int)
df["has_any_chronic"] = ((df["years_until_db"] != -1) | (df["years_until_hibpe"] != -1) | (df["has_hibpe"] == 1)).astype(int)
df["age_male_diff"] = df["male_age"] - df["age"]

# ✅ 필터링 (age_group5 ≥ 10)
df = df[df["age_group5"] >= 10].copy()

# ✅ 입력 변수 / 타겟 정의
target_col = "years_until_ad"
feature_cols = [
    'age', 'edu_yrs', 'male_age', 'female_age', 'risk_weighted_age',
    'years_until_db', 'years_until_hibpe', 'has_hibpe',
    'age_edu_ratio', 'is_old_and_low_edu', 'has_db_and_hibpe',
    'has_any_chronic', 'age_male_diff'
]
X = df[feature_cols]
y = df[target_col]

# ✅ 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# ✅ 학습 (Early Stopping 적용)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[early_stopping(stopping_rounds=50)]
)

# ✅ 예측 및 평가
y_pred = model.predict(X_test)
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, y_pred)

# ✅ 중요 변수 출력
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# ✅ 결과 출력
print(f"\n🔹 모델: LightGBM (파생변수 포함)")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")
print("\n📌 중요 변수 Top 5:")
for i, row in importance_df.head(5).iterrows():
    print(f"  - {row['feature']}: {row['importance']}")

# 결과
# 🔹 모델: LightGBM (파생변수 포함)
#   - Train R² = 0.4717
#   - Test  R² = 0.3215

# 📌 중요 변수 Top 5:
#   - age_edu_ratio: 1295
#   - age: 996
#   - risk_weighted_age: 861
#   - edu_yrs: 624
#   - age_male_diff: 613