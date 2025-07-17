# 파생변수 기여도 확인
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import early_stopping

# ✅ 데이터 로드
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/09.AD_train_lgbm_final_pruned.csv")

# ✅ 타겟 및 제거 대상 정의
target_col = "years_until_ad"
exclude_cols = [
    "years_until_mci", "male_female_age_diff", "AD_MCI_status", "age_squared",
    "has_db", "edu_level", "edu_squared", "gender", "is_young_high_risk",
    "hhid", "year", "hhid_year"  # 식별자 제외
]

# ✅ 입력 / 타겟 분리
X = df.drop(columns=[target_col] + exclude_cols, errors="ignore")
y = df[target_col]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 모델 정의 및 학습
model = lgb.LGBMRegressor(
    learning_rate=0.01,
    num_leaves=80,
    max_depth=6,
    reg_alpha=0.4,
    reg_lambda=1.0,
    subsample=0.9,
    colsample_bytree=0.9,
    n_estimators=1000,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    callbacks=[early_stopping(stopping_rounds=50)]
)

# ✅ 평가
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))
print(f"\n🔹 모델: LightGBM (중요도 낮은 변수 제거 후)")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")

# ✅ 중요도 상위 10개 출력
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)
print("\n📌 중요 변수 Top 10:")
print(importance_df.head(10))

# 결과
# 🔹 모델: LightGBM (중요도 낮은 변수 제거 후)
#   - Train R² = 0.4846
#   - Test  R² = 0.3347

# 📌 중요 변수 Top 10:
#              feature  importance
# 7      age_edu_ratio        1246
# 0                age         932
# 4  risk_weighted_age         755
# 3     years_until_db         711
# 8      edu_male_diff         668
# 1            edu_yrs         577
# 6         female_age         547
# 9      age_male_diff         518
# 5           male_age         423
# 2          has_hibpe         276