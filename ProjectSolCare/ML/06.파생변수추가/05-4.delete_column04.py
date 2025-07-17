# 중요도 낮은 변수 제거 후 실험
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import early_stopping

# ✅ 데이터 로드
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/08.AD_train_selected_features.csv")

# ✅ 제거할 중요도 낮은 변수 리스트
low_importance_cols = [
    "edu_yrs", "years_until_db_missing", "years_until_mci", "years_until_mci_missing",
    "log_age_ratio", "log_risk_weighted_age", "edu_is_low", "age_group5", "has_hibpe_missing",
    "high_risk_group", "has_db", "risk_factor_sum", "edu_level", "is_old", 
    "years_until_hibpe_missing", "risk_to_age_ratio", "gender", "edu_yrs_missing", "is_low_edu"
]

# ✅ 불필요 컬럼 제거
df.drop(columns=low_importance_cols + ["hhid", "year", "hhid_year"], inplace=True, errors='ignore')

# ✅ 입력 변수와 타겟 분리
X = df.drop(columns=["years_until_ad"])
y = df["years_until_ad"]

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

# ✅ 모델 학습 (early stopping 포함)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    callbacks=[early_stopping(stopping_rounds=50)]
)

# ✅ 성능 평가
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))
print(f"\n🔹 모델: LightGBM (불필요 변수 제거 후)")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")

# ✅ 중요 변수 상위 10개 출력
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)
print("\n📌 중요 변수 Top 10:")
print(importance_df.head(10))

# 결과
# 🔹 모델: LightGBM (불필요 변수 제거 후)
#   - Train R² = 0.4697
#   - Test  R² = 0.3313

# 📌 중요 변수 Top 10:
#              feature  importance
# 7      age_edu_ratio        1431
# 0                age         917
# 4  risk_weighted_age         714
# 8      edu_male_diff         652
# 3     years_until_db         635
# 6         female_age         599
# 9      age_male_diff         541
# 5           male_age         477
# 2          has_hibpe         277
# 1      AD_MCI_status         109