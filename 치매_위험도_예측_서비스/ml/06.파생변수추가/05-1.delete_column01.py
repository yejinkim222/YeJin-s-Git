# 새로 저장한 데이터로 학습시작..
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import early_stopping

# ✅ 데이터 로드
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/08.AD_train_selected_features.csv")

# ✅ 변수 정의
target_col = "years_until_ad"
drop_cols = ["hhid", "year", "hhid_year", "AD_MCI_status"]
feature_cols = [col for col in df.columns if col not in drop_cols + [target_col]]

X = df[feature_cols]
y = df[target_col]

# ✅ 학습/검증 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 모델 정의
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
    callbacks=[early_stopping(stopping_rounds=50)],
    eval_metric="rmse"
)

# ✅ 예측 및 성능 측정
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n🔹 모델: LightGBM (최종 데이터셋)")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")

# ✅ 중요 변수 출력
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n📌 중요 변수 Top 10:")
print(importance_df.head(10))

# 결과
# 🔹 모델: LightGBM (최종 데이터셋)
#   - Train R² = 0.4869
#   - Test  R² = 0.3453

# 📌 중요 변수 Top 10:
#               feature  importance
# 23      age_edu_ratio        1150
# 0                 age         866
# 16  risk_weighted_age         597
# 25      edu_male_diff         584
# 26      age_male_diff         581
# 18         female_age         535
# 7      years_until_db         528
# 2             edu_yrs         438
# 17           male_age         372
# 4           has_hibpe         244