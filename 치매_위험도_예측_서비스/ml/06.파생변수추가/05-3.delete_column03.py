# 성능 확인하고
# 변수 중요도 출력하기
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import early_stopping

# 📌 1. 데이터 로드
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/08.AD_train_selected_features.csv")

# 📌 2. 파생변수 추가
df["age_edu_ratio"] = df["age"] / (df["edu_yrs"] + 1)
df["edu_male_diff"] = df["edu_yrs"] - df["male_age"]
df["age_male_diff"] = df["male_age"] - df["age"]
df["log_age_ratio"] = np.log1p(df["age"] / (df["edu_yrs"] + 1))
df["risk_to_age_ratio"] = df["risk_weighted_age"] / (df["age"] + 1)
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["edu_x_risk"] = df["edu_yrs"] * df["risk_weighted_age"]
df["edu_ratio_to_risk"] = df["edu_yrs"] / (df["risk_weighted_age"] + 1)
df["male_age_x_edu"] = df["male_age"] * df["edu_yrs"]

# 📌 3. 학습 조건 설정
df = df[df["age_group5"] >= 10].copy()
target_col = "years_until_ad"
X = df.drop(columns=["years_until_ad", "hhid", "year", "hhid_year", "AD_MCI_status"])
y = df[target_col]

# 📌 4. train/test 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 5. LightGBM 모델 정의 및 학습
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

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
)

# 📌 6. 예측 및 성능 평가
y_pred = model.predict(X_test)
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, y_pred)

# 📌 7. 변수 중요도 정리
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n🔹 모델: LightGBM (전체 파생변수 포함)")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")

print("\n📌 중요 변수 Top 10:")
print(importance_df.head(10).to_string(index=False))

print("\n📌 제거 후보 (중요도 < 200):")
print(importance_df[importance_df["importance"] < 200].to_string(index=False))

# 결과
# 🔹 모델: LightGBM (전체 파생변수 포함)
#   - Train R² = 0.5144
#   - Test  R² = 0.3211

# 📌 중요 변수 Top 10:
#           feature  importance
#     age_edu_ratio         704
#               age         662
#        edu_x_risk         594
#         age_x_edu         526
#        female_age         504
#     edu_male_diff         434
#     age_male_diff         401
#    male_age_x_edu         397
# edu_ratio_to_risk         337
#    years_until_db         308

# 📌 제거 후보 (중요도 < 200):
#                   feature  importance
#                   edu_yrs         166
#    years_until_db_missing         140
#           years_until_mci         124
#   years_until_mci_missing          88
#             log_age_ratio          75
#     log_risk_weighted_age          36
#                edu_is_low          31
#                age_group5          26
#         has_hibpe_missing          24
#           high_risk_group           9
#                    has_db           8
#           risk_factor_sum           3
#                 edu_level           3
#                    is_old           1
# years_until_hibpe_missing           0
#         risk_to_age_ratio           0
#                    gender           0
#           edu_yrs_missing           0
#                is_low_edu           0