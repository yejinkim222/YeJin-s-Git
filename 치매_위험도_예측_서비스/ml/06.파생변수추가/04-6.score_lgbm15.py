# 파생변수 다시 실험
# 제거한 파생변수 목록: age_gap_sex, edu_yrs_level_diff, log_risk_weighted_age, edu_log
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 1. 데이터 로드
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 2. 제거할 파생 변수 목록
remove_features = [
    "age_gap_sex", "edu_yrs_level_diff", "log_risk_weighted_age", "edu_log"
]
df = df.drop(columns=[col for col in remove_features if col in df.columns])

# ✅ 3. 대상 feature / target 정의
target_col = "years_until_ad"
feature_cols = [col for col in df.columns if col not in ["hhid", "year", "hhid_year", target_col]]

X = df[feature_cols]
y = df[target_col]

# ✅ 4. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ✅ 5. 하이퍼파라미터 튜닝 실험
param_grid = [
    {"learning_rate": 0.01, "num_leaves": 60, "max_depth": 6, "reg_alpha": 0.3, "reg_lambda": 1.2, "subsample": 0.8, "colsample_bytree": 1.0},
    {"learning_rate": 0.007, "num_leaves": 110, "max_depth": 7, "reg_alpha": 0.6, "reg_lambda": 3.0, "subsample": 0.85, "colsample_bytree": 0.9},
    {"learning_rate": 0.005, "num_leaves": 100, "max_depth": 8, "reg_alpha": 0.6, "reg_lambda": 2.0, "subsample": 0.8, "colsample_bytree": 0.8},
]

SEED = 42

for i, params in enumerate(param_grid):
    model = lgb.LGBMRegressor(
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=SEED,
        n_estimators=1000
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )

    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))

    print(f"\n🔹 모델: model_remove_{i+1}")
    print(f"  - Train R² = {train_r2:.4f}")
    print(f"  - Test  R² = {test_r2:.4f}")
    importance = pd.Series(model.feature_importances_, index=X.columns)
    print("📌 중요 변수 Top 5:")
    print(importance.sort_values(ascending=False).head(5))

# 결과
# 🔹 모델: model_remove_3
#   - Train R² = 0.3838
#   - Test  R² = 0.3056
# 📌 중요 변수 Top 5:
# age                  1961
# female_age           1823
# edu_yrs              1439
# risk_weighted_age    1398
# male_age             1295
# dtype: int32
