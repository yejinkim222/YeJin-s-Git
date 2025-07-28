# 파생변수 실험
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 경로
data_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(data_path)

# ✅ 기본 설정
target_col = "years_until_ad"
exclude_cols = ["ad_year", "hhid_year"]
X_base = df.drop(columns=[target_col] + exclude_cols)
y = df[target_col]

# ✅ 파생변수 후보
derived_features = {
    "log_age": lambda df: np.log(df["age"] + 1),
    "edu_yrs_per_age": lambda df: df["edu_yrs"] / (df["age"] + 1e-5),
    "has_hibpe_and_old": lambda df: ((df["has_hibpe"] == 1) & (df["age"] >= 80)).astype(int),
    "is_middle_old": lambda df: ((df["age"] >= 75) & (df["age"] < 85)).astype(int),
    "mci_and_low_edu": lambda df: ((df["AD_MCI_status"] == 1) & (df["edu_is_low"] == 1)).astype(int),
    "edu_level_bucket": lambda df: df["edu_level"].replace({0: 0, 1: 1, 2: 1, 3: 2})
}

# ✅ 성능 비교 기준
baseline_test_r2 = 0.8335

results = []

# ✅ 실험 반복
for name, func in derived_features.items():
    df_temp = X_base.copy()
    df_temp[name] = func(df)

    X_train, X_test, y_train, y_test = train_test_split(df_temp, y, random_state=42)

    model = XGBRegressor(
        n_estimators=260,
        max_depth=6,
        learning_rate=0.2,
        subsample=1.0,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=2,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=False)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))

    results.append({
        "파생변수": name,
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4),
        "Δ Test R²": round(test_r2 - baseline_test_r2, 4)
    })

# ✅ 결과 출력
results_df = pd.DataFrame(results)
print("📊 추천 파생변수 실험 결과:\n", results_df)

# 결과
# 📊 추천 파생변수 실험 결과:
#                  파생변수  Train R²  Test R²  Δ Test R²
# 0            log_age    0.9992   0.8248    -0.0087
# 1    edu_yrs_per_age    0.9987   0.8165    -0.0170
# 2  has_hibpe_and_old    0.9896   0.8216    -0.0119
# 3      is_middle_old    0.9993   0.8285    -0.0050
# 4    mci_and_low_edu    0.9896   0.8216    -0.0119
# 5   edu_level_bucket    0.9983   0.8370     0.0035