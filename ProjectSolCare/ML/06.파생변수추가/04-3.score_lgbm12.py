# 파생변수 추가하면 효과적인지 실험하기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# 📌 기본 필터링
df = df[df["age_group5"] >= 10].copy()
target_col = "years_until_ad"

# 📌 새 파생변수 추가
df["age_edu_ratio"] = df["age"] / (df["edu_yrs"] + 1)
df["age_male_diff"] = df["age"] - df["male_age"]
df["age_female_diff"] = df["age"] - df["female_age"]
df["edu_missing_age"] = df["edu_yrs_missing"] * df["age"]
df["hibpe_missing_age"] = df["has_hibpe_missing"] * df["age"]

# 📌 기존 파생변수
existing_features = [
    'age', 'edu_yrs', 'risk_weighted_age', 'male_age', 'female_age',
    'log_risk_weighted_age', 'is_low_edu', 'is_old', 'high_risk_group'
]

# 📌 새 파생변수
new_features = [
    "age_edu_ratio", "age_male_diff", "age_female_diff",
    "edu_missing_age", "hibpe_missing_age"
]

# 📌 공통 하이퍼파라미터
params = {
    'objective': 'regression',
    'learning_rate': 0.01,
    'num_leaves': 60,
    'max_depth': 6,
    'reg_alpha': 0.3,
    'reg_lambda': 1.2,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': 42
}

# 📌 실험 정의
experiments = {
    "exp_5_new_features": new_features,
    "exp_6_full_combo": existing_features + new_features,
}

results = []

for name, feature_list in experiments.items():
    df_exp = df[feature_list + [target_col]].copy()
    df_exp = df_exp[df_exp[target_col] != -1].copy()
    X = df_exp[feature_list]
    y = df_exp[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results.append({
        "실험명": name,
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4)
    })

results_df = pd.DataFrame(results)
print(results_df)

# 결과
#                   실험명  Train R²  Test R²
# 0  exp_5_new_features    0.2336   0.2175
# 1    exp_6_full_combo    0.2502   0.2256