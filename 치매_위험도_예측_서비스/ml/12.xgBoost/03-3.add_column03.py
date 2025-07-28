# 파생변수 추가 실험
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(file_path)

# ✅ 공통 설정
target_col = "years_until_ad"
exclude_cols = ["ad_year", "hhid_year"]
X_base = df.drop(columns=[target_col] + exclude_cols).copy()
y = df[target_col]

# ✅ 기준 모델 성능 측정
X_train, X_test, y_train, y_test = train_test_split(X_base, y, random_state=42)
base_model = XGBRegressor(random_state=42)
base_model.fit(X_train, y_train)
base_test_r2 = r2_score(y_test, base_model.predict(X_test))

# ✅ 파생변수 정의 함수 목록
def make_features(df):
    features = {}

    features["has_db_and_hibpe_flag"] = (df["has_db"] == 1) & (df["has_hibpe"] == 1)
    features["is_very_old"] = (df["age"] >= 85).astype(int)
    features["is_low_edu_old"] = ((df["edu_is_low"] == 1) & (df["age"] >= 80)).astype(int)
    features["edu_level_squared"] = df["edu_level"] ** 2
    features["age_squared"] = df["age"] ** 2
    features["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-6)
    features["risk_weighted_age_ratio"] = df["risk_weighted_age"] / (df["age"] + 1e-6)

    return features

# ✅ 파생변수 실험
results = []

derived = make_features(df)
for var_name, var_values in derived.items():
    X = X_base.copy()
    X[var_name] = var_values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)

    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    delta = test_r2 - base_test_r2

    results.append([var_name, round(train_r2, 4), round(test_r2, 4), round(delta, 4)])

# ✅ 결과 출력
results_df = pd.DataFrame(results, columns=["파생변수", "Train R²", "Test R²", "Δ Test R²"])
print("\n✅ 추천 파생변수 실험 결과:")
print(results_df.sort_values("Δ Test R²", ascending=False).to_string(index=False))

# 결과
# ✅ 추천 파생변수 실험 결과:
#                    파생변수  Train R²  Test R²  Δ Test R²
# hibpe_onset_delay_ratio    0.9987   0.7991     0.0035
#   has_db_and_hibpe_flag    0.9987   0.7955     0.0000
#             is_very_old    0.9987   0.7955     0.0000
#          is_low_edu_old    0.9987   0.7955     0.0000
#       edu_level_squared    0.9987   0.7955     0.0000
#             age_squared    0.9987   0.7955     0.0000
# risk_weighted_age_ratio    0.9987   0.7955     0.0000