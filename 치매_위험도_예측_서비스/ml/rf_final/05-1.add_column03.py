import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/model_storage/xgb_data_final2.csv")

# ✅ 파생변수 정의 함수
derived_feature_funcs = {
    "cog_flag_interact": lambda df: df["cognitive_decline_flag"] * df["age"],
    "age_squared": lambda df: df["age"] ** 2,
    "risk_weighted_age2": lambda df: df["age"] / (1 + df["risk_factor_sum"] ** 2),
    "log_edu_yrs": lambda df: np.log1p(df["edu_yrs"]),
    "age_div_edu": lambda df: df["age"] / (df["edu_yrs"] + 1),
    "has_any_risk": lambda df: ((df["has_db"] == 1) | (df["has_hibpe"] == 1) | (df["AD_MCI_status"] >= 1)).astype(int),
    "age_plus_onset": lambda df: df[["age", "db_onset_after", "hibpe_onset_after", "mci_onset_after"]].fillna(0).sum(axis=1),
    "hibpe_age_ratio": lambda df: df["hibpe_onset_after"] / (df["age"] + 1)
}

# ✅ 공통 모델 설정
model_params = {
    "n_estimators": 256,
    "max_depth": 20,
    "min_samples_split": 4,
    "min_samples_leaf": 1,
    "max_features": 'sqrt',
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1
}

# ✅ inf/nan 제거 공통 처리
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ✅ 베이스라인 성능 확인
base_features = [col for col in df.columns if col != "years_until_ad"]
X_base = df[base_features]
y = df["years_until_ad"]
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_base, y, test_size=0.2, random_state=42)

baseline_model = RandomForestRegressor(**model_params)
baseline_model.fit(X_train_b, y_train_b)
baseline_r2 = r2_score(y_test_b, baseline_model.predict(X_test_b))

# ✅ 파생변수별 성능 비교
results = []
for feat, func in derived_feature_funcs.items():
    df_copy = df.copy()
    df_copy[feat] = func(df_copy)
    df_copy = df_copy.replace([np.inf, -np.inf], np.nan).dropna()

    X = df_copy[base_features + [feat]]
    y = df_copy["years_until_ad"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))

    results.append((feat, r2))

# ✅ 결과 출력
results.sort(key=lambda x: x[1], reverse=True)

print(f"# ✅ BASELINE ▶ Test R²: {baseline_r2:.4f}\n")
print("# ✅ 파생변수별 성능 비교:")
for i, (feat, score) in enumerate(results, 1):
    print(f"# {i:02d} ▶ R²: {score:.4f} | {feat}")
