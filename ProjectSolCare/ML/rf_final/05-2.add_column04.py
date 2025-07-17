import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/model_storage/xgb_data_final2.csv")

# ✅ 파생변수 생성 함수
feature_funcs = {
    "cog_flag_interact": lambda df: df["cognitive_decline_flag"] * df["age"],
    "hibpe_age_ratio": lambda df: df["hibpe_onset_after"] / (df["age"] + 1),
    "age_squared": lambda df: df["age"] ** 2,
    "has_any_risk": lambda df: ((df["has_db"] == 1) | (df["has_hibpe"] == 1) | (df["AD_MCI_status"] >= 1)).astype(int),
    "risk_weighted_age2": lambda df: df["age"] / (1 + df["risk_factor_sum"] ** 2),
    "age_plus_onset": lambda df: df[["age", "db_onset_after", "hibpe_onset_after", "mci_onset_after"]].fillna(0).sum(axis=1),
    "log_edu_yrs": lambda df: np.log1p(df["edu_yrs"])
}

# ✅ 모델 설정
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

# ✅ inf/nan 처리
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ✅ 파생변수 추가
for name, func in feature_funcs.items():
    df[name] = func(df)

# ✅ 전체 feature 목록
base_features = [col for col in df.columns if col != "years_until_ad"]
all_feats = list(feature_funcs.keys())

# ✅ full model 기준 R²
X_full = df[base_features]
y = df["years_until_ad"]
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_full, y, test_size=0.2, random_state=42)

model_full = RandomForestRegressor(**model_params)
model_full.fit(X_train_f, y_train_f)
full_r2 = r2_score(y_test_f, model_full.predict(X_test_f))

print(f"# ✅ ALL ADDED ▶ Test R²: {full_r2:.4f}\n")

# ✅ 변수 제거별 성능 비교
results = []

for feat in all_feats:
    reduced_feats = [f for f in all_feats if f != feat]
    use_cols = [col for col in base_features if col not in all_feats] + reduced_feats

    X = df[use_cols]
    y = df["years_until_ad"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    results.append((feat, r2))

# ✅ 정렬 및 출력
results.sort(key=lambda x: x[1])  # 제거 시 점수 낮은 순 정렬

print("# ✅ 변수 제거별 R²:")
for i, (feat, score) in enumerate(results, 1):
    delta = full_r2 - score
    print(f"# {i:02d} ▶ R²: {score:.4f} | 제거한 변수: {feat} | ↓Δ: {delta:.4f}")

# 결과
# ✅ ALL ADDED ▶ Test R²: 0.7080

# ✅ 변수 제거별 R²:
# 01 ▶ R²: 0.6949 | 제거한 변수: cog_flag_interact | ↓Δ: 0.0131
# 02 ▶ R²: 0.7005 | 제거한 변수: age_squared | ↓Δ: 0.0075
# 03 ▶ R²: 0.7040 | 제거한 변수: hibpe_age_ratio | ↓Δ: 0.0040
# 04 ▶ R²: 0.7048 | 제거한 변수: log_edu_yrs | ↓Δ: 0.0032
# 05 ▶ R²: 0.7095 | 제거한 변수: age_plus_onset | ↓Δ: -0.0015
# 06 ▶ R²: 0.7104 | 제거한 변수: risk_weighted_age2 | ↓Δ: -0.0024
# 07 ▶ R²: 0.7123 | 제거한 변수: has_any_risk | ↓Δ: -0.0043