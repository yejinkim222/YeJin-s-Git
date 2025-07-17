import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 1. 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/model_storage/xgb_data_final2.csv")

# ✅ 2. 파라미터 고정 (베이스라인)
base_params = {
    'n_estimators': 256,
    'max_depth': 20,
    'min_samples_split': 4,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': False,
    'random_state': 42,
    'n_jobs': -1
}

# ✅ 3. 기본 피처와 타겟 분리
target = "years_until_ad"
base_cols = [col for col in df.columns if col != target]
X_base = df[base_cols].copy()
y = df[target]

# ✅ 4. 실험할 파생변수 정의
derived_features = {
    "age_squared": lambda df: df["age"] ** 2,
    "log_edu_yrs": lambda df: np.log1p(df["edu_yrs"]),
    "age_div_edu": lambda df: df["age"] / (df["edu_yrs"] + 1),
    "risk_weighted_age2": lambda df: df["age"] / (1 + df["risk_factor_sum"]**2),
    "cog_flag_interact": lambda df: df["age"] * df["cognitive_decline_flag"]
}

# ✅ 5. 베이스라인 성능 확인
X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)
base_model = RandomForestRegressor(**base_params)
base_model.fit(X_train, y_train)
base_score = r2_score(y_test, base_model.predict(X_test))
print(f"# ✅ BASELINE ▶ Test R²: {base_score:.4f}")

# ✅ 6. 각 파생변수 실험
results = []

for name, func in derived_features.items():
    X_aug = X_base.copy()
    X_aug[name] = func(df)
    X_aug.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_aug.dropna(inplace=True)
    
    # 타겟도 인덱스 맞춰서 추출
    y_aug = y.loc[X_aug.index]

    X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)
    model = RandomForestRegressor(**base_params)
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    results.append((name, score))

# ✅ 7. 결과 출력
print("\n# ✅ 파생변수별 성능 비교:")
results.sort(key=lambda x: x[1], reverse=True)
for idx, (name, score) in enumerate(results, 1):
    print(f"# {idx:02d} ▶ R²: {score:.4f} | {name}")

# 결과
# ✅ BASELINE ▶ Test R²: 0.6869

# ✅ 파생변수별 성능 비교:
# 01 ▶ R²: 0.7024 | cog_flag_interact
# 02 ▶ R²: 0.6974 | age_squared
# 03 ▶ R²: 0.6939 | risk_weighted_age2
# 04 ▶ R²: 0.6889 | log_edu_yrs
# 05 ▶ R²: 0.6827 | age_div_edu