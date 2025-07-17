# 파생변수 추가해보기
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")

# ✅ 타겟 변수
target_col = "years_until_ad"

# ✅ 성능 비교용 베이스라인 정의 함수
def evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    return train_r2, test_r2

# ✅ 제거 컬럼 (파생 변수 생성 이후에 제거)
remove_cols = [
    "mci_onset_after_missing", "edu_yrs_missing", "db_onset_after_missing",
    "ad_year_missing", "year_missing", "edu_level", "age_group5",
    "edu_is_low", "hibpe_onset_after_missing", "has_hibpe_missing"
]

# ✅ 초기 학습용 데이터셋 구성
non_features = ["hhid", "year", "hhid_year", target_col]
X_base = df.drop(columns=non_features).copy()
y = df[target_col]

# ✅ 파생 변수 생성
df["risk_age_ratio"] = df["risk_factor_sum"] / (df["age"] + 1e-5)
df["is_old_and_risky"] = ((df["age"] >= 80) & (df["risk_factor_sum"] >= 2)).astype(int)
df["risk_weighted_edu"] = df["edu_yrs"] / (1 + df["risk_factor_sum"])
df["risk_sum_squared"] = df["risk_factor_sum"] ** 2
df["risk_group_flag_median"] = (df["risk_weighted_age"] >= df["risk_weighted_age"].median()).astype(int)
df["risk_group_flag_q75"] = (df["risk_weighted_age"] >= df["risk_weighted_age"].quantile(0.75)).astype(int)

# ✅ 파생 변수 리스트
derived_vars = [
    "risk_age_ratio", "is_old_and_risky", "risk_weighted_edu",
    "risk_sum_squared", "risk_group_flag_median", "risk_group_flag_q75"
]

# ✅ 파생 변수별 성능 비교 실험
results = []
for var in derived_vars:
    # 파생 변수 추가
    X = X_base.copy()
    X[var] = df[var]
    X = X.drop(columns=remove_cols)  # 파생 변수 생성 후 제거

    X = X.dropna()
    y_clean = y.loc[X.index]

    train_r2, test_r2 = evaluate_model(X, y_clean)
    results.append({
        "feature": var,
        "train_r2": train_r2,
        "test_r2": test_r2
    })

# ✅ 베이스라인 성능 계산 (파생 변수 없이, 컬럼만 제거한 상태)
X_baseline = X_base.drop(columns=remove_cols).dropna()
y_baseline = y.loc[X_baseline.index]
base_train_r2, base_test_r2 = evaluate_model(X_baseline, y_baseline)

# ✅ 결과 출력
print(f"📊 Baseline Train R²: {base_train_r2:.4f}")
print(f"📊 Baseline Test  R²: {base_test_r2:.4f}\n")

print("✅ 파생 변수별 성능 비교:")
for r in results:
    delta = r["test_r2"] - base_test_r2
    print(f"{r['feature']:25s} ▶ Train R²: {r['train_r2']:.4f} | Test R²: {r['test_r2']:.4f} | Δ Test R²: {delta:+.4f}")

# 결과
# 📊 Baseline Train R²: 0.9406
# 📊 Baseline Test  R²: 0.7518

# ✅ 파생 변수별 성능 비교:
# risk_age_ratio            ▶ Train R²: 0.9434 | Test R²: 0.7535 | Δ Test R²: +0.0017
# is_old_and_risky          ▶ Train R²: 0.9406 | Test R²: 0.7518 | Δ Test R²: +0.0000
# risk_weighted_edu         ▶ Train R²: 0.9425 | Test R²: 0.7558 | Δ Test R²: +0.0040
# risk_sum_squared          ▶ Train R²: 0.9406 | Test R²: 0.7518 | Δ Test R²: +0.0000
# risk_group_flag_median    ▶ Train R²: 0.9406 | Test R²: 0.7518 | Δ Test R²: +0.0000
# risk_group_flag_q75       ▶ Train R²: 0.9406 | Test R²: 0.7518 | Δ Test R²: +0.0000