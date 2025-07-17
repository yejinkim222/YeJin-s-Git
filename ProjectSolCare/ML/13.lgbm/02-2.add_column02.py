# 파생변수실험
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")

# ✅ 타겟 설정
target_col = "years_until_ad"

# ✅ 결측 마스킹 컬럼 제거 (파생 변수 생성 이후 적용)
remove_cols = [
    "mci_onset_after_missing", "edu_yrs_missing", "db_onset_after_missing",
    "ad_year_missing", "year_missing", "edu_level", "age_group5",
    "edu_is_low", "hibpe_onset_after_missing", "has_hibpe_missing"
]

# ✅ 타겟 및 베이스 피처 정의
non_features = ["hhid", "year", "hhid_year", target_col]
X_base = df.drop(columns=non_features).copy()
y = df[target_col]

# ✅ 파생 변수 생성

# 1. years_until_ad_ratio
df["years_until_ad_ratio"] = df["years_until_ad"] / (df["age"] + 1e-5)

# 2. is_midage_and_db
df["is_midage_and_db"] = ((df["age"] >= 60) & (df["age"] <= 75) & (df["has_db"] == 1)).astype(int)

# 3. onset_after_total (결측은 0으로 대체)
onset_vars = ["db_onset_after", "hibpe_onset_after", "mci_onset_after"]
df["onset_after_total"] = df[onset_vars].apply(lambda row: sum([v if v >= 0 else 0 for v in row]), axis=1)

# 4. has_multiple_risk (결측 -1은 제외하고 0, 1만 집계)
df["has_multiple_risk"] = (
    ((df["has_db"] == 1).astype(int)) +
    ((df["has_hibpe"] == 1).astype(int)) +
    ((df["AD_MCI_status"] >= 1).astype(int))
) >= 2
df["has_multiple_risk"] = df["has_multiple_risk"].astype(int)

# ✅ 파생 변수 리스트
derived_vars = [
    "years_until_ad_ratio", "is_midage_and_db",
    "onset_after_total", "has_multiple_risk"
]

# ✅ 평가 함수 정의
def evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    return train_r2, test_r2

# ✅ 베이스라인 계산 (파생 변수 없이, 컬럼만 제거 후)
X_baseline = X_base.drop(columns=remove_cols).dropna()
y_baseline = y.loc[X_baseline.index]
base_train_r2, base_test_r2 = evaluate_model(X_baseline, y_baseline)

print(f"📊 Baseline Train R²: {base_train_r2:.4f}")
print(f"📊 Baseline Test  R²: {base_test_r2:.4f}\n")

# ✅ 개별 파생 변수 실험
results = []
for var in derived_vars:
    X = X_base.copy()
    X[var] = df[var]  # 파생 변수 추가
    X = X.drop(columns=remove_cols).dropna()
    y_clean = y.loc[X.index]

    train_r2, test_r2 = evaluate_model(X, y_clean)
    results.append({
        "feature": var,
        "train_r2": train_r2,
        "test_r2": test_r2
    })

# ✅ 결과 출력
print("✅ 파생 변수별 성능 비교:")
for r in results:
    delta = r["test_r2"] - base_test_r2
    print(f"{r['feature']:25s} ▶ Train R²: {r['train_r2']:.4f} | Test R²: {r['test_r2']:.4f} | Δ Test R²: {delta:+.4f}")

# 결과
# ✅ 파생 변수별 성능 비교:
# years_until_ad_ratio      ▶ Train R²: 0.9989 | Test R²: 0.9972 | Δ Test R²: +0.2454
# is_midage_and_db          ▶ Train R²: 0.9402 | Test R²: 0.7507 | Δ Test R²: -0.0010
# onset_after_total         ▶ Train R²: 0.9423 | Test R²: 0.7545 | Δ Test R²: +0.0027
# has_multiple_risk         ▶ Train R²: 0.9404 | Test R²: 0.7528 | Δ Test R²: +0.0010