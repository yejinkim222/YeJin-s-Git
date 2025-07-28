import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/model_storage/xgb_data_final2.csv")

# ✅ 베이스라인 피처 및 타겟
base_features = df.drop(columns=["years_until_ad"]).columns.tolist()
X_base = df[base_features].copy()
y = df["years_until_ad"]

# ✅ 데이터 분할
X_train_base, X_test_base, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)

# ✅ 고정 파라미터 (베이스라인)
best_rf_params = {
    'n_estimators': 256,
    'max_depth': 20,
    'min_samples_split': 4,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': False,
    'random_state': 42,
    'n_jobs': -1
}

# ✅ 베이스라인 모델 성능
baseline_model = RandomForestRegressor(**best_rf_params)
baseline_model.fit(X_train_base, y_train)
baseline_r2 = r2_score(y_test, baseline_model.predict(X_test_base))

# ✅ 새로운 파생변수 생성 함수
def create_new_features(df):
    features = {}
    features["log_age"] = np.log1p(df["age"])
    features["age_minus_edu"] = df["age"] - df["edu_yrs"]
    features["risk_over_age"] = df["risk_factor_sum"] / (df["age"] + 1e-5)
    features["onset_product"] = df["db_onset_after"].fillna(0) * df["mci_onset_after"].fillna(0)
    features["db_edu_interact"] = df["has_db"] * df["edu_yrs"]
    features["hibpe_age_ratio"] = df["hibpe_onset_after"].fillna(0) / (df["age"] + 1e-3)
    features["edu_sqrt"] = np.sqrt(df["edu_yrs"])
    features["risk_x_edu"] = df["risk_factor_sum"] * df["edu_yrs"]
    features["age_plus_onset"] = df["age"] + df["mci_onset_after"].fillna(0)
    features["has_any_risk"] = ((df["has_db"] + df["has_hibpe"] + (df["AD_MCI_status"] >= 1)) >= 1).astype(int)
    return pd.DataFrame(features)

# ✅ 성능 비교
results = [("BASELINE", baseline_r2)]

for col in create_new_features(df).columns:
    df_copy = df.copy()
    df_copy[col] = create_new_features(df)[col]
    temp_X = df_copy.drop(columns=["years_until_ad"])
    temp_y = df_copy["years_until_ad"]

    X_train, X_test, y_train, y_test = train_test_split(temp_X, temp_y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(**best_rf_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    results.append((col, r2))

# ✅ 결과 정렬 및 출력
results_sorted = sorted(results[1:], key=lambda x: x[1], reverse=True)
print(f"# ✅ BASELINE ▶ Test R²: {baseline_r2:.4f}\n")
print("# ✅ 파생변수별 성능 비교:")
for i, (feat, r2) in enumerate(results_sorted, 1):
    print(f"# {i:02d} ▶ R²: {r2:.4f} | {feat}")

# 결과
# ✅ BASELINE ▶ Test R²: 0.6869

# ✅ 파생변수별 성능 비교:
# 01 ▶ R²: 0.7019 | has_any_risk
# 02 ▶ R²: 0.6998 | age_plus_onset
# 03 ▶ R²: 0.6975 | hibpe_age_ratio
# 04 ▶ R²: 0.6966 | age_minus_edu
# 05 ▶ R²: 0.6961 | log_age
# 06 ▶ R²: 0.6944 | risk_x_edu
# 07 ▶ R²: 0.6918 | risk_over_age
# 08 ▶ R²: 0.6904 | onset_product
# 09 ▶ R²: 0.6889 | edu_sqrt
# 10 ▶ R²: 0.6846 | db_edu_interact