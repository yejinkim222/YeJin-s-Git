import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")

# ✅ 기존 파생 변수
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)

# ✅ 제외 컬럼 정의
exclude_cols = ["years_until_ad", "ad_year", "hhid_year", "hhid", "ad_year_missing", "year", "year_missing"]
target = "years_until_ad"

# ✅ 모델용 데이터 구성
df_model = df.drop(columns=exclude_cols).copy()
df_model[target] = df[target]
df_model = df_model.dropna()

X_base = df_model.drop(columns=[target])
y = df_model[target]
X_base = X_base.dropna()
y = y.loc[X_base.index]

print("📌 사용 컬럼:", X_base.columns.tolist())

# ✅ 새로운 파생변수 목록
new_features = {
    "age_squared": lambda df: df["age"] ** 2,
    "log_age": lambda df: np.log(df["age"] + 1e-3),
    "age_edu_ratio": lambda df: df["age"] / (df["edu_yrs"] + 1),
    "log_risk_weighted_age": lambda df: np.log(df["risk_weighted_age"] + 1e-5),
    "multiple_onset_flag": lambda df: (
        ((df["db_onset_after"] < 5) &
         (df["hibpe_onset_after"] < 5) &
         (df["mci_onset_after"] < 5)).astype(int)
    ),
    "risk_density": lambda df: (df["risk_factor_sum"] + 1) / (df["age"] + 1),
    "onset_weighted_sum": lambda df: (
        0.5 * df["db_onset_after"] +
        0.3 * df["hibpe_onset_after"] +
        0.2 * df["mci_onset_after"]
    ),
    "age_x_has_db": lambda df: df["age"] * df["has_db"],
    "risk_category": lambda df: pd.cut(df["risk_factor_sum"], bins=[-1, 0, 1, 2, 3], labels=[0, 1, 2, 3]).astype(int)
}

# ✅ 고정 파라미터 (베스트 조합)
best_params = {
    'n_estimators': 498,
    'max_depth': 12,
    'learning_rate': 0.14997745579336536,
    'subsample': 0.8989092608354626,
    'colsample_bytree': 0.9284142503510097,
    'reg_alpha': 1.4386504039680161,
    'reg_lambda': 0.03803761653911136,
    'min_child_weight': 1,
    'gamma': 9.219422177382182,
    'max_delta_step': 5
}

# ✅ 테스트 루프
for name, func in new_features.items():
    X = X_base.copy()
    try:
        X[name] = func(df_model)

        # inf, -inf 처리
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # NaN 제거
        valid_idx = X.dropna().index
        X_valid = X.loc[valid_idx]
        y_valid = y.loc[valid_idx]

        X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42)

        model = XGBRegressor(**best_params)
        model.fit(X_train, y_train)

        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))

        print(f"{name:25} ▶ Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    except Exception as e:
        print(f"{name:25} ▶ ❌ 오류 발생: {e}")

# 결과
# 📌 사용 컬럼: ['age', 'gender', 'edu_yrs', 'has_db', 'AD_MCI_status', 'has_hibpe', 'edu_level', 'db_onset_after', 'hibpe_onset_after', 'mci_onset_after', 'age_group5', 'risk_factor_sum', 'edu_is_low', 'risk_weighted_age', 'age_gender_interact', 'hibpe_onset_after_missing', 'has_hibpe_missing', 'mci_onset_after_missing', 'edu_yrs_missing', 'db_onset_after_missing', 'cognitive_decline_flag', 'age_x_edu', 'hibpe_onset_delay_ratio']
# age_squared               ▶ Train R²: 0.9494 | Test R²: 0.6458
# log_age                   ▶ Train R²: 0.9494 | Test R²: 0.6458
# age_edu_ratio             ▶ Train R²: 0.9468 | Test R²: 0.7344
# log_risk_weighted_age     ▶ Train R²: 0.9480 | Test R²: 0.6478
# multiple_onset_flag       ▶ Train R²: 0.9480 | Test R²: 0.6288
# risk_density              ▶ Train R²: 0.9477 | Test R²: 0.6306
# onset_weighted_sum        ▶ Train R²: 0.9488 | Test R²: 0.6320
# age_x_has_db              ▶ Train R²: 0.9475 | Test R²: 0.6313
# risk_category             ▶ Train R²: 0.9470 | Test R²: 0.6425