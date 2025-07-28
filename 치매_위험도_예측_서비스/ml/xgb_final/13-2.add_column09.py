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
# 추가 추천 파생변수
df["age_edu_ratio"] = df["age"] / (df["edu_yrs"] + 1)

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

# ✅ 실험할 파생변수 정의 (NaN/inf 보정 포함)
new_features = {
    "age_log_edu": lambda df: (df["age"] * np.log(df["edu_yrs"].replace(0, np.nan) + 1)).replace([np.inf, -np.inf], np.nan).fillna(0),
    "risk_factor_interaction": lambda df: (df["risk_factor_sum"] * df["age"]).fillna(0),
    "delayed_onset_ratio": lambda df: (
        (df["db_onset_after"] + df["hibpe_onset_after"] + df["mci_onset_after"]) / (df["age"] + 1)
    ).replace([np.inf, -np.inf], np.nan).fillna(0),
    "age_group5_x_risk": lambda df: (df["age_group5"] * df["risk_factor_sum"]).fillna(0),
    "edu_level_x_has_db": lambda df: (df["edu_level"] * df["has_db"]).fillna(0),
    "mci_onset_delay_ratio": lambda df: (
        df["mci_onset_after"] / (df["age"] + 1e-3)
    ).replace([np.inf, -np.inf], np.nan).fillna(0),
    "risk_factor_sq": lambda df: (df["risk_factor_sum"] ** 2).fillna(0),
    "has_multiple_onsets": lambda df: (
        ((df["has_db"] + df["has_hibpe"] + (df["AD_MCI_status"] >= 1).astype(int)) >= 2).astype(int)
    ),
    "edu_missing_flag": lambda df: df["edu_yrs"].isna().astype(int),
    "age_bucket": lambda df: pd.cut(df["age"], bins=[49, 60, 70, 80, 90, 120], labels=False).fillna(0).astype(int)
}

# ✅ 고정 파라미터
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
    try:
        X = X_base.copy()
        X[name] = func(df_model)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(**best_params)
        model.fit(X_train, y_train)

        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))

        print(f"{name:25} ▶ Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    except Exception as e:
        print(f"{name:25} ▶ Skipped due to error: {e}")

# 결과(베이스라인: ▶ Train R²: 0.9468 | Test R²: 0.7344)
#   result = getattr(ufunc, method)(*inputs, **kwargs)
# age_log_edu               ▶ Train R²: 0.9501 | Test R²: 0.6334
# risk_factor_interaction   ▶ Train R²: 0.9487 | Test R²: 0.6393
# delayed_onset_ratio       ▶ Train R²: 0.9507 | Test R²: 0.6175
# age_group5_x_risk         ▶ Train R²: 0.9477 | Test R²: 0.6330
# edu_level_x_has_db        ▶ Train R²: 0.9501 | Test R²: 0.6533
# mci_onset_delay_ratio     ▶ Train R²: 0.9494 | Test R²: 0.6399
# risk_factor_sq            ▶ Train R²: 0.9496 | Test R²: 0.6345
# has_multiple_onsets       ▶ Train R²: 0.9496 | Test R²: 0.6345
# edu_missing_flag          ▶ Train R²: 0.9496 | Test R²: 0.6345
# age_bucket                ▶ Train R²: 0.9492 | Test R²: 0.6376