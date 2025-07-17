import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/model_storage/xgb_data_final2.csv")

# ✅ 성능 향상에 기여한 파생변수 8개 생성
df["cog_flag_interact"] = df["age"] * df["cognitive_decline_flag"]
df["age_squared"] = df["age"] ** 2
df["risk_weighted_age2"] = df["age"] / (1 + df["risk_factor_sum"] + 1e-3)
df["has_any_risk"] = ((df["has_db"] + df["has_hibpe"] + (df["AD_MCI_status"] >= 1)) >= 1).astype(int)
df["age_plus_onset"] = df["age"] + df["db_onset_after"].fillna(0) + df["hibpe_onset_after"].fillna(0) + df["mci_onset_after"].fillna(0)
df["hibpe_age_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["age_minus_edu"] = df["age"] - df["edu_yrs"]
df["log_age"] = np.log(df["age"] + 1)

# ✅ NaN/inf 처리
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ✅ 변수 정의
target = "years_until_ad"
X = df.drop(columns=["years_until_ad", "ad_year", "hhid_year", "hhid", "ad_year_missing", "year", "year_missing"], errors="ignore")
y = df["years_until_ad"]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Optuna 목적 함수
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="r2", n_jobs=-1)
    return score.mean()

# ✅ Optuna 탐색 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# ✅ 상위 5개 결과 출력
top_trials = sorted(study.trials, key=lambda x: x.value, reverse=True)[:5]
for i, trial in enumerate(top_trials, 1):
    print(f"# {i:02} ▶ R²: {trial.value:.4f} | params: {trial.params}")

# 결과
# 01 ▶ R²: 0.6627 | params: {'n_estimators': 583, 'max_depth': 24, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': False}
# 02 ▶ R²: 0.6625 | params: {'n_estimators': 581, 'max_depth': 25, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': False}
# 03 ▶ R²: 0.6625 | params: {'n_estimators': 581, 'max_depth': 25, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': False}
# 04 ▶ R²: 0.6625 | params: {'n_estimators': 574, 'max_depth': 25, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': False}
# 05 ▶ R²: 0.6625 | params: {'n_estimators': 580, 'max_depth': 20, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': False}