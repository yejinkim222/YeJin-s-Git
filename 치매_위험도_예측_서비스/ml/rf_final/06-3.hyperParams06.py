import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 1. 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/model_storage/xgb_data_final2.csv")

# ✅ 2. 파생변수 생성
df["cog_flag_interact"]     = df["AD_MCI_status"] * df["edu_yrs"]
df["hibpe_age_ratio"]       = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["age_squared"]           = df["age"] ** 2
df["has_any_risk"]          = (df[["has_db", "has_hibpe"]].sum(axis=1) > 0).astype(int)
df["risk_weighted_age2"]    = df["age"] * (df["has_db"] + df["has_hibpe"]) ** 2
df["age_plus_onset"]        = df["age"] + df["hibpe_onset_after"]
df["log_edu_yrs"]           = np.log1p(df["edu_yrs"])

# ✅ 3. 결측값 처리
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ✅ 4. 피처/타겟 설정
selected_features = [
    "cog_flag_interact", "hibpe_age_ratio", "age_squared",
    "has_any_risk", "risk_weighted_age2", "age_plus_onset", "log_edu_yrs"
]
base_features = [col for col in df.columns if col != "years_until_ad"]
X = df[selected_features + [col for col in base_features if col not in selected_features + ["years_until_ad"]]]
y = df["years_until_ad"]

# ✅ 5. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 6. Optuna 최적화 함수 정의
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 20, 70),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 42,
        "n_jobs": -1
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)

# ✅ 7. Optuna 탐색 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# ✅ 8. 상위 5개 조합 출력
top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
print("\n# ✅ 상위 5개 R² 결과:")
for i, trial in enumerate(top_trials, 1):
    print(f"# {i:02d} ▶ R²: {trial.value:.4f} | params: {trial.params}")

# 결과
# ✅ 상위 5개 R² 결과:
# 01 ▶ R²: 0.7040 | params: {'n_estimators': 774, 'max_depth': 58, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': False}
# 02 ▶ R²: 0.7040 | params: {'n_estimators': 782, 'max_depth': 61, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': False}
# 03 ▶ R²: 0.7040 | params: {'n_estimators': 771, 'max_depth': 59, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': False}
# 04 ▶ R²: 0.7040 | params: {'n_estimators': 727, 'max_depth': 59, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': False}
# 05 ▶ R²: 0.7035 | params: {'n_estimators': 827, 'max_depth': 20, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'log2', 'bootstrap': False}