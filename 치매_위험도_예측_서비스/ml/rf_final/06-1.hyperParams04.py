import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/model_storage/xgb_data_final2.csv")

# ✅ 파생변수 생성
df["cog_flag_interact"] = df["cognitive_decline_flag"] * df["age"]
df["hibpe_age_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1)
df["age_squared"] = df["age"] ** 2
df["has_any_risk"] = ((df["has_db"] == 1) | (df["has_hibpe"] == 1) | (df["AD_MCI_status"] >= 1)).astype(int)
df["risk_weighted_age2"] = df["age"] / (1 + df["risk_factor_sum"] ** 2)
df["age_plus_onset"] = df[["age", "db_onset_after", "hibpe_onset_after", "mci_onset_after"]].fillna(0).sum(axis=1)
df["log_edu_yrs"] = np.log1p(df["edu_yrs"])

# ✅ inf/nan 처리
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ✅ feature 및 타겟 설정
X = df.drop(columns=["years_until_ad"])
y = df["years_until_ad"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Optuna 목적 함수
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 42,
        "n_jobs": -1
    }
    model = RandomForestRegressor(**params)
    return cross_val_score(model, X_train, y_train, cv=3, scoring="r2", n_jobs=-1).mean()

# ✅ 최적화 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# ✅ 최적 모델 학습 및 결과
best_params = study.best_params
model = RandomForestRegressor(**best_params)
model.fit(X_train, y_train)

train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))

print("# ✅ Best Params:", best_params)
print(f"# ✅ Train R²: {train_r2:.4f}")
print(f"# ✅ Test  R²:  {test_r2:.4f}")

# 결과
# ✅ Best Params: {'n_estimators': 193, 'max_depth': 17, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': False}
# ✅ Train R²: 0.9601
# ✅ Test  R²:  0.7082