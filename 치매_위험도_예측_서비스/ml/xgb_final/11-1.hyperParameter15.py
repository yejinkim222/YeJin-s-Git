# 📌 Optuna 기반 XGBoost 하이퍼파라미터 튜닝
import optuna
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_for_lgbm.csv")
target_col = "years_until_ad"
non_features = ["hhid", "hhid_year", "year", "ad_year", "ad_year_missing", "year_missing", "years_until_ad_ratio"]
df = df.drop(columns=[col for col in non_features if col in df.columns])

X = df.drop(columns=[target_col])
y = df[target_col]

# ✅ 결측치 제거
X = X.dropna()
y = y.loc[X.index]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Optuna 목적 함수
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0, 10.0),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10)
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)

# ✅ 최적화 수행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500, n_jobs=1)  # 필요시 n_trials 수 늘려도 됨

# ✅ 상위 10개 조합 출력
top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]
for i, trial in enumerate(top_trials, 1):
    model = XGBRegressor(**trial.params)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    print(f"#{i:02d} ▶ Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | params: {trial.params}")

# 결과
#01 ▶ Train R²: 0.9529 | Test R²: 0.6184 | params: {'n_estimators': 572, 'max_depth': 15, 'learning_rate': 0.1760916652996189, 'subsample': 0.9148976849968271, 'colsample_bytree': 0.8305427744547373, 'reg_alpha': 1.2300708070382902, 'reg_lambda': 0.030574734762785306, 'min_child_weight': 1, 'gamma': 8.791502764339128, 'max_delta_step': 5}
#02 ▶ Train R²: 0.9501 | Test R²: 0.6253 | params: {'n_estimators': 580, 'max_depth': 15, 'learning_rate': 0.17852474594199613, 'subsample': 0.9075294180406309, 'colsample_bytree': 0.8390375430300911, 'reg_alpha': 1.9315357173666954, 'reg_lambda': 0.0004309142393525438, 'min_child_weight': 1, 'gamma': 8.86725856464206, 'max_delta_step': 4}
#03 ▶ Train R²: 0.9537 | Test R²: 0.6342 | params: {'n_estimators': 567, 'max_depth': 14, 'learning_rate': 0.18107564733742879, 'subsample': 0.8863543502394853, 'colsample_bytree': 0.875186363636511, 'reg_alpha': 1.0273560735477874, 'reg_lambda': 0.0014684863200251882, 'min_child_weight': 1, 'gamma': 8.532729643653939, 'max_delta_step': 4}
#04 ▶ Train R²: 0.9522 | Test R²: 0.6408 | params: {'n_estimators': 351, 'max_depth': 15, 'learning_rate': 0.2026193439967354, 'subsample': 0.9987069058414941, 'colsample_bytree': 0.8685171708253362, 'reg_alpha': 0.004668724396071655, 'reg_lambda': 0.006676349462231117, 'min_child_weight': 2, 'gamma': 8.117270525971067, 'max_delta_step': 5}
#05 ▶ Train R²: 0.9502 | Test R²: 0.6323 | params: {'n_estimators': 583, 'max_depth': 15, 'learning_rate': 0.14322259055403977, 'subsample': 0.9127360750018768, 'colsample_bytree': 0.8340684766709088, 'reg_alpha': 1.163635213707856, 'reg_lambda': 0.032722923052554795, 'min_child_weight': 1, 'gamma': 9.125706076301478, 'max_delta_step': 5}
#06 ▶ Train R²: 0.9678 | Test R²: 0.6400 | params: {'n_estimators': 577, 'max_depth': 13, 'learning_rate': 0.13894573706288235, 'subsample': 0.9242257786463289, 'colsample_bytree': 0.8555775786825284, 'reg_alpha': 0.5426276074319061, 'reg_lambda': 0.040487438617336176, 'min_child_weight': 2, 'gamma': 3.171166707760684, 'max_delta_step': 10}
#07 ▶ Train R²: 0.9515 | Test R²: 0.6336 | params: {'n_estimators': 569, 'max_depth': 15, 'learning_rate': 0.20357954409854387, 'subsample': 0.9241017410248432, 'colsample_bytree': 0.8392158299608916, 'reg_alpha': 1.2839387100869992, 'reg_lambda': 0.01918913027796636, 'min_child_weight': 1, 'gamma': 8.792631093294677, 'max_delta_step': 4}
#08 ▶ Train R²: 0.9502 | Test R²: 0.6463 | params: {'n_estimators': 558, 'max_depth': 15, 'learning_rate': 0.17790739002040387, 'subsample': 0.922872037260931, 'colsample_bytree': 0.8454880771367957, 'reg_alpha': 1.7513859374997904, 'reg_lambda': 0.035193620317457826, 'min_child_weight': 1, 'gamma': 8.731763551063658, 'max_delta_step': 4}
#09 ▶ Train R²: 0.9502 | Test R²: 0.6339 | params: {'n_estimators': 568, 'max_depth': 15, 'learning_rate': 0.1866935775000649, 'subsample': 0.915149404822314, 'colsample_bytree': 0.825863713185244, 'reg_alpha': 1.1907405944718243, 'reg_lambda': 0.032987181658213675, 'min_child_weight': 1, 'gamma': 9.13022596456112, 'max_delta_step': 5}
#10 ▶ Train R²: 0.9564 | Test R²: 0.6391 | params: {'n_estimators': 546, 'max_depth': 15, 'learning_rate': 0.18779000533152432, 'subsample': 0.9037887512260397, 'colsample_bytree': 0.849876235225252, 'reg_alpha': 0.6526035286773174, 'reg_lambda': 0.013996919352191854, 'min_child_weight': 1, 'gamma': 8.191914003799546, 'max_delta_step': 4}