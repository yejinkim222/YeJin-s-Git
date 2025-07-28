import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 1. 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/model_storage/rf_data_final2.csv")

# ✅ 2. 피처/타겟 정의
X = df.drop(columns=["years_until_ad"])
y = df["years_until_ad"]

# ✅ 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 4. Optuna Objective 정의
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "n_jobs": -1
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)

# ✅ 5. Optuna 탐색 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# ✅ 6. 상위 5개 조합 출력
top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
print("\n# ✅ 상위 5개 R² 결과:")
for i, trial in enumerate(top_trials, 1):
    print(f"# {i:02d} ▶ R²: {trial.value:.4f} | params: {trial.params}")

# 결과
# ✅ 상위 5개 R² 결과:
# 01 ▶ R²: 0.7178 | params: {'n_estimators': 307, 'max_depth': 50, 'learning_rate': 0.1267370409669164, 'num_leaves': 280, 'min_child_samples': 7, 'subsample': 0.6019019821779996, 'colsample_bytree': 0.7692212083773909, 'reg_alpha': 3.59676931586859, 'reg_lambda': 2.011915192433604}
# 02 ▶ R²: 0.7123 | params: {'n_estimators': 429, 'max_depth': 50, 'learning_rate': 0.12504797196519293, 'num_leaves': 276, 'min_child_samples': 6, 'subsample': 0.7365129522018836, 'colsample_bytree': 0.7624132912177002, 'reg_alpha': 2.88848388526122, 'reg_lambda': 2.238411865560119}
# 03 ▶ R²: 0.7091 | params: {'n_estimators': 206, 'max_depth': 48, 'learning_rate': 0.12462703163194533, 'num_leaves': 155, 'min_child_samples': 5, 'subsample': 0.7601982370535232, 'colsample_bytree': 0.7243447061607622, 'reg_alpha': 3.051677213711435, 'reg_lambda': 2.601159549350637}
# 04 ▶ R²: 0.7058 | params: {'n_estimators': 213, 'max_depth': 46, 'learning_rate': 0.09996773563243548, 'num_leaves': 261, 'min_child_samples': 5, 'subsample': 0.7548489592721711, 'colsample_bytree': 0.9073699681281936, 'reg_alpha': 3.2831511705696697, 'reg_lambda': 2.9014966509330495}
# 05 ▶ R²: 0.7051 | params: {'n_estimators': 482, 'max_depth': 49, 'learning_rate': 0.13947952288423018, 'num_leaves': 291, 'min_child_samples': 5, 'subsample': 0.7580791341706331, 'colsample_bytree': 0.8915899143279955, 'reg_alpha': 2.984337686124848, 'reg_lambda': 2.644576298610584}