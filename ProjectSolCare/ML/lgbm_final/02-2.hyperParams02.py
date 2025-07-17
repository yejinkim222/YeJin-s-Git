import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 1. 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/model_storage/rf_data_final2.csv")

# ✅ 2. 타겟/피처 분리
y = df["years_until_ad"]
X = df.drop(columns=["years_until_ad"])

# ✅ 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 4. Optuna 최적화 함수 정의
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 10, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
        "num_leaves": trial.suggest_int("num_leaves", 30, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
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

# ✅ 6. 상위 5개 결과 출력
top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
print("\n# ✅ 상위 5개 R² 결과:")
for i, trial in enumerate(top_trials, 1):
    print(f"# {i:02d} ▶ R²: {trial.value:.4f} | params: {trial.params}")

# 결과
# ✅ 상위 5개 R² 결과:
# 01 ▶ R²: 0.7110 | params: {'n_estimators': 393, 'max_depth': 14, 'learning_rate': 0.1157131273322641, 'num_leaves': 77, 'min_child_samples': 5, 'subsample': 0.5461659542213151, 'colsample_bytree': 0.8462173721638144, 'reg_alpha': 3.6183765409317505, 'reg_lambda': 3.589211129939158}
# 02 ▶ R²: 0.7093 | params: {'n_estimators': 502, 'max_depth': 15, 'learning_rate': 0.08909235395525388, 'num_leaves': 136, 'min_child_samples': 5, 'subsample': 0.6348701900963607, 'colsample_bytree': 0.8310945014731589, 'reg_alpha': 3.1321012447108547, 'reg_lambda': 3.704373035537751}
# 03 ▶ R²: 0.7073 | params: {'n_estimators': 494, 'max_depth': 21, 'learning_rate': 0.09962945057959278, 'num_leaves': 176, 'min_child_samples': 5, 'subsample': 0.6159708645119968, 'colsample_bytree': 0.8327264431774768, 'reg_alpha': 4.027149755078882, 'reg_lambda': 4.037526844744179}
# 04 ▶ R²: 0.7064 | params: {'n_estimators': 559, 'max_depth': 19, 'learning_rate': 0.1092282035738191, 'num_leaves': 210, 'min_child_samples': 6, 'subsample': 0.5475345345406701, 'colsample_bytree': 0.8354883672488698, 'reg_alpha': 3.414824356599198, 'reg_lambda': 2.9716708944519765}
# 05 ▶ R²: 0.7061 | params: {'n_estimators': 431, 'max_depth': 14, 'learning_rate': 0.11708243101314118, 'num_leaves': 79, 'min_child_samples': 7, 'subsample': 0.5460736211747896, 'colsample_bytree': 0.8711314623358302, 'reg_alpha': 3.157588121635414, 'reg_lambda': 4.7452899690619}