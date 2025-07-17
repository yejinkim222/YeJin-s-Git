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

# ✅ 4. Optuna 최적화 함수 정의 (정밀한 범위로 좁힘)
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 800),
        "max_depth": trial.suggest_int("max_depth", 30, 40),
        "learning_rate": trial.suggest_float("learning_rate", 0.072, 0.080),
        "num_leaves": trial.suggest_int("num_leaves", 160, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 5),
        "subsample": trial.suggest_float("subsample", 0.58, 0.65),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.92, 0.96),
        "reg_alpha": trial.suggest_float("reg_alpha", 2.5, 2.8),
        "reg_lambda": trial.suggest_float("reg_lambda", 2.5, 2.8),
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
# 01 ▶ R²: 0.7221 | params: {'n_estimators': 776, 'max_depth': 37, 'learning_rate': 0.07689774287364474, 'num_leaves': 184, 'min_child_samples': 3, 'subsample': 0.6401051140170831, 'colsample_bytree': 0.939529231783635, 'reg_alpha': 2.5388493064554645, 'reg_lambda': 2.7580180847936497}
# 02 ▶ R²: 0.7214 | params: {'n_estimators': 783, 'max_depth': 37, 'learning_rate': 0.07736409845530354, 'num_leaves': 182, 'min_child_samples': 3, 'subsample': 0.6301525113761699, 'colsample_bytree': 0.9361856448518707, 'reg_alpha': 2.528227599711778, 'reg_lambda': 2.768710491253943}
# 03 ▶ R²: 0.7211 | params: {'n_estimators': 772, 'max_depth': 37, 'learning_rate': 0.07665517966401407, 'num_leaves': 185, 'min_child_samples': 3, 'subsample': 0.6422816243659788, 'colsample_bytree': 0.938591005382844, 'reg_alpha': 2.521282933328162, 'reg_lambda': 2.755772636135241}
# 04 ▶ R²: 0.7210 | params: {'n_estimators': 767, 'max_depth': 40, 'learning_rate': 0.07362438625908002, 'num_leaves': 182, 'min_child_samples': 3, 'subsample': 0.6326875429407282, 'colsample_bytree': 0.9545719511463097, 'reg_alpha': 2.5339393336350424, 'reg_lambda': 2.573814995224175}
# 05 ▶ R²: 0.7206 | params: {'n_estimators': 784, 'max_depth': 38, 'learning_rate': 0.07659967398458095, 'num_leaves': 182, 'min_child_samples': 3, 'subsample': 0.6462274331761618, 'colsample_bytree': 0.9407611855264827, 'reg_alpha': 2.542794812800453, 'reg_lambda': 2.7809264283216963}