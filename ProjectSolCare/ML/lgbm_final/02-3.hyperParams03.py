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

# ✅ 4. Optuna 최적화 함수 정의 (탐색 범위 세밀화 + n_trials 증가)
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 12, 40),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.13),
        "num_leaves": trial.suggest_int("num_leaves", 50, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 2.5, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 2.5, 5.0),
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

# ✅ 6. 상위 5개 결과 정리
top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
print("\n# ✅ 상위 5개 R² 결과:")
for i, trial in enumerate(top_trials, 1):
    print(f"# {i:02d} ▶ R²: {trial.value:.4f} | params: {trial.params}")

# 결과
# ✅ 상위 5개 R² 결과:
# 01 ▶ R²: 0.7216 | params: {'n_estimators': 557, 'max_depth': 36, 'learning_rate': 0.07822212914096797, 'num_leaves': 172, 'min_child_samples': 3, 'subsample': 0.5876746354727588, 'colsample_bytree': 0.9399199907892275, 'reg_alpha': 2.500366327393991, 'reg_lambda': 2.580953962752256}
# 02 ▶ R²: 0.7208 | params: {'n_estimators': 564, 'max_depth': 39, 'learning_rate': 0.07821363227418039, 'num_leaves': 194, 'min_child_samples': 3, 'subsample': 0.6192000591962319, 'colsample_bytree': 0.9375797499217539, 'reg_alpha': 2.65887659484396, 'reg_lambda': 2.719768386640982}
# 03 ▶ R²: 0.7182 | params: {'n_estimators': 690, 'max_depth': 34, 'learning_rate': 0.07913199429247114, 'num_leaves': 82, 'min_child_samples': 3, 'subsample': 0.5769789468985265, 'colsample_bytree': 0.7897587229353621, 'reg_alpha': 2.5037848618415217, 'reg_lambda': 2.6381608908137824}
# 04 ▶ R²: 0.7172 | params: {'n_estimators': 552, 'max_depth': 38, 'learning_rate': 0.07756436539012722, 'num_leaves': 185, 'min_child_samples': 3, 'subsample': 0.6324278933118138, 'colsample_bytree': 0.9360650070594544, 'reg_alpha': 2.5101599020559484, 'reg_lambda': 2.6156249028939498}
# 05 ▶ R²: 0.7153 | params: {'n_estimators': 549, 'max_depth': 38, 'learning_rate': 0.07405078635587226, 'num_leaves': 192, 'min_child_samples': 3, 'subsample': 0.6420873106877745, 'colsample_bytree': 0.9268896750790052, 'reg_alpha': 2.508390845861939, 'reg_lambda': 2.713255890175597}