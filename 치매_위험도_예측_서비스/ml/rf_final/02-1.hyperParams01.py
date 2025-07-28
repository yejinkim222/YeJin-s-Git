import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

# ✅ 1. 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/model_storage/xgb_data_final2.csv")

# ✅ 2. 타겟 / 피처 분리
target = "years_until_ad"
X = df.drop(columns=[target])
y = df[target]

# ✅ 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 4. Optuna 목적 함수 정의
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="r2", n_jobs=-1)
    return np.mean(score)

# ✅ 5. Optuna 최적화 수행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# ✅ 6. 최적 모델 성능 평가
best_params = study.best_params
best_model = RandomForestRegressor(**best_params)
best_model.fit(X_train, y_train)

train_r2 = r2_score(y_train, best_model.predict(X_train))
test_r2 = r2_score(y_test, best_model.predict(X_test))

print(f"✅ Best Params: {best_params}")
print(f"✅ Train R²: {train_r2:.4f}")
print(f"✅ Test  R²:  {test_r2:.4f}")

# 결과
# ✅ Best Params: {'n_estimators': 256, 'max_depth': 20, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': False}
# ✅ Train R²: 0.9613
# ✅ Test  R²:  0.6904