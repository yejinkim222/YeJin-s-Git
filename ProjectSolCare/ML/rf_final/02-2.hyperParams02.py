import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import optuna

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/model_storage/xgb_data_final2.csv")

# ✅ 특성과 타겟 분리
X = df.drop(columns=["years_until_ad"])
y = df["years_until_ad"]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 결과 저장용 리스트
results = []

# ✅ Optuna 목적 함수
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 400),
        'max_depth': trial.suggest_int('max_depth', 10, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 6),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='r2', n_jobs=-1).mean()
    results.append((score, params))
    return score

# ✅ Optuna 최적화 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# ✅ 상위 5개 결과 정렬 후 출력
top_5 = sorted(results, key=lambda x: x[0], reverse=True)[:5]
for idx, (r2, params) in enumerate(top_5, 1):
    print(f"# {idx:02d} ▶ R²: {r2:.4f} | params: {params}")

# 결과
# 01 ▶ R²: 0.6743 | params: {'n_estimators': 362, 'max_depth': 21, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'log2', 'bootstrap': False, 'random_state': 42, 'n_jobs': -1}
# 02 ▶ R²: 0.6735 | params: {'n_estimators': 313, 'max_depth': 21, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'log2', 'bootstrap': False, 'random_state': 42, 'n_jobs': -1}
# 03 ▶ R²: 0.6735 | params: {'n_estimators': 333, 'max_depth': 24, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'log2', 'bootstrap': False, 'random_state': 42, 'n_jobs': -1}
# 04 ▶ R²: 0.6734 | params: {'n_estimators': 350, 'max_depth': 25, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'log2', 'bootstrap': False, 'random_state': 42, 'n_jobs': -1}
# 05 ▶ R²: 0.6734 | params: {'n_estimators': 339, 'max_depth': 24, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'log2', 'bootstrap': False, 'random_state': 42, 'n_jobs': -1}