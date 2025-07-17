# 언더피팅이라 비교가안돼서 하이퍼파라미터 튜닝
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import random

# 🔧 재현성 고정
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 필터링: age_group5 ≥ 10
df = df[df['age_group5'] >= 10].copy()

# ✅ 타겟 & 피처 설정
target_col = "years_until_ad"
drop_cols = ["hhid", "hhid_year", "year", target_col]
X = df.drop(columns=drop_cols)
y = df[target_col]

# ✅ 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# ✅ 실험할 파라미터 세트
param_grid = [
    {"name": "lgbm_tuned1", "learning_rate": 0.01, "num_leaves": 60, "max_depth": 6, "reg_alpha": 0.3, "reg_lambda": 1.0, "subsample": 0.8, "colsample_bytree": 1.0},
    {"name": "lgbm_tuned2", "learning_rate": 0.015, "num_leaves": 80, "max_depth": 7, "reg_alpha": 0.4, "reg_lambda": 1.2, "subsample": 0.9, "colsample_bytree": 0.9},
    {"name": "lgbm_tuned3", "learning_rate": 0.02, "num_leaves": 90, "max_depth": 8, "reg_alpha": 0.2, "reg_lambda": 1.0, "subsample": 0.85, "colsample_bytree": 0.8},
    {"name": "lgbm_tuned4", "learning_rate": 0.025, "num_leaves": 100, "max_depth": 9, "reg_alpha": 0.1, "reg_lambda": 0.8, "subsample": 0.95, "colsample_bytree": 0.9},
    {"name": "lgbm_tuned5", "learning_rate": 0.03, "num_leaves": 120, "max_depth": 10, "reg_alpha": 0.5, "reg_lambda": 1.5, "subsample": 1.0, "colsample_bytree": 1.0},
]

# ✅ 실험
results = []
for params in param_grid:
    model = lgb.LGBMRegressor(
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=SEED,
        n_estimators=1000  # 충분히 크게 지정
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="rmse", callbacks=[lgb.early_stopping(stopping_rounds=20)])

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"🔹 모델: {params['name']}")
    print(f"  - Train R² = {train_r2:.4f}")
    print(f"  - Test  R² = {test_r2:.4f}")
    print(f"  📌 중요 변수 Top 5:\n    - " + "\n    - ".join(
        [f"{feat}: {score:.0f}" for feat, score in sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]]
    ))
    print(f"  🔧 Params: {params}")
    print("-" * 50)

    results.append({"실험명": params["name"], "Train R²": train_r2, "Test R²": test_r2})

# ✅ 결과 정리
results_df = pd.DataFrame(results)
print(results_df.sort_values(by="Test R²", ascending=False))

# 결과
#            실험명  Train R²   Test R²
# 0  lgbm_tuned1  0.395129  0.326034
# 2  lgbm_tuned3  0.422561  0.316599
# 3  lgbm_tuned4  0.440625  0.316126
# 1  lgbm_tuned2  0.409129  0.312426
# 4  lgbm_tuned5  0.447644  0.310453