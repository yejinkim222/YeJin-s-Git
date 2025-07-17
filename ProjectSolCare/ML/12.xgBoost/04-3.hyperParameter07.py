# 여전히 오버피팅
# 다시 하이퍼파라미터 튜닝
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from itertools import product

# ✅ 데이터 로딩 및 파생변수 생성
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["edu_level_bucket"] = df["edu_level"] // 2

# ✅ 피처 및 타겟
X = df.drop(columns=["years_until_ad", "ad_year", "hhid_year"])
y = df["years_until_ad"]

# ✅ 파라미터 그리드 정의
param_grid = {
    "n_estimators": [160, 200],
    "max_depth": [5, 6],
    "learning_rate": [0.1, 0.2],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.8],
    "reg_alpha": [0.005],
    "reg_lambda": [0.5],
    "min_child_weight": [3, 5],
    "gamma": [0, 1],
    "max_delta_step": [0, 1],
}
param_combinations = list(product(*param_grid.values()))
param_names = list(param_grid.keys())

# ✅ 교차검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for params in param_combinations:
    param_dict = dict(zip(param_names, params))

    # 모델 초기화 (early stopping 포함)
    model = XGBRegressor(
        objective="reg:squarederror",
        early_stopping_rounds=10,
        eval_metric="rmse",
        verbosity=0,
        random_state=42,
        n_jobs=-1,
        **param_dict
    )

    fold_train_scores = []
    fold_test_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        fold_train_scores.append(model.score(X_train, y_train))
        fold_test_scores.append(model.score(X_val, y_val))

    results.append({
        **param_dict,
        "train_r2": round(np.mean(fold_train_scores), 6),
        "test_r2": round(np.mean(fold_test_scores), 6)
    })

# ✅ 상위 10개 출력
results_df = pd.DataFrame(results)
top10 = results_df.sort_values(by="test_r2", ascending=False).head(10)
print("✅ XGBoost Cross-Validation 기반 정밀 튜닝 결과 (상위 10개):")
print(top10.to_string(index=False))

# 결과
# ✅ XGBoost Cross-Validation 기반 정밀 튜닝 결과 (상위 10개):
#  n_estimators  max_depth  learning_rate  subsample  colsample_bytree  reg_alpha  reg_lambda  min_child_weight  gamma  max_delta_step  train_r2  test_r2
#           200          6            0.2        1.0               0.8      0.005         0.5                 3      0               0  0.986964 0.852593
#           160          6            0.2        1.0               0.8      0.005         0.5                 3      0               0  0.986964 0.852593
#           200          6            0.1        1.0               0.8      0.005         0.5                 3      0               0  0.980810 0.852090
#           160          6            0.1        1.0               0.8      0.005         0.5                 3      0               0  0.978095 0.850183
#           160          6            0.2        1.0               0.8      0.005         0.5                 3      1               0  0.982966 0.850153
#           200          6            0.2        1.0               0.8      0.005         0.5                 3      1               0  0.982966 0.850153
#           200          6            0.1        1.0               0.8      0.005         0.5                 3      1               0  0.978260 0.849851
#           200          6            0.2        0.7               0.8      0.005         0.5                 3      1               0  0.987626 0.849847
#           160          6            0.2        0.7               0.8      0.005         0.5                 3      1               0  0.987591 0.849830
#           200          6            0.2        1.0               0.8      0.005         0.5                 5      1               0  0.982546 0.849470