# 오버피팅 해결 위해
# 하이퍼 파라미터 조정
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import itertools
import warnings

warnings.filterwarnings("ignore")

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(file_path)

# ✅ 피처/타겟 분리
target_col = "years_until_ad"
exclude_cols = ["ad_year", "hhid_year"]
X = df.drop(columns=[target_col] + exclude_cols)
y = df[target_col]

# ✅ train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ✅ 하이퍼파라미터 그리드
param_grid = {
    "n_estimators": [300, 400, 500],
    "max_depth": [5, 6, 7],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8],
    "reg_alpha": [0.1, 0.5],
    "reg_lambda": [1],
    "min_child_weight": [5, 10],
    "gamma": [0, 1],
    "max_delta_step": [0]
}

param_combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

results_split = []
results_kfold = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for comb in param_combinations:
    params = dict(zip(param_names, comb))

    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        eval_metric='rmse',
        early_stopping_rounds=20,
        **params
    )

    # ✅ train/test split 기준 평가
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    results_split.append({**params, "train_r2": train_r2, "test_r2": test_r2})

    # ✅ KFold 기반 평가
    cv_r2s = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        val_pred = model.predict(X_val)
        cv_r2s.append(r2_score(y_val, val_pred))
    results_kfold.append({**params, "cv_r2": np.mean(cv_r2s)})

# ✅ 결과 정렬 및 출력
df_split = pd.DataFrame(results_split).sort_values(by="test_r2", ascending=False).reset_index(drop=True)
df_kfold = pd.DataFrame(results_kfold).sort_values(by="cv_r2", ascending=False).reset_index(drop=True)

print("📊 [train/test split 기준] 상위 10개:")
print(df_split.head(10)[["n_estimators", "max_depth", "learning_rate", "subsample", 
                         "colsample_bytree", "reg_alpha", "reg_lambda", 
                         "min_child_weight", "gamma", "max_delta_step", 
                         "train_r2", "test_r2"]])
print("\n📊 [KFold 기준] 상위 10개:")
print(df_kfold.head(10)[["n_estimators", "max_depth", "learning_rate", "subsample", 
                         "colsample_bytree", "reg_alpha", "reg_lambda", 
                         "min_child_weight", "gamma", "max_delta_step", 
                         "cv_r2"]])

# 결과
# 📊 [train/test split 기준] 상위 10개:
#    n_estimators  max_depth  learning_rate  subsample  ...  gamma  max_delta_step  train_r2   test_r2
# 0           400          7           0.05        1.0  ...      0               0  0.989674  0.822427 
# 1           500          7           0.05        1.0  ...      0               0  0.989674  0.822427 
# 2           400          7           0.10        1.0  ...      0               0  0.997274  0.821489 
# 3           500          7           0.10        1.0  ...      0               0  0.997274  0.821489 
# 4           300          7           0.10        1.0  ...      0               0  0.996275  0.821070 
# 5           500          7           0.05        0.8  ...      0               0  0.983777  0.821012 
# 6           300          7           0.05        1.0  ...      0               0  0.983278  0.820395 
# 7           500          7           0.10        1.0  ...      0               0  0.995073  0.818555 
# 8           300          7           0.10        1.0  ...      0               0  0.995073  0.818555 
# 9           400          7           0.10        1.0  ...      0               0  0.995073  0.818555 

# [10 rows x 12 columns]

# 📊 [KFold 기준] 상위 10개:
#    n_estimators  max_depth  learning_rate  ...  gamma  max_delta_step     cv_r2
# 0           500          7           0.10  ...      0               0  0.862874
# 1           400          7           0.10  ...      0               0  0.862610
# 2           500          7           0.10  ...      0               0  0.860995
# 3           300          7           0.10  ...      0               0  0.860938
# 4           400          7           0.10  ...      0               0  0.860821
# 5           300          7           0.10  ...      0               0  0.859838
# 6           400          7           0.10  ...      0               0  0.859243
# 7           500          7           0.10  ...      0               0  0.859243
# 8           300          7           0.10  ...      0               0  0.858511
# 9           500          7           0.05  ...      0               0  0.857543