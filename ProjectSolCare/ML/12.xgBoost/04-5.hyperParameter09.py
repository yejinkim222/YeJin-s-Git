# train/test split, K-Fold 교차검증 방식 모두 사용해서 조합해보기
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ✅ 데이터 로드
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(file_path)

# ✅ 타겟 및 제외 컬럼 설정
target_col = "years_until_ad"
exclude_cols = ["ad_year", "hhid_year"]

# ✅ 성능 향상한 파생변수만 생성
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["edu_level_bucket"] = df["edu_level"].apply(lambda x: 0 if x <= 1 else (1 if x == 2 else 2))
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1)

# ✅ X, y 설정
X = df.drop(columns=[target_col] + exclude_cols)
y = df[target_col]

# ✅ 하이퍼파라미터 그리드
param_grid = {
    "n_estimators": [160, 200],
    "max_depth": [5, 6],
    "learning_rate": [0.1, 0.2],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 0.8],
    "reg_alpha": [0.01, 0.1],
    "reg_lambda": [0.5, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 1],
    "max_delta_step": [0],
}

# ✅ 모든 조합 생성
param_list = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

# ✅ 결과 저장 리스트
results_split = []
results_kfold = []

# ✅ 데이터 분할 (고정 분할 기준)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ✅ 각 조합에 대해 train/test split 성능 & KFold 성능 모두 저장
for params in param_list:
    param_dict = dict(zip(param_names, params))
    
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
        early_stopping_rounds=20,
        eval_metric="rmse",
        **param_dict
    )

    # 📌 (1) train/test split 방식
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    results_split.append({**param_dict, "train_r2": train_r2, "test_r2": test_r2})

    # 📌 (2) K-Fold 교차검증 방식
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2s = []
    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        fold_r2s.append(model.score(X_val, y_val))
    kfold_mean_r2 = np.mean(fold_r2s)
    results_kfold.append({**param_dict, "cv_r2": kfold_mean_r2})

# ✅ 결과 DataFrame으로 정리
df_split = pd.DataFrame(results_split).sort_values(by="test_r2", ascending=False).reset_index(drop=True)
df_kfold = pd.DataFrame(results_kfold).sort_values(by="cv_r2", ascending=False).reset_index(drop=True)

# ✅ 상위 10개 출력
print("\n📊 [train/test split 기준] 상위 10개:")
print(df_split.head(10)[param_names + ["train_r2", "test_r2"]])

print("\n📊 [KFold 기준] 상위 10개:")
print(df_kfold.head(10)[param_names + ["cv_r2"]])

# 결과

# 📊 [train/test split 기준] 상위 10개:
#    n_estimators  max_depth  learning_rate  subsample  ...  gamma  max_delta_step  train_r2   test_r2
# 0           160          6            0.2        1.0  ...      0               0  0.996669  0.845535 
# 1           200          6            0.2        1.0  ...      0               0  0.996669  0.845535 
# 2           200          6            0.2        1.0  ...      0               0  0.998171  0.840632 
# 3           160          6            0.2        1.0  ...      0               0  0.997578  0.840327 
# 4           200          5            0.2        1.0  ...      0               0  0.997019  0.837202 
# 5           200          6            0.2        1.0  ...      0               0  0.999603  0.837135 
# 6           200          6            0.1        1.0  ...      0               0  0.991749  0.836860 
# 7           160          6            0.2        1.0  ...      0               0  0.999323  0.836579 
# 8           200          6            0.2        1.0  ...      0               0  0.998373  0.836342 
# 9           200          6            0.2        1.0  ...      0               0  0.998407  0.835784 

# 📊 [KFold 기준] 상위 10개:
#    n_estimators  max_depth  learning_rate  ...  gamma  max_delta_step     cv_r2
# 0           200          6            0.2  ...      0               0  0.864761
# 1           160          6            0.2  ...      0               0  0.864470
# 2           200          6            0.2  ...      0               0  0.863650
# 3           160          6            0.2  ...      0               0  0.863597
# 4           200          6            0.1  ...      0               0  0.863039
# 5           200          6            0.2  ...      0               0  0.862827
# 6           200          6            0.2  ...      0               0  0.862574
# 7           200          6            0.2  ...      0               0  0.862368
# 8           160          6            0.2  ...      0               0  0.862335
# 9           160          6            0.2  ...      0               0  0.862088