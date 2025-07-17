# 하이퍼파라미터 튜닝 다시해보기
# 아까 추가했던 파생변수 추가까지 한 뒤에 학습
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import product

# ✅ 데이터 로딩
data_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(data_path)

# ✅ 파생변수 생성
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["edu_level_bucket"] = df["edu_level"] // 2  # 예: 0, 1, 2 같은 group

# ✅ 학습용 변수 설정
target_col = "years_until_ad"
exclude_cols = ["ad_year", "hhid_year"]
X = df.drop(columns=[target_col] + exclude_cols)
y = df[target_col]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ✅ 하이퍼파라미터 튜닝 조합 정의 (오버피팅 완화 위주)
param_grid = {
    "n_estimators": [140, 160, 180, 200, 220],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.05, 0.1, 0.15, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_alpha": [0.0, 0.005, 0.01],
    "reg_lambda": [0.5, 1, 2],
}

# ✅ 파라미터 조합 생성
param_combinations = list(product(*param_grid.values()))
param_names = list(param_grid.keys())

# ✅ 결과 저장용 리스트
results = []

# ✅ 모든 조합 반복
for params in param_combinations:
    param_dict = dict(zip(param_names, params))
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
        **param_dict
    )
    
    model.fit(X_train, y_train)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    results.append({
        **param_dict,
        "train_r2": round(train_r2, 6),
        "test_r2": round(test_r2, 6)
    })

# ✅ 결과 정렬 및 상위 10개 출력
results_df = pd.DataFrame(results)
top10 = results_df.sort_values(by="test_r2", ascending=False).head(10)
print("✅ XGBoost 정밀 튜닝 결과 (상위 10개)\n", top10.to_string(index=False))

# 결과
# ✅ XGBoost 정밀 튜닝 결과 (상위 10개)
#   n_estimators  max_depth  learning_rate  subsample  colsample_bytree  reg_alpha  reg_lambda  train_r2  test_r2
#           220          5           0.20        0.8               0.9      0.005         0.5  0.999459 0.831436
#           220          6           0.15        1.0               0.7      0.010         0.5  0.999272 0.831255
#           180          6           0.20        0.8               0.9      0.005         0.5  0.999905 0.831209
#           200          6           0.20        0.8               0.9      0.005         0.5  0.999955 0.831167
#           160          6           0.20        0.8               0.9      0.005         0.5  0.999799 0.831064
#           220          6           0.20        0.8               0.9      0.005         0.5  0.999977 0.831022
#           200          6           0.15        1.0               0.7      0.010         0.5  0.998889 0.831017
#           220          5           0.20        0.7               0.9      0.010         1.0  0.998981 0.830846
#           200          5           0.20        0.7               0.9      0.010         1.0  0.998482 0.830830
#           220          5           0.20        0.7               0.9      0.005         0.5  0.999359 0.830796