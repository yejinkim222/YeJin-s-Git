# 성능 향상된 파생변수 포함해서
# 베이스라인 확인하고
# 하이퍼 파라미터 튜닝해서 비교해보기
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import itertools

# ✅ 데이터 경로
data_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(data_path)

# ✅ 파생변수 (성능 향상 확인된 3개)
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / df["age"]
df["edu_level_bucket"] = df["edu_level"].map({0: 0, 1: 1, 2: 1, 3: 2})

# ✅ 학습/타겟 변수 설정
target_col = "years_until_ad"
exclude_cols = ["ad_year", "hhid_year"]
X = df.drop(columns=[target_col] + exclude_cols)
y = df[target_col]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ✅ 베이스라인 모델 (디폴트 XGBRegressor)
baseline_model = XGBRegressor(random_state=42)
baseline_model.fit(X_train, y_train)

y_pred_train = baseline_model.predict(X_train)
y_pred_test = baseline_model.predict(X_test)
print("✅ XGBoost 베이스라인")
print(f"Train R²: {r2_score(y_train, y_pred_train):.4f}, Test R²: {r2_score(y_test, y_pred_test):.4f}")

# ✅ 하이퍼파라미터 후보 설정
param_grid = {
    "n_estimators": [100, 140, 180, 220],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8],
    "reg_alpha": [0.0, 0.01],
    "reg_lambda": [1, 2]
}

# ✅ 조합 생성 및 실험 실행
results = []

for params in itertools.product(*param_grid.values()):
    param_dict = dict(zip(param_grid.keys(), params))

    model = XGBRegressor(**param_dict, random_state=42, verbosity=0)
    model.fit(X_train, y_train)

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    results.append({
        **param_dict,
        "train_r2": train_r2,
        "test_r2": test_r2
    })

# ✅ 결과 DataFrame 정렬 및 상위 10개 출력
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="test_r2", ascending=False)

print("\n✅ XGBoost 튜닝 결과 (상위 10개)")
print(results_df.head(10).to_string(index=False))

# 결과
# ✅ XGBoost 베이스라인
# Train R²: 0.9987, Test R²: 0.7823

# ✅ XGBoost 튜닝 결과 (상위 10개)
#  n_estimators  max_depth  learning_rate  subsample  colsample_bytree  reg_alpha  reg_lambda  train_r2  test_r2
#           220          6            0.2        1.0               0.8       0.00           1  0.999755 0.828941
#           180          6            0.2        1.0               0.8       0.01           1  0.999486 0.828496
#           180          6            0.2        1.0               0.8       0.00           1  0.999388 0.828332
#           220          6            0.2        1.0               0.8       0.01           1  0.999845 0.827986
#           140          6            0.2        1.0               0.8       0.01           1  0.998501 0.827612
#           140          6            0.2        1.0               0.8       0.00           1  0.998244 0.827247
#           100          6            0.2        1.0               0.8       0.00           1  0.994679 0.825656
#           220          6            0.1        1.0               0.8       0.00           2  0.992531 0.825421
#           100          6            0.2        1.0               0.8       0.01           1  0.995588 0.825092
#           220          5            0.2        1.0               0.8       0.01           2  0.996403 0.823533