# 규제 강화해서 학습해보기
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from itertools import product

# ✅ 데이터 로딩 및 전처리
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")

# 파생변수 생성
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["edu_level_bucket"] = df["edu_level"] // 2

# 입력 및 타겟 정의
X = df.drop(columns=["years_until_ad", "ad_year", "hhid_year"])
y = df["years_until_ad"]
df_model = pd.concat([X, y], axis=1).dropna()
X = df_model.drop(columns=["years_until_ad"])
y = df_model["years_until_ad"]

# 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 튜닝 대상 조합
param_grid = {
    "reg_alpha": [0.01, 0.1, 1],
    "reg_lambda": [0.5, 1.0, 2.0],
    "gamma": [0.1, 0.3, 0.5],
    "min_child_weight": [5, 7, 10]
}

# 고정 파라미터
fixed_params = {
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_estimators": 200,
    "learning_rate": 0.2,
    "max_depth": 6,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "max_delta_step": 0
}

# 실험 조합 생성
keys, values = zip(*param_grid.items())
experiments = [dict(zip(keys, v)) for v in product(*values)]

# 결과 저장
results = []

# 실험 반복
for i, setting in enumerate(experiments, 1):
    params = {**fixed_params, **setting}
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    results.append({
        "experiment": f"{i}/{len(experiments)}",
        "train_r2": round(train_r2, 4),
        "test_r2": round(test_r2, 4),
        **setting
    })
    print(f"{i}/{len(experiments)}")

# 결과 정렬 및 상위 10개 출력
results_df = pd.DataFrame(results).sort_values(by="test_r2", ascending=False)
print("\n✅ Test R² 기준 상위 10개 조합:\n")
print(results_df.head(10).to_string(index=False))

# 결과
# ✅ Test R² 기준 상위 10개 조합:

# experiment  train_r2  test_r2  reg_alpha  reg_lambda  gamma  min_child_weight
#      16/81    0.9911   0.8356       0.01         1.0    0.5                 5
#      10/81    0.9969   0.8318       0.01         1.0    0.1                 5
#      11/81    0.9942   0.8267       0.01         1.0    0.1                 7
#      30/81    0.9909   0.8263       0.10         0.5    0.1                10
#      13/81    0.9948   0.8256       0.01         1.0    0.3                 5
#      75/81    0.9916   0.8249       1.00         2.0    0.1                10
#      31/81    0.9954   0.8247       0.10         0.5    0.3                 5
#      78/81    0.9885   0.8240       1.00         2.0    0.3                10
#      44/81    0.9898   0.8234       0.10         1.0    0.5                 7
#      71/81    0.9903   0.8229       1.00         1.0    0.5                 7