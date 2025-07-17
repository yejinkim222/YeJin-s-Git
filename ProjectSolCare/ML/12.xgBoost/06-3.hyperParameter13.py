# 오버피팅 완화하기
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from itertools import product

# ✅ 데이터 로딩 및 전처리
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")

# 파생변수 생성
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["edu_level_bucket"] = df["edu_level"] // 2

# 모델 입력 피처 및 타겟 정의
X = df.drop(columns=["years_until_ad", "ad_year", "hhid_year"])
y = df["years_until_ad"]
df_model = pd.concat([X, y], axis=1).dropna()
X = df_model.drop(columns=["years_until_ad"])
y = df_model["years_until_ad"]

# 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 튜닝 대상 파라미터 그리드
param_grid = {
    "max_depth": [4, 5, 6],
    "min_child_weight": [3, 5],
    "gamma": [0, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.6, 0.8]
}

# 고정 파라미터
fixed_params = {
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_estimators": 200,
    "learning_rate": 0.2,
    "reg_alpha": 0.005,
    "reg_lambda": 0.5,
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
# experiment  train_r2  test_r2  max_depth  min_child_weight  gamma  subsample  colsample_bytree
#      36/48    0.9993   0.8337          6                 3    0.0        1.0               0.8       
#      40/48    0.9977   0.8267          6                 3    0.1        1.0               0.8       
#      20/48    0.9961   0.8219          5                 3    0.0        1.0               0.8       
#      24/48    0.9965   0.8213          5                 3    0.1        1.0               0.8       
#      17/48    0.9964   0.8212          5                 3    0.0        0.8               0.6       
#      34/48    0.9997   0.8200          6                 3    0.0        0.8               0.8       
#      27/48    0.9912   0.8199          5                 5    0.0        1.0               0.6       
#       4/48    0.9849   0.8198          4                 3    0.0        1.0               0.8       
#      48/48    0.9976   0.8193          6                 5    0.1        1.0               0.8       
#      44/48    0.9979   0.8183          6                 5    0.0        1.0               0.8