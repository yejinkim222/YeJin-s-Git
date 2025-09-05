# 랜덤 포레스트로 2가지 방식으로 depth=1~20 R² score 확인
# 1. 랜덤(80%/20%)
# 2. 2014 + 2016
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 데이터 로딩
file_path = (
    "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/03.AD_train_derived.csv"
)
df = pd.read_csv(file_path)

# 사용 변수
features = [
    "age",
    "gender",
    "edu_yrs",
    "has_db",
    "AD_MCI_status",
    "has_hibpe",
    "edu_level",
    "years_until_mci",
    "years_until_db",
    "years_until_hibpe",
    "years_until_hibpe_missing",
    "has_hibpe_missing",
    "years_until_mci_missing",
    "years_until_db_missing",
    "edu_yrs_missing",
    "age_group5",
    "risk_factor_sum",
    "edu_is_low",
    "risk_weighted_age",
    "male_age",
    "female_age",
]
target = "years_until_ad"

# 결과 저장용
results = {"random_split": [], "time_split_14_16": []}

# 1. 랜덤 분할 (train 80%, test 20%)
X_rand_train, X_rand_test, y_rand_train, y_rand_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

# 2. 시간 기반 분할: 2014+2016 → test
df_time_14_16 = df[df["year"].isin([2014, 2016])]
df_time_rest = df[~df["year"].isin([2014, 2016])]

# 공통 반복
for depth in range(1, 21):
    model = RandomForestRegressor(max_depth=depth, random_state=42)

    # 1. 랜덤
    model.fit(X_rand_train, y_rand_train)
    pred_rand = model.predict(X_rand_test)
    score_rand = r2_score(y_rand_test, pred_rand)
    mse_rand = mean_squared_error(y_rand_test, pred_rand)
    results["random_split"].append((depth, score_rand, mse_rand))

    # 2. 시간: 2014+2016 test
    model.fit(df_time_rest[features], df_time_rest[target])
    pred_14_16 = model.predict(df_time_14_16[features])
    score_14_16 = r2_score(df_time_14_16[target], pred_14_16)
    mse_14_16 = mean_squared_error(df_time_14_16[target], pred_14_16)
    results["time_split_14_16"].append((depth, score_14_16, mse_14_16))

# 출력
print("\nRandom 80/20 Split")
for d, s, m in results["random_split"]:
    print(f"  depth={d:2d} → R²: {s:.4f}, MSE: {m:.4f}")

print("\nTime Split (2014+2016 as Test)")
for d, s, m in results["time_split_14_16"]:
    print(f"  depth={d:2d} → R²: {s:.4f}, MSE: {m:.4f}")

# 출력
#  Random 80/20 Split
#   depth= 1 → R²: 0.0988
#   depth= 2 → R²: 0.1655
#   depth= 3 → R²: 0.2258
#   depth= 4 → R²: 0.2576
#   depth= 5 → R²: 0.2626
#   depth= 6 → R²: 0.2621
#   depth= 7 → R²: 0.2427
#   depth= 8 → R²: 0.2349
#   depth= 9 → R²: 0.2279
#   depth=10 → R²: 0.2265
#   depth=11 → R²: 0.2275
#   depth=12 → R²: 0.2156
#   depth=13 → R²: 0.2118
#   depth=14 → R²: 0.2072
#   depth=15 → R²: 0.2077
#   depth=16 → R²: 0.2011
#   depth=17 → R²: 0.2020
#   depth=18 → R²: 0.2005
#   depth=19 → R²: 0.2000
#   depth=20 → R²: 0.1978

# Time Split (2014+2016 as Test)
#   depth= 1 → R²: -186.4828
#   depth= 2 → R²: -167.5195
#   depth= 3 → R²: -150.0309
#   depth= 4 → R²: -138.1431
#   depth= 5 → R²: -133.1575
#   depth= 6 → R²: -130.7806
#   depth= 7 → R²: -126.4577
#   depth= 8 → R²: -133.8931
#   depth= 9 → R²: -134.9615
#   depth=10 → R²: -138.3431
#   depth=11 → R²: -134.0220
#   depth=12 → R²: -134.2645
#   depth=13 → R²: -132.0165
#   depth=14 → R²: -137.1262
#   depth=15 → R²: -135.1355
#   depth=16 → R²: -133.1469
#   depth=17 → R²: -132.9919
#   depth=18 → R²: -132.9886
#   depth=19 → R²: -133.9270
#   depth=20 → R²: -134.3245
