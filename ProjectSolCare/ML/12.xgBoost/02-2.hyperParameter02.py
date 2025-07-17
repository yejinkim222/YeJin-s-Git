# 살짝의 오버피팅 있어서
# 하이퍼파라미터 좀 더 튜닝
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import product

# ✅ 데이터 로딩
data_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(data_path)

# ✅ 타겟 및 피처 분리
target_col = "years_until_ad"
exclude_cols = ["ad_year", "hhid_year"]
X = df.drop(columns=[target_col] + exclude_cols)
y = df[target_col]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ✅ 정밀 하이퍼파라미터 튜닝 그리드 정의
param_grid = {
    "n_estimators": [100, 120, 140],
    "max_depth": [5, 6],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma": [0, 0.1, 0.5],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [1, 2]
}

# ✅ 파라미터 조합 생성
param_combinations = list(product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["learning_rate"],
    param_grid["subsample"],
    param_grid["colsample_bytree"],
    param_grid["gamma"],
    param_grid["reg_alpha"],
    param_grid["reg_lambda"]
))

results = []

# ✅ 반복 학습 및 평가
for params in param_combinations:
    model = XGBRegressor(
        n_estimators=params[0],
        max_depth=params[1],
        learning_rate=params[2],
        subsample=params[3],
        colsample_bytree=params[4],
        gamma=params[5],
        reg_alpha=params[6],
        reg_lambda=params[7],
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    train_r2 = model.score(X_train, y_train)
    results.append({
        "n_estimators": params[0],
        "max_depth": params[1],
        "learning_rate": params[2],
        "subsample": params[3],
        "colsample_bytree": params[4],
        "gamma": params[5],
        "reg_alpha": params[6],
        "reg_lambda": params[7],
        "train_r2": train_r2,
        "test_r2": test_r2
    })

# ✅ 상위 10개 결과 출력
results_df = pd.DataFrame(results)
top10 = results_df.sort_values(by="test_r2", ascending=False).head(10)

print("✅ XGBoost 정밀 튜닝 결과 (상위 10개):")
print(top10)

# 결과
# ✅ XGBoost 정밀 튜닝 결과 (상위 10개):
#       n_estimators  max_depth  learning_rate  subsample  ...  reg_alpha  reg_lambda  train_r2   test_r2
# 1268           140          6            0.2        1.0  ...       0.01           1  0.997242  0.834668
# 836            120          6            0.2        1.0  ...       0.01           1  0.995651  0.833693
# 1263           140          6            0.2        1.0  ...       0.01           2  0.995910  0.831278
# 404            100          6            0.2        1.0  ...       0.01           1  0.992397  0.830970
# 1261           140          6            0.2        1.0  ...       0.00           2  0.995964  0.830405
# 831            120          6            0.2        1.0  ...       0.01           2  0.993121  0.829626
# 1260           140          6            0.2        1.0  ...       0.00           1  0.997606  0.826098
# 1247           140          6            0.2        0.8  ...       0.10           2  0.998223  0.825437
# 829            120          6            0.2        1.0  ...       0.00           2  0.992752  0.825374
# 397            100          6            0.2        1.0  ...       0.00           2  0.988859  0.824363
