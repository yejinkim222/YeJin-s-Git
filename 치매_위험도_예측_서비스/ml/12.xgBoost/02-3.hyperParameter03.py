# 하이퍼파라미터 계속 튜닝해보기
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import product

# ✅ 데이터 로딩
data_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(data_path)

# ✅ 타겟 및 제외 컬럼 정의
target_col = "years_until_ad"
exclude_cols = ["ad_year", "hhid_year"]
X = df.drop(columns=[target_col] + exclude_cols)
y = df[target_col]

# ✅ 고정된 train/test 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ✅ 하이퍼파라미터 범위 설정
param_grid = {
    "n_estimators": [100, 140, 180, 220, 260],
    "max_depth": [5, 6, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [1, 2]
}

# ✅ 조합 생성
param_combinations = list(product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["learning_rate"],
    param_grid["subsample"],
    param_grid["colsample_bytree"],
    param_grid["reg_alpha"],
    param_grid["reg_lambda"]
))

# ✅ 결과 저장
results = []

# ✅ 모델 학습 및 성능 저장
for n, d, lr, ss, cs, alpha, lamb in param_combinations:
    model = XGBRegressor(
        n_estimators=n,
        max_depth=d,
        learning_rate=lr,
        subsample=ss,
        colsample_bytree=cs,
        reg_alpha=alpha,
        reg_lambda=lamb,
        random_state=42,
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=30,
        verbose=False
    )
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results.append({
        "n_estimators": n,
        "max_depth": d,
        "learning_rate": lr,
        "subsample": ss,
        "colsample_bytree": cs,
        "reg_alpha": alpha,
        "reg_lambda": lamb,
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred)
    })

# ✅ 상위 10개 결과 출력
results_df = pd.DataFrame(results)
top10 = results_df.sort_values(by="test_r2", ascending=False).head(10)

print("✅ XGBoost 정밀 튜닝 결과 (상위 10개):")
print(top10)

# 결과
# ✅ XGBoost 정밀 튜닝 결과 (상위 10개):
#       n_estimators  max_depth  learning_rate  subsample  ...  reg_alpha  reg_lambda  train_r2   test_r2
# 999            260          6            0.2        1.0  ...       0.01           2  0.999329  0.833498
# 783            220          6            0.2        1.0  ...       0.01           2  0.999217  0.833220
# 567            180          6            0.2        1.0  ...       0.01           2  0.997722  0.832970
# 1044           260          7            0.1        1.0  ...       0.00           1  0.998949  0.832755
# 424            140          7            0.2        1.0  ...       0.10           1  0.998983  0.831805
# 856            220          7            0.2        1.0  ...       0.10           1  0.998983  0.831805
# 1072           260          7            0.2        1.0  ...       0.10           1  0.998983  0.831805
# 640            180          7            0.2        1.0  ...       0.10           1  0.998983  0.831805
# 1046           260          7            0.1        1.0  ...       0.01           1  0.999090  0.831706
# 1045           260          7            0.1        1.0  ...       0.00           2  0.997572  0.831602