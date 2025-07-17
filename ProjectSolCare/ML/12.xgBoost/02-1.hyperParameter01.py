# 언더피팅 생겨서
# 하이퍼파라미터 튜닝해보기
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import product

# ✅ 데이터 경로 및 로딩
data_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(data_path)

# ✅ 타겟 및 제외 컬럼 설정
target_col = "years_until_ad"
exclude_cols = ["ad_year", "hhid_year"]  # 불필요한 컬럼

# ✅ 피처/타겟 분리
X = df.drop(columns=[target_col] + exclude_cols)
y = df[target_col]

# ✅ 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ✅ 하이퍼파라미터 조합 정의
param_grid = {
    "n_estimators": [80, 100, 120],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# ✅ 조합 생성
combinations = list(product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["learning_rate"],
    param_grid["subsample"],
    param_grid["colsample_bytree"]
))

results = []

# ✅ 전체 조합 반복
for n, d, lr, subsample, colsample in combinations:
    model = XGBRegressor(
        n_estimators=n,
        max_depth=d,
        learning_rate=lr,
        subsample=subsample,
        colsample_bytree=colsample,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "n_estimators": n,
        "max_depth": d,
        "learning_rate": lr,
        "subsample": subsample,
        "colsample_bytree": colsample,
        "train_r2": model.score(X_train, y_train),
        "test_r2": r2_score(y_test, y_pred)
    })

# ✅ 결과 정리
results_df = pd.DataFrame(results)
top10 = results_df.sort_values(by="test_r2", ascending=False).head(10)

# ✅ 상위 10개 조합 출력
print("✅ XGBoost 하이퍼파라미터 튜닝 결과 (상위 10개):")
print(top10)

# 결과
# ✅ XGBoost 하이퍼파라미터 튜닝 결과 (상위 10개):
#      n_estimators  max_depth  learning_rate  subsample  colsample_bytree  train_r2   test_r2
# 106           120          6            0.2        1.0               0.8  0.995779  0.824317
# 70            100          6            0.2        1.0               0.8  0.992737  0.823010
# 34             80          6            0.2        1.0               0.8  0.988015  0.821295
# 102           120          6            0.1        1.0               0.8  0.975110  0.810759
# 66            100          6            0.1        1.0               0.8  0.965941  0.810363
# 104           120          6            0.2        0.8               0.8  0.997878  0.809495
# 68            100          6            0.2        0.8               0.8  0.996192  0.807203
# 100           120          6            0.1        0.8               0.8  0.983874  0.806771
# 30             80          6            0.1        1.0               0.8  0.955856  0.806138
# 93            120          5            0.2        0.8               1.0  0.991558  0.805746
