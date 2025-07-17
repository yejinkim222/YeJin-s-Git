# 하이퍼 파라미터 튜닝해보기
# 기본 컬럼 유지, 제외하고 확인하기
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import product

# ✅ 데이터 경로 및 로딩
data_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(data_path)

# ✅ 타겟/피처 분리
y = df["years_until_ad"]
X = df.drop(columns=["years_until_ad"])

# ✅ 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ✅ 미세 조정 범위 설정 (베스트 모델 기준: 100, depth=15, leaf=1, log2)
param_grid = {
    "n_estimators": [80, 100, 120],
    "max_depth": [13, 14, 15],
    "min_samples_leaf": [1, 2],
    "min_samples_split": [2, 3],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False],
    "criterion": ["squared_error", "absolute_error"]
}

# ✅ 모든 조합 생성
param_combinations = list(product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["min_samples_leaf"],
    param_grid["min_samples_split"],
    param_grid["max_features"],
    param_grid["bootstrap"],
    param_grid["criterion"]
))

# ✅ 결과 저장
results = []

# ✅ 반복 학습 및 평가
for n, d, leaf, split, feat, boot, crit in param_combinations:
    model = RandomForestRegressor(
        n_estimators=n,
        max_depth=d,
        min_samples_leaf=leaf,
        min_samples_split=split,
        max_features=feat,
        bootstrap=boot,
        criterion=crit,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    train_r2 = model.score(X_train, y_train)

    results.append({
        "n_estimators": n,
        "max_depth": d,
        "min_samples_leaf": leaf,
        "min_samples_split": split,
        "max_features": feat,
        "bootstrap": boot,
        "criterion": crit,
        "train_r2": train_r2,
        "test_r2": test_r2
    })

# ✅ 상위 10개 출력
results_df = pd.DataFrame(results)
top10 = results_df.sort_values(by="test_r2", ascending=False).head(10)
print("✅ RandomForest 정밀 튜닝 결과 (상위 10개):")
print(top10)

# 결과
# ✅ RandomForest 정밀 튜닝 결과 (상위 10개):
#      n_estimators  max_depth  min_samples_leaf  ...       criterion  train_r2   test_r2
# 75             80         15                 1  ...  absolute_error  0.998602  0.957699
# 171           100         15                 1  ...  absolute_error  0.998587  0.957657
# 66             80         15                 1  ...   squared_error  0.999832  0.956342
# 162           100         15                 1  ...   squared_error  0.999832  0.956152
# 267           120         15                 1  ...  absolute_error  0.998479  0.956105
# 258           120         15                 1  ...   squared_error  0.999851  0.954566
# 131           100         14                 1  ...  absolute_error  0.998220  0.953834
# 170           100         15                 1  ...   squared_error  0.999358  0.953695
# 35             80         14                 1  ...  absolute_error  0.998279  0.953419
# 259           120         15                 1  ...  absolute_error  0.998622  0.953147