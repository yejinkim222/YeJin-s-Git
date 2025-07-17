# randomforest 튜닝
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")

# ✅ 타겟 / 피처 분리
y = df["years_until_ad"]
X = df.drop(columns=["years_until_ad", "hhid", "year", "hhid_year", "ad_year"], errors="ignore")

# ✅ 학습/검증 세트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ✅ 하이퍼파라미터 조합 정의 (넓게 탐색)
param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, 15],
    "min_samples_leaf": [1, 3, 5],
    "max_features": ["sqrt", "log2"]
}

# ✅ 모든 조합 생성
param_combinations = list(product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["min_samples_leaf"],
    param_grid["max_features"]
))

# ✅ 결과 저장 리스트
results = []

# ✅ 반복 실행
for n_est, depth, min_leaf, max_feat in param_combinations:
    model = RandomForestRegressor(
        n_estimators=n_est,
        max_depth=depth,
        min_samples_leaf=min_leaf,
        max_features=max_feat,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    
    results.append({
        "n_estimators": n_est,
        "max_depth": depth,
        "min_samples_leaf": min_leaf,
        "max_features": max_feat,
        "test_r2": r2
    })

# ✅ 결과 정렬 및 출력
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="test_r2", ascending=False).reset_index(drop=True)
print("✅ RandomForest 튜닝 결과 상위 10 조합:")
print(results_df.head(10))

# 결과
# ✅ RandomForest 튜닝 결과 상위 10 조합:
#    n_estimators  max_depth  min_samples_leaf max_features   test_r2
# 0           100         15                 1         log2  0.639967
# 1           100         15                 1         sqrt  0.639967
# 2           300         15                 1         sqrt  0.633817
# 3           300         15                 1         log2  0.633817
# 4           500         15                 1         log2  0.630292
# 5           500         15                 1         sqrt  0.630292
# 6           100         10                 1         log2  0.577526
# 7           100         10                 1         sqrt  0.577526
# 8           300         10                 1         log2  0.569930
# 9           300         10                 1         sqrt  0.569930