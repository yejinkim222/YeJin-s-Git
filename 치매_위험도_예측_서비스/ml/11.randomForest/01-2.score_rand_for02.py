# 정밀 튜닝
# 오버피팅이었구..
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")

# ✅ 타겟 및 피처 정의
y = df["years_until_ad"]
X = df.drop(columns=["years_until_ad", "hhid", "year", "hhid_year", "ad_year"], errors="ignore")

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ✅ 세밀한 하이퍼파라미터 조합 (기존 베이스라인 기반)
param_grid = {
    "n_estimators": [80, 100, 120],
    "max_depth": [14, 15, 16],
    "min_samples_leaf": [1, 2],
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
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    r2_train = r2_score(y_train, train_preds)
    r2_test = r2_score(y_test, test_preds)
    
    results.append({
        "n_estimators": n_est,
        "max_depth": depth,
        "min_samples_leaf": min_leaf,
        "max_features": max_feat,
        "train_r2": round(r2_train, 5),
        "test_r2": round(r2_test, 5)
    })

# ✅ 결과 정렬
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="test_r2", ascending=False).reset_index(drop=True)

# ✅ 상위 10개 출력
print("✅ RandomForest 세밀 튜닝 결과 (train/test R² 함께 출력):")
print(results_df.head(10))

# 결과
