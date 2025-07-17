# 하이퍼파라미터 튜닝 실험
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/10.AD_train_xgboost_filled.csv"
df = pd.read_csv(file_path)

# 📌 제외 컬럼 및 타겟 설정
exclude_cols = ['hhid', 'year', 'hhid_year', 'AD_MCI_status', 'edu_yrs', 'edu_level', 'years_until_mci', 'gender']
target_col = 'years_until_ad'

X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

# 📌 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 튜닝할 하이퍼파라미터 그리드 정의
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 10, 15],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 'log2']
}

# 📌 조합 생성
all_params = list(product(
    param_grid['n_estimators'],
    param_grid['max_depth'],
    param_grid['min_samples_leaf'],
    param_grid['max_features']
))

print(f"🔍 총 튜닝 조합 수: {len(all_params)}")

# 📌 결과 저장
results = []

for i, (n_est, max_d, min_leaf, max_feat) in enumerate(all_params):
    model = RandomForestRegressor(
        n_estimators=n_est,
        max_depth=max_d,
        min_samples_leaf=min_leaf,
        max_features=max_feat,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'n_estimators': n_est,
        'max_depth': max_d,
        'min_samples_leaf': min_leaf,
        'max_features': max_feat,
        'test_r2': r2
    })

    print(f"[{i+1:03d}/{len(all_params)}] R²: {r2:.4f} | n={n_est}, depth={max_d}, leaf={min_leaf}, feat={max_feat}")

# 📌 결과 정리
results_df = pd.DataFrame(results)
results_df.sort_values(by='test_r2', ascending=False, inplace=True)

top_n = 10
print(f"\n✅ RandomForest 튜닝 결과 상위 {top_n} 조합:")
print(results_df.head(top_n).to_string(index=False))

# 결과
# ✅ RandomForest 튜닝 결과 상위 10 조합:
#  n_estimators  max_depth  min_samples_leaf max_features  test_r2
#           500         10                 1         sqrt 0.296674
#           500         10                 1         log2 0.296674
#           500         10                 3         log2 0.294957
#           500         10                 3         sqrt 0.294957
#           300         10                 3         sqrt 0.289263
#           300         10                 3         log2 0.289263
#           500         10                 5         log2 0.288973
#           500         10                 5         sqrt 0.288973
#           100         10                 1         sqrt 0.286450
#           100         10                 1         log2 0.286450