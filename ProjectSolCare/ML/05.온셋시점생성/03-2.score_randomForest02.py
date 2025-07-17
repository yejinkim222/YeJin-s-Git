# randomForest hyperParameter 조정
# overfitting 아니고 underfitting이었음..
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import product

# ✅ 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/05.AD_train_add_row.csv"
df = pd.read_csv(file_path)

# ✅ 전처리
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]
X = X.dropna()
y = y.loc[X.index]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 하이퍼파라미터 조합
n_estimators_list = [50, 100]
max_depth_list = [3, 5, 10, None]
min_samples_leaf_list = [1, 5, 10]

# ✅ 결과 저장 리스트
results = []

# ✅ 전체 실험 (출력 없음)
for n_estimators, max_depth, min_samples_leaf in product(n_estimators_list, max_depth_list, min_samples_leaf_list):
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_test_pred = rf.predict(X_test)
    test_score = r2_score(y_test, y_test_pred)

    results.append({
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf,
        'R2': round(test_score, 4),
        'model': rf
    })

# ✅ 상위 3개 결과만 출력 (이때 train score도 계산)
top_results = sorted(results, key=lambda x: x['R2'], reverse=True)[:3]

for i, res in enumerate(top_results, 1):
    model = res['model']
    y_train_pred = model.predict(X_train)
    train_score = r2_score(y_train, y_train_pred)

    print(f"\n🔹 Top {i}:")
    print(f"  - Train R² = {train_score:.4f}")
    print(f"  - Test R²  = {res['R2']:.4f}")
    print(f"  - n_estimators: {res['n_estimators']}")
    print(f"  - max_depth: {res['max_depth']}")
    print(f"  - min_samples_leaf: {res['min_samples_leaf']}")

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(5)
    print("  📌 중요 변수 Top 5:")
    print(top_features.to_string())

# 결과

# 🔹 Top 1:
#   - Train R² = 0.3498
#   - Test R²  = 0.3020
#   - n_estimators: 50
#   - max_depth: 5
#   - min_samples_leaf: 5
#   📌 중요 변수 Top 5:
# age                  0.182086
# has_hibpe            0.177904
# has_hibpe_missing    0.172002
# female_age           0.092769
# male_age             0.060164

# 🔹 Top 2:
#   - Train R² = 0.3498
#   - Test R²  = 0.2971
#   - n_estimators: 100
#   - max_depth: 5
#   - min_samples_leaf: 5
#   📌 중요 변수 Top 5:
# age                  0.196152
# has_hibpe_missing    0.189351
# has_hibpe            0.145809
# female_age           0.092059
# male_age             0.063582

# 🔹 Top 3:
#   - Train R² = 0.3869
#   - Test R²  = 0.2962
#   - n_estimators: 100
#   - max_depth: 5
#   - min_samples_leaf: 1
#   📌 중요 변수 Top 5:
# age                  0.189064
# has_hibpe_missing    0.172482
# has_hibpe            0.132432
# female_age           0.089300
# male_age             0.064536