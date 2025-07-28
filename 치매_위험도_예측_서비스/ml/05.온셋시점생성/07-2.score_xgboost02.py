# xgBoost 실험
# hyperParameter 튜닝해보기
# overFitting 해결중
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import product

# ✅ 1. 데이터 로드
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/06.AD_train_error_fixed.csv"
df = pd.read_csv(file_path)

# ✅ 2. X, y 분리
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

X = X.dropna()
y = y.loc[X.index]

# ✅ 3. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 4. 하이퍼파라미터 조합 설정
param_grid = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.03, 0.05],
    'max_depth': [3, 5],
    'subsample': [0.7],
    'colsample_bytree': [0.7],
    'reg_alpha': [0, 1],
    'reg_lambda': [1]
}

results = []
for n, lr, d, ss, cs, a, l in product(*param_grid.values()):
    model = xgb.XGBRegressor(
        n_estimators=n,
        learning_rate=lr,
        max_depth=d,
        subsample=ss,
        colsample_bytree=cs,
        reg_alpha=a,
        reg_lambda=l,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              early_stopping_rounds=30,
              verbose=False)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    results.append({
        'n_estimators': n,
        'learning_rate': lr,
        'max_depth': d,
        'reg_alpha': a,
        'reg_lambda': l,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'model': model
    })

# ✅ 5. 상위 3개 결과 출력
top_results = sorted(results, key=lambda x: x['test_r2'], reverse=True)[:3]

for i, res in enumerate(top_results, 1):
    model = res['model']
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(5)

    print(f"\n🔹 Top {i}:")
    print(f"  - Train R² = {res['train_r2']:.4f}")
    print(f"  - Test  R² = {res['test_r2']:.4f}")
    print(f"  - n_estimators: {res['n_estimators']}")
    print(f"  - learning_rate: {res['learning_rate']}")
    print(f"  - max_depth: {res['max_depth']}")
    print(f"  - reg_alpha: {res['reg_alpha']}")
    print(f"  - reg_lambda: {res['reg_lambda']}")
    print("  📌 중요 변수 Top 5:")
    print(top_features.to_string())

# 결과
# 🔹 Top 1:
#   - Train R² = 0.4321
#   - Test  R² = 0.3010
#   - n_estimators: 500
#   - learning_rate: 0.05
#   - max_depth: 3
#   - reg_alpha: 0
#   - reg_lambda: 1
#   📌 중요 변수 Top 5:
# has_hibpe_missing            0.193372
# has_hibpe                    0.144233
# years_until_db_missing       0.072820
# years_until_hibpe_missing    0.055846
# years_until_hibpe            0.053179

# 🔹 Top 2:
#   - Train R² = 0.4321
#   - Test  R² = 0.3010
#   - n_estimators: 1000
#   - learning_rate: 0.05
#   - max_depth: 3
#   - reg_alpha: 0
#   - reg_lambda: 1
#   📌 중요 변수 Top 5:
# has_hibpe_missing            0.193372
# has_hibpe                    0.144233
# years_until_db_missing       0.072820
# years_until_hibpe_missing    0.055846
# years_until_hibpe            0.053179

# 🔹 Top 3:
#   - Train R² = 0.4265
#   - Test  R² = 0.2954
#   - n_estimators: 500
#   - learning_rate: 0.05
#   - max_depth: 3
#   - reg_alpha: 1
#   - reg_lambda: 1
#   📌 중요 변수 Top 5:
# has_hibpe_missing            0.193001
# has_hibpe                    0.142521
# years_until_db_missing       0.076499
# years_until_hibpe_missing    0.058812
# years_until_hibpe            0.054750