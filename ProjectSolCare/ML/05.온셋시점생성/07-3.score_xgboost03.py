# xgBoost 실험
# hyperParameter 튜닝해보기
# overFitting 해결중
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import product

# ✅ 1. 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/06.AD_train_error_fixed.csv"
df = pd.read_csv(file_path)

# ✅ 2. X, y 정의
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
missing_cols = [col for col in df.columns if 'missing' in col]

X = df.drop(columns=exclude_cols + [target_col] + missing_cols)
y = df[target_col]

X = X.dropna()
y = y.loc[X.index]

# ✅ 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 4. 하이퍼파라미터 조합
param_grid = {
    "n_estimators": [1000, 1500],
    "max_depth": [6, 7],
    "learning_rate": [0.03, 0.05],
    "reg_alpha": [0],
    "reg_lambda": [1],
}

results = []

# ✅ 5. 실험 실행
for n, d, lr, alpha, lam in product(*param_grid.values()):
    model = XGBRegressor(
        n_estimators=n,
        max_depth=d,
        learning_rate=lr,
        reg_alpha=alpha,
        reg_lambda=lam,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    results.append({
        "n_estimators": n,
        "max_depth": d,
        "learning_rate": lr,
        "reg_alpha": alpha,
        "reg_lambda": lam,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "model": model
    })

# ✅ 6. 상위 3개 출력
top_results = sorted(results, key=lambda x: x["test_r2"], reverse=True)[:3]

for i, res in enumerate(top_results, 1):
    print(f"\n🔹 Top {i}:")
    print(f"  - Train R² = {res['train_r2']:.4f}")
    print(f"  - Test  R² = {res['test_r2']:.4f}")
    print(f"  - n_estimators: {res['n_estimators']}")
    print(f"  - learning_rate: {res['learning_rate']}")
    print(f"  - max_depth: {res['max_depth']}")
    print(f"  - reg_alpha: {res['reg_alpha']}")
    print(f"  - reg_lambda: {res['reg_lambda']}")

    importances = pd.Series(res['model'].feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(5)
    print("  📌 중요 변수 Top 5:")
    print(top_features.to_string())

# 결과
# 🔹 Top 1:
#   - Train R² = 0.9461
#   - Test  R² = 0.0439
#   - n_estimators: 1000
#   - learning_rate: 0.03
#   - max_depth: 7
#   - reg_alpha: 0
#   - reg_lambda: 1
#   📌 중요 변수 Top 5:
# has_hibpe            0.455797
# years_until_hibpe    0.101557
# female_age           0.061280
# male_age             0.056647
# risk_weighted_age    0.050691

# 🔹 Top 2:
#   - Train R² = 0.9220
#   - Test  R² = 0.0384
#   - n_estimators: 1000
#   - learning_rate: 0.03
#   - max_depth: 6
#   - reg_alpha: 0
#   - reg_lambda: 1
#   📌 중요 변수 Top 5:
# has_hibpe            0.379610
# years_until_hibpe    0.082238
# female_age           0.065162
# male_age             0.060909
# edu_is_low           0.058455

# 🔹 Top 3:
#   - Train R² = 0.9490
#   - Test  R² = 0.0232
#   - n_estimators: 1000
#   - learning_rate: 0.05
#   - max_depth: 6
#   - reg_alpha: 0
#   - reg_lambda: 1
#   📌 중요 변수 Top 5:
# has_hibpe            0.414184
# years_until_hibpe    0.075596
# female_age           0.059016
# edu_is_low           0.055568
# male_age             0.054998