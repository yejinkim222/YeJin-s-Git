# xgBoost 실험
# hyperParameter 튜닝해보기
# overFitting 해결중
# has_hibpe 관련 변수들 영향이 너무 커서 없애보기
# 이제 언더피팅 일어나네...
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from itertools import product

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/06.AD_train_error_fixed.csv"
df = pd.read_csv(file_path)

# ✅ 변수 정의
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year', target_col]

# ✅ 방법 A: has_hibpe 제거
X_a = df.drop(columns=exclude_cols + ['has_hibpe']).dropna()
y_a = df.loc[X_a.index, target_col]

# ✅ 방법 B: has_hibpe + has_hibpe_missing만 사용
X_b = df[['has_hibpe', 'has_hibpe_missing']].copy().dropna()
y_b = df.loc[X_b.index, target_col]

# ✅ 하이퍼파라미터 조합
param_grid = {
    'learning_rate': [0.01],
    'n_estimators': [3000],
    'max_depth': [3, 5],
    'reg_alpha': [0],
    'reg_lambda': [1]
}
param_combinations = list(product(*param_grid.values()))

# ✅ 실험 함수
def run_experiment(X, y, tag):
    results = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for lr, est, depth, alpha, lambd in param_combinations:
        model = XGBRegressor(
            learning_rate=lr,
            n_estimators=est,
            max_depth=depth,
            reg_alpha=alpha,
            reg_lambda=lambd,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importances.sort_values(ascending=False).head(5)

        results.append({
            "tag": tag,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "params": {
                'n_estimators': est,
                'learning_rate': lr,
                'max_depth': depth,
                'reg_alpha': alpha,
                'reg_lambda': lambd
            },
            "top_features": top_features
        })
    return results

# ✅ 두 가지 방법 실행
results = []
results += run_experiment(X_a, y_a, '방법 A: has_hibpe 제거')
results += run_experiment(X_b, y_b, '방법 B: has_hibpe만 사용')

# ✅ Top 3 결과 출력
top3 = sorted(results, key=lambda x: x['test_r2'], reverse=True)[:3]

for i, res in enumerate(top3, 1):
    print(f"\n🔹 Top {i}:")
    print(f"  - Tag: {res['tag']}")
    print(f"  - Train R² = {res['train_r2']:.4f}")
    print(f"  - Test  R² = {res['test_r2']:.4f}")
    print(f"  - n_estimators: {res['params']['n_estimators']}")
    print(f"  - learning_rate: {res['params']['learning_rate']}")
    print(f"  - max_depth: {res['params']['max_depth']}")
    print(f"  - reg_alpha: {res['params']['reg_alpha']}")
    print(f"  - reg_lambda: {res['params']['reg_lambda']}")
    print("  📌 중요 변수 Top 5:")
    print(res["top_features"].to_string())

# 결과
# 🔹 Top 1:
#   - Tag: 방법 A: has_hibpe 제거
#   - Train R² = 0.6047
#   - Test  R² = 0.2169
#   - n_estimators: 3000
#   - learning_rate: 0.01
#   - max_depth: 3
#   - reg_alpha: 0
#   - reg_lambda: 1
#   📌 중요 변수 Top 5:
# has_hibpe_missing            0.301877
# years_until_db_missing       0.098467
# years_until_hibpe_missing    0.090919
# years_until_mci_missing      0.081874
# edu_is_low                   0.052800

# 🔹 Top 2:
#   - Tag: 방법 B: has_hibpe만 사용
#   - Train R² = 0.1177
#   - Test  R² = 0.1844
#   - n_estimators: 3000
#   - learning_rate: 0.01
#   - max_depth: 3
#   - reg_alpha: 0
#   - reg_lambda: 1
#   📌 중요 변수 Top 5:
# has_hibpe            1.0
# has_hibpe_missing    0.0

# 🔹 Top 3:
#   - Tag: 방법 B: has_hibpe만 사용
#   - Train R² = 0.1177
#   - Test  R² = 0.1844
#   - n_estimators: 3000
#   - learning_rate: 0.01
#   - max_depth: 5
#   - reg_alpha: 0
#   - reg_lambda: 1
#   📌 중요 변수 Top 5:
# has_hibpe            1.0
# has_hibpe_missing    0.0