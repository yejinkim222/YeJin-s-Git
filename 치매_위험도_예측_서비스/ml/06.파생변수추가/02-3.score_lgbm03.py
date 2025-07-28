# 오버피팅 해결하기
# 깊이 줄이고 느리고 정밀하게 학습하기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ target 및 예외 컬럼 지정
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

# ✅ train/test split (결측치 drop 안 함!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 실험 파라미터 세트 (오버피팅 방지 목적)
param_grid = [
    {"name": "lgbm_tune1", "learning_rate": 0.01, "num_leaves": 60, "max_depth": 5, "reg_alpha": 0.6, "reg_lambda": 3},
    {"name": "lgbm_tune2", "learning_rate": 0.01, "num_leaves": 50, "max_depth": 4, "reg_alpha": 0.8, "reg_lambda": 3.5},
    {"name": "lgbm_tune3", "learning_rate": 0.015, "num_leaves": 60, "max_depth": 5, "reg_alpha": 0.6, "reg_lambda": 2.5},
    {"name": "lgbm_tune4", "learning_rate": 0.01, "num_leaves": 70, "max_depth": 5, "reg_alpha": 0.4, "reg_lambda": 2.5,
     "subsample": 0.8, "colsample_bytree": 0.8},
    {"name": "lgbm_tune5", "learning_rate": 0.008, "num_leaves": 60, "max_depth": 5, "reg_alpha": 0.7, "reg_lambda": 3.2}
]

# ✅ 결과 저장
results = []

for params in param_grid:
    model_params = {
        'learning_rate': params['learning_rate'],
        'num_leaves': params['num_leaves'],
        'max_depth': params['max_depth'],
        'reg_alpha': params['reg_alpha'],
        'reg_lambda': params['reg_lambda'],
        'random_state': 42
    }
    if 'subsample' in params:
        model_params['subsample'] = params['subsample']
    if 'colsample_bytree' in params:
        model_params['colsample_bytree'] = params['colsample_bytree']

    model = lgb.LGBMRegressor(**model_params)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(5)

    results.append({
        "모델": params["name"],
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4),
        "Top 5 Features": top_features.to_dict(),
        "Params": params
    })

# ✅ 상위 5개만 출력
results = sorted(results, key=lambda x: x['Test R²'], reverse=True)[:5]

print("\n✅ LightGBM 상위 5개 결과 (Test R² 기준)\n")
for res in results:
    print(f"🔹 모델: {res['모델']}")
    print(f"  - Train R² = {res['Train R²']}")
    print(f"  - Test  R² = {res['Test R²']}")
    print("  📌 중요 변수 Top 5:")
    for k, v in res["Top 5 Features"].items():
        print(f"    - {k}: {v:.6f}")
    print(f"  🔧 Params: {res['Params']}")
    print("-" * 50)

# 결과
# ✅ LightGBM 상위 5개 결과 (Test R² 기준)

# 🔹 모델: lgbm_tune3
#   - Train R² = 0.2685
#   - Test  R² = 0.2931
#   📌 중요 변수 Top 5:
#     - age: 259.000000
#     - female_age: 214.000000
#     - edu_yrs: 157.000000
#     - male_age: 142.000000
#     - years_until_db: 130.000000
#   🔧 Params: {'name': 'lgbm_tune3', 'learning_rate': 0.015, 'num_leaves': 60, 'max_depth': 5, 'reg_alpha': 0.6, 'reg_lambda': 2.5}
# --------------------------------------------------
# 🔹 모델: lgbm_tune1
#   - Train R² = 0.2234
#   - Test  R² = 0.2481
#   📌 중요 변수 Top 5:
#     - age: 295.000000
#     - female_age: 225.000000
#     - edu_yrs: 171.000000
#     - male_age: 124.000000
#     - years_until_db: 117.000000
#   🔧 Params: {'name': 'lgbm_tune1', 'learning_rate': 0.01, 'num_leaves': 60, 'max_depth': 5, 'reg_alpha': 0.6, 'reg_lambda': 3}
# --------------------------------------------------
# 🔹 모델: lgbm_tune4
#   - Train R² = 0.2243
#   - Test  R² = 0.246
#   📌 중요 변수 Top 5:
#     - age: 263.000000
#     - female_age: 219.000000
#     - edu_yrs: 128.000000
#     - male_age: 124.000000
#     - years_until_db: 93.000000
#   🔧 Params: {'name': 'lgbm_tune4', 'learning_rate': 0.01, 'num_leaves': 70, 'max_depth': 5, 'reg_alpha': 0.4, 'reg_lambda': 2.5, 'subsample': 0.8, 'colsample_bytree': 0.8}
# --------------------------------------------------
# 🔹 모델: lgbm_tune2
#   - Train R² = 0.2086
#   - Test  R² = 0.2372
#   📌 중요 변수 Top 5:
#     - age: 253.000000
#     - female_age: 175.000000
#     - has_hibpe: 100.000000
#     - edu_yrs: 89.000000
#     - years_until_db: 83.000000
#   🔧 Params: {'name': 'lgbm_tune2', 'learning_rate': 0.01, 'num_leaves': 50, 'max_depth': 4, 'reg_alpha': 0.8, 'reg_lambda': 3.5}
# --------------------------------------------------
# 🔹 모델: lgbm_tune5
#   - Train R² = 0.1969
#   - Test  R² = 0.2138
#   📌 중요 변수 Top 5:
#     - age: 303.000000
#     - female_age: 234.000000
#     - edu_yrs: 169.000000
#     - male_age: 112.000000
#     - has_hibpe: 100.000000
#   🔧 Params: {'name': 'lgbm_tune5', 'learning_rate': 0.008, 'num_leaves': 60, 'max_depth': 5, 'reg_alpha': 0.7, 'reg_lambda': 3.2}
