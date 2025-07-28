# 모델 복잡도 높이기
# 오버피팅 일어났다...
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 결측 마스킹은 이미 -1 처리된 상태라고 가정
# ※ hhid, hhid_year, year, target 제외
target_col = "years_until_ad"
exclude_cols = ["hhid", "hhid_year", "year", target_col]

X = df.drop(columns=exclude_cols)
y = df[target_col]

# ✅ train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 실험할 하이퍼파라미터 조합
param_grid = [
    {"name": "lgbm_restore1", "learning_rate": 0.02, "num_leaves": 80, "max_depth": 7, "reg_alpha": 0.3, "reg_lambda": 1.5},
    {"name": "lgbm_restore2", "learning_rate": 0.02, "num_leaves": 100, "max_depth": 8, "reg_alpha": 0.5, "reg_lambda": 2},
    {"name": "lgbm_restore3", "learning_rate": 0.03, "num_leaves": 120, "max_depth": 9, "reg_alpha": 0.3, "reg_lambda": 1},
    {"name": "lgbm_restore4", "learning_rate": 0.015, "num_leaves": 90, "max_depth": 7, "reg_alpha": 0.4, "reg_lambda": 2},
    {"name": "lgbm_restore5", "learning_rate": 0.02, "num_leaves": 110, "max_depth": 8, "reg_alpha": 0.2, "reg_lambda": 1.8},
]

results = []

for params in param_grid:
    model = lgb.LGBMRegressor(
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        n_estimators=1000,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top5_features = feature_importances.sort_values(ascending=False).head(5)

    results.append({
        "model": params["name"],
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4),
        "Top 5 Features": top5_features.to_dict(),
        "Params": params
    })

# ✅ 상위 5개만 출력
top_results = sorted(results, key=lambda x: x["Test R²"], reverse=True)[:5]

print("✅ LightGBM 상위 5개 결과 (Test R² 기준)\n")
for res in top_results:
    print(f"🔹 모델: {res['model']}")
    print(f"  - Train R² = {res['Train R²']}")
    print(f"  - Test  R² = {res['Test R²']}")
    print("  📌 중요 변수 Top 5:")
    for k, v in res["Top 5 Features"].items():
        print(f"    - {k}: {v:.6f}")
    print(f"  🔧 Params: {res['Params']}")
    print("-" * 50)

# 결과
# ✅ LightGBM 상위 5개 결과 (Test R² 기준)

# 🔹 모델: lgbm_restore4
#   - Train R² = 0.6194
#   - Test  R² = 0.2335
#   📌 중요 변수 Top 5:
#     - age: 3002.000000
#     - edu_yrs: 2858.000000
#     - risk_weighted_age: 2516.000000
#     - female_age: 2438.000000
#     - male_age: 1932.000000
#   🔧 Params: {'name': 'lgbm_restore4', 'learning_rate': 0.015, 'num_leaves': 90, 'max_depth': 7, 'reg_alpha': 0.4, 'reg_lambda': 2}
# --------------------------------------------------
# 🔹 모델: lgbm_restore2
#   - Train R² = 0.6839
#   - Test  R² = 0.2017
#   📌 중요 변수 Top 5:
#     - edu_yrs: 3570.000000
#     - age: 3356.000000
#     - risk_weighted_age: 2985.000000
#     - female_age: 2870.000000
#     - male_age: 2103.000000
#   🔧 Params: {'name': 'lgbm_restore2', 'learning_rate': 0.02, 'num_leaves': 100, 'max_depth': 8, 'reg_alpha': 0.5, 'reg_lambda': 2}
# --------------------------------------------------
# 🔹 모델: lgbm_restore1
#   - Train R² = 0.6629
#   - Test  R² = 0.1973
#   📌 중요 변수 Top 5:
#     - age: 3181.000000
#     - edu_yrs: 3103.000000
#     - female_age: 2680.000000
#     - risk_weighted_age: 2561.000000
#     - male_age: 1835.000000
#   🔧 Params: {'name': 'lgbm_restore1', 'learning_rate': 0.02, 'num_leaves': 80, 'max_depth': 7, 'reg_alpha': 0.3, 'reg_lambda': 1.5}
# --------------------------------------------------
# 🔹 모델: lgbm_restore5
#   - Train R² = 0.6823
#   - Test  R² = 0.1911
#   📌 중요 변수 Top 5:
#     - edu_yrs: 3418.000000
#     - age: 3417.000000
#     - risk_weighted_age: 2816.000000
#     - female_age: 2741.000000
#     - male_age: 2051.000000
#   🔧 Params: {'name': 'lgbm_restore5', 'learning_rate': 0.02, 'num_leaves': 110, 'max_depth': 8, 'reg_alpha': 0.2, 'reg_lambda': 1.8}
# --------------------------------------------------
# 🔹 모델: lgbm_restore3
#   - Train R² = 0.7627
#   - Test  R² = 0.1558
#   📌 중요 변수 Top 5:
#     - edu_yrs: 3704.000000
#     - age: 3497.000000
#     - risk_weighted_age: 3289.000000
#     - female_age: 3088.000000
#     - male_age: 2363.000000
#   🔧 Params: {'name': 'lgbm_restore3', 'learning_rate': 0.03, 'num_leaves': 120, 'max_depth': 9, 'reg_alpha': 0.3, 'reg_lambda': 1}