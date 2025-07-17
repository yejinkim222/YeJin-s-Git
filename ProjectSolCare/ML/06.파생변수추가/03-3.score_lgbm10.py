# lgbm 실험
# 재현 가능한 무작위성
# age_group5 10 이상
# 얼리스타핑
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 불러오기
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv")

# ✅ 조건: age_group5 ≥ 10
df = df[df["age_group5"] >= 10].copy()

# ✅ feature / target 분리
target_col = "years_until_ad"
exclude_cols = ["hhid", "hhid_year", "year"]
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

# ✅ 학습/테스트 분할 (재현 가능성 보장)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 실험 파라미터들
param_list = [
    {"name": "lgbm_boost1", "learning_rate": 0.01, "num_leaves": 80, "max_depth": 7, "reg_alpha": 0.4, "reg_lambda": 1.5, "subsample": 0.9, "colsample_bytree": 0.9},
    {"name": "lgbm_boost2", "learning_rate": 0.005, "num_leaves": 100, "max_depth": 8, "reg_alpha": 0.6, "reg_lambda": 2.0, "subsample": 0.8, "colsample_bytree": 0.8},
    {"name": "lgbm_boost3", "learning_rate": 0.01, "num_leaves": 120, "max_depth": 9, "reg_alpha": 0.5, "reg_lambda": 2.5, "subsample": 1.0, "colsample_bytree": 0.8},
    {"name": "lgbm_boost4", "learning_rate": 0.008, "num_leaves": 60, "max_depth": 6, "reg_alpha": 0.3, "reg_lambda": 1.2, "subsample": 0.8, "colsample_bytree": 1.0},
    {"name": "lgbm_boost5", "learning_rate": 0.007, "num_leaves": 110, "max_depth": 7, "reg_alpha": 0.6, "reg_lambda": 3.0, "subsample": 0.85, "colsample_bytree": 0.9},
]

# ✅ 결과 저장
results = []

for params in param_list:
    model = lgb.LGBMRegressor(
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=42,
        n_estimators=2000
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

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

# ✅ 상위 5개 출력
results_sorted = sorted(results, key=lambda x: x["Test R²"], reverse=True)
for result in results_sorted[:5]:
    print(f"\n🔹 모델: {result['모델']}")
    print(f"  - Train R² = {result['Train R²']}")
    print(f"  - Test  R² = {result['Test R²']}")
    print(f"  📌 중요 변수 Top 5:")
    for k, v in result["Top 5 Features"].items():
        print(f"    - {k}: {v}")
    print(f"  🔧 Params: {result['Params']}")

# 결과
# 🔹 모델: lgbm_boost4
#   - Train R² = 0.3974
#   - Test  R² = 0.3225
#   📌 중요 변수 Top 5:
#     - age: 1104
#     - edu_yrs: 881
#     - male_age: 866
#     - risk_weighted_age: 714
#     - female_age: 694
#   🔧 Params: {'name': 'lgbm_boost4', 'learning_rate': 0.008, 'num_leaves': 60, 'max_depth': 6, 'reg_alpha': 0.3, 'reg_lambda': 1.2, 'subsample': 0.8, 'colsample_bytree': 1.0}

# 🔹 모델: lgbm_boost1
#   - Train R² = 0.4032
#   - Test  R² = 0.3192
#   📌 중요 변수 Top 5:
#     - age: 948
#     - male_age: 766
#     - risk_weighted_age: 674
#     - edu_yrs: 662
#     - female_age: 607
#   🔧 Params: {'name': 'lgbm_boost1', 'learning_rate': 0.01, 'num_leaves': 80, 'max_depth': 7, 'reg_alpha': 0.4, 'reg_lambda': 1.5, 'subsample': 0.9, 'colsample_bytree': 0.9}

# 🔹 모델: lgbm_boost5
#   - Train R² = 0.4161
#   - Test  R² = 0.3171
#   📌 중요 변수 Top 5:
#     - age: 1496
#     - male_age: 1254
#     - risk_weighted_age: 1160
#     - edu_yrs: 1142
#     - female_age: 995
#   🔧 Params: {'name': 'lgbm_boost5', 'learning_rate': 0.007, 'num_leaves': 110, 'max_depth': 7, 'reg_alpha': 0.6, 'reg_lambda': 3.0, 'subsample': 0.85, 'colsample_bytree': 0.9}

# 🔹 모델: lgbm_boost2
#   - Train R² = 0.4227
#   - Test  R² = 0.3125
#   📌 중요 변수 Top 5:
#     - age: 2204
#     - male_age: 1717
#     - risk_weighted_age: 1661
#     - edu_yrs: 1538
#     - female_age: 1431
#   🔧 Params: {'name': 'lgbm_boost2', 'learning_rate': 0.005, 'num_leaves': 100, 'max_depth': 8, 'reg_alpha': 0.6, 'reg_lambda': 2.0, 'subsample': 0.8, 'colsample_bytree': 0.8}

# 🔹 모델: lgbm_boost3
#   - Train R² = 0.4343
#   - Test  R² = 0.3071
#   📌 중요 변수 Top 5:
#     - age: 1201
#     - risk_weighted_age: 994
#     - male_age: 950
#     - edu_yrs: 855
#     - female_age: 823
#   🔧 Params: {'name': 'lgbm_boost3', 'learning_rate': 0.01, 'num_leaves': 120, 'max_depth': 9, 'reg_alpha': 0.5, 'reg_lambda': 2.5, 'subsample': 1.0, 'colsample_bytree': 0.8}