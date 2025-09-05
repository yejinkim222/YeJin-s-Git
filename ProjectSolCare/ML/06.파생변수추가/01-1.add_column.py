# 모델 성능이 너무 안 나와서
# 파생 변수 추가해보기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import lightgbm as lgb

# 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# 정규화 대상 수치형 컬럼
numeric_cols = [
    "age",
    "edu_yrs",
    "risk_weighted_age",
    "male_age",
    "female_age",
    "log_risk_weighted_age",
    "age_group5",
    "years_until_db",
    "years_until_hibpe",
    "years_until_mci",
]

# 정규화 방식 정의
scalers = {
    "minmax": MinMaxScaler(),
    "zscore": StandardScaler(),
    "robust": RobustScaler(),
}

# LightGBM 하이퍼파라미터 실험 세트
param_grid = [
    {
        "name": "lgbm1",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1,
    },
    {
        "name": "lgbm2",
        "learning_rate": 0.01,
        "num_leaves": 40,
        "max_depth": 6,
        "reg_alpha": 0.3,
        "reg_lambda": 1,
    },
    {
        "name": "lgbm3",
        "learning_rate": 0.01,
        "num_leaves": 50,
        "max_depth": 6,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
    },
    {
        "name": "lgbm4",
        "learning_rate": 0.01,
        "num_leaves": 70,
        "max_depth": 8,
        "reg_alpha": 0.3,
        "reg_lambda": 2,
    },
    {
        "name": "lgbm5",
        "learning_rate": 0.01,
        "num_leaves": 90,
        "max_depth": 9,
        "reg_alpha": 0.5,
        "reg_lambda": 2,
    },
]

results = []

# 정규화 방식 루프
for scaler_name, scaler in scalers.items():
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    # X, y 분리
    target_col = "years_until_ad"
    exclude_cols = ["hhid", "hhid_year", "year"]
    X = df_scaled.drop(columns=exclude_cols + [target_col])
    y = df_scaled[target_col]

    # y에서 NaN 제거한 인덱스 기준으로 X, y 필터링
    non_null_index = y.dropna().index
    X = X.loc[non_null_index]
    y = y.loc[non_null_index]

    # Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for params in param_grid:
        model = lgb.LGBMRegressor(
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importances.sort_values(ascending=False).head(5)

        results.append(
            {
                "정규화": scaler_name,
                "모델": params["name"],
                "Train R²": round(train_r2, 4),
                "Test R²": round(test_r2, 4),
                "Top 5 Features": top_features.to_dict(),
                "Params": params,
            }
        )

# 상위 5개 결과만 출력
top_results = sorted(results, key=lambda x: x["Test R²"], reverse=True)[:5]

print("\n상위 5개 LightGBM 결과\n")
for res in top_results:
    print(f"모델: {res['모델']}")
    print(f"  - Train R² = {res['Train R²']}")
    print(f"  - Test  R² = {res['Test R²']}")
    print("  중요 변수 Top 5:")
    for k, v in res["Top 5 Features"].items():
        print(f"    - {k}: {v:.6f}")
    print(f"  Params: {res['Params']}")
    print("-" * 50)

# 결과
# 모델: lgbm1
#   - Train R² = 0.2295
#   - Test  R² = 0.2553
#   중요 변수 Top 5:
#     - age: 303.000000
#     - female_age: 242.000000
#     - edu_yrs: 157.000000
#     - male_age: 125.000000
#     - years_until_db: 124.000000
#   Params: {'name': 'lgbm1', 'learning_rate': 0.01, 'num_leaves': 31, 'max_depth': 5, 'reg_alpha':
# 0.1, 'reg_lambda': 1}
# --------------------------------------------------
# 모델: lgbm1
#   - Train R² = 0.2295
#   - Test  R² = 0.2553
#   중요 변수 Top 5:
#     - age: 303.000000
#     - female_age: 242.000000
#     - edu_yrs: 157.000000
#     - male_age: 125.000000
#     - years_until_db: 124.000000
#   Params: {'name': 'lgbm1', 'learning_rate': 0.01, 'num_leaves': 31, 'max_depth': 5, 'reg_alpha':
# 0.1, 'reg_lambda': 1}
# --------------------------------------------------
# 모델: lgbm1
#   - Train R² = 0.2295
#   - Test  R² = 0.2553
#   중요 변수 Top 5:
#     - age: 303.000000
#     - female_age: 242.000000
#     - edu_yrs: 157.000000
#     - male_age: 125.000000
#     - years_until_db: 124.000000
#   Params: {'name': 'lgbm1', 'learning_rate': 0.01, 'num_leaves': 31, 'max_depth': 5, 'reg_alpha':
# 0.1, 'reg_lambda': 1}
# --------------------------------------------------
# 모델: lgbm2
#   - Train R² = 0.2398
#   - Test  R² = 0.242
#   중요 변수 Top 5:
#     - age: 331.000000
#     - female_age: 314.000000
#     - edu_yrs: 232.000000
#     - male_age: 184.000000
#     - risk_weighted_age: 119.000000
#   Params: {'name': 'lgbm2', 'learning_rate': 0.01, 'num_leaves': 40, 'max_depth': 6, 'reg_alpha':
# 0.3, 'reg_lambda': 1}
# --------------------------------------------------
# 모델: lgbm2
#   - Train R² = 0.2398
#   - Test  R² = 0.242
#   중요 변수 Top 5:
#     - age: 331.000000
#     - female_age: 314.000000
#     - edu_yrs: 232.000000
#     - male_age: 184.000000
#     - risk_weighted_age: 119.000000
#   Params: {'name': 'lgbm2', 'learning_rate': 0.01, 'num_leaves': 40, 'max_depth': 6, 'reg_alpha':
# 0.3, 'reg_lambda': 1}
