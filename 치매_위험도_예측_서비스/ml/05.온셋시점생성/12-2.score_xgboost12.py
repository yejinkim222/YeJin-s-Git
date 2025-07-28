# xgboost 하이퍼파라미터 좀 더 미세하게 튜닝하기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import xgboost as xgb

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 사전 처리: 결측 마스킹 → has_hibpe에 -1 할당 후 해당 컬럼 제거
df.loc[df['has_hibpe_missing'] == 1, 'has_hibpe'] = -1
df = df.drop(columns=['has_hibpe_missing'])

# ✅ 정규화 대상 컬럼
numeric_cols = [
    "age", "edu_yrs", "risk_weighted_age", "male_age", "female_age",
    "log_risk_weighted_age", "age_group5",
    "years_until_db", "years_until_hibpe", "years_until_mci"
]

# ✅ 스케일러 정의
scalers = {
    "minmax": MinMaxScaler(),
    "zscore": StandardScaler(),
    "robust": RobustScaler()
}

# ✅ 하이퍼파라미터 실험 조합 리스트 (20개 중 일부)
param_grid = [
    {"name": f"Tuned{i+1}", "n_estimators": 700, "max_depth": 3, "learning_rate": 0.015,
     "reg_alpha": alpha, "reg_lambda": lam}
    for i, (alpha, lam) in enumerate([
        (0.3, 1), (0.3, 1.5), (0.3, 2),
        (0.5, 1), (0.5, 1.5), (0.5, 2),
        (0.7, 1), (0.7, 1.5), (0.7, 2),
        (0.1, 1), (0.1, 1.5), (0.1, 2),
        (0.3, 0.5), (0.3, 2.5), (0.3, 3),
        (0.7, 0.5), (0.7, 2.5), (0.7, 3),
        (0.5, 0.5), (0.5, 2.5)
    ])
]

# ✅ 결과 저장
results = []

# ✅ 정규화 방식별 루프
for scaler_name, scaler in scalers.items():
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    # ✅ feature/target 분리
    target_col = 'years_until_ad'
    exclude_cols = ['hhid', 'hhid_year', 'year']
    X = df_scaled.drop(columns=exclude_cols + [target_col])
    y = df_scaled[target_col]
    X = X.dropna()
    y = y.loc[X.index]

    # ✅ train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for params in param_grid:
        model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # ✅ 예측 및 평가
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # ✅ 중요 피처
        importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importances.sort_values(ascending=False).head(5)

        results.append({
            "정규화": scaler_name,
            "모델": params['name'],
            "Train R²": round(train_r2, 4),
            "Test R²": round(test_r2, 4),
            "Top 5 Features": top_features.to_dict(),
            "Params": params
        })

# ✅ 상위 5개만 Test R² 기준으로 정렬
top_results = sorted(results, key=lambda x: x['Test R²'], reverse=True)[:5]

# ✅ 출력
print("\n✅ 상위 5개 실험 결과 (Test R² 기준)\n")
for res in top_results:
    print(f"🔹 정규화: {res['정규화']} / 모델: {res['모델']}")
    print(f"  - Train R² = {res['Train R²']}")
    print(f"  - Test  R² = {res['Test R²']}")
    print("  📌 중요 변수 Top 5:")
    for k, v in res["Top 5 Features"].items():
        print(f"    - {k}: {v:.6f}")
    print(f"  🔧 Params: {res['Params']}")
    print("-" * 50)

# 결과
# ✅ 상위 5개 실험 결과 (Test R² 기준)

# 🔹 정규화: zscore / 모델: Tuned18
#   - Train R² = 0.4824
#   - Test  R² = 0.2867
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.178363
#     - years_until_db_missing: 0.085278
#     - years_until_hibpe_missing: 0.067128
#     - age_group5: 0.053584
#     - age: 0.050026
#   🔧 Params: {'name': 'Tuned18', 'n_estimators': 700, 'max_depth': 3, 'learning_rate': 0.015, 'reg_alpha': 0.7, 'reg_lambda': 3}
# --------------------------------------------------
# 🔹 정규화: robust / 모델: Tuned18
#   - Train R² = 0.4826
#   - Test  R² = 0.286
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.177256
#     - years_until_db_missing: 0.084716
#     - years_until_hibpe_missing: 0.066649
#     - age_group5: 0.053392
#     - age: 0.050038
#   🔧 Params: {'name': 'Tuned18', 'n_estimators': 700, 'max_depth': 3, 'learning_rate': 0.015, 'reg_alpha': 0.7, 'reg_lambda': 3}
# --------------------------------------------------
# 🔹 정규화: minmax / 모델: Tuned18
#   - Train R² = 0.4827
#   - Test  R² = 0.2853
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.159780
#     - is_old: 0.099541
#     - years_until_db_missing: 0.076084
#     - years_until_hibpe_missing: 0.059891
#     - age_group5: 0.048666
#   🔧 Params: {'name': 'Tuned18', 
#               'n_estimators': 700, 
#               'max_depth': 3, 
#               'learning_rate': 0.015, 
#               'reg_alpha': 0.7, 
#               'reg_lambda': 3}
# --------------------------------------------------
# 🔹 정규화: minmax / 모델: Tuned12
#   - Train R² = 0.4916
#   - Test  R² = 0.2852
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.185416
#     - years_until_db_missing: 0.082299
#     - years_until_hibpe_missing: 0.063876
#     - age_group5: 0.051096
#     - years_until_hibpe: 0.051032
#   🔧 Params: {'name': 'Tuned12', 
#               'n_estimators': 700, 
#               'max_depth': 3, 
#               'learning_rate': 0.015, 
#               'reg_alpha': 0.1, 
#               'reg_lambda': 2}
# --------------------------------------------------
# 🔹 정규화: robust / 모델: Tuned12
#   - Train R² = 0.4919
#   - Test  R² = 0.2851
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.185709
#     - years_until_db_missing: 0.080937
#     - years_until_hibpe_missing: 0.063724
#     - age_group5: 0.052528
#     - years_until_hibpe: 0.050881
#   🔧 Params: {'name': 'Tuned12', 
#               'n_estimators': 700, 
#               'max_depth': 3, 
#               'learning_rate': 0.015, 
#               'reg_alpha': 0.1, 
#               'reg_lambda': 2}