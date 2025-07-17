# has_hibpe_missing == 1이면 has_hibpe 값을 -1로 설정
# 이후 has_hibpe_missing 컬럼은 제거
# 지정된 수치형 컬럼 정규화(MinMax, Z-Score, Robust 방식)
# XGBoost로 학습 및 평가
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import xgboost as xgb

# ✅ 원본 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 조건: has_hibpe_missing == 1이면 has_hibpe에 -1 넣기
df.loc[df['has_hibpe_missing'] == 1, 'has_hibpe'] = -1

# ✅ has_hibpe_missing 컬럼 제거
df = df.drop(columns=['has_hibpe_missing'])

# ✅ 정규화 대상 컬럼
numeric_cols = [
    "age", "edu_yrs", "risk_weighted_age", "male_age", "female_age",
    "log_risk_weighted_age", "age_group5",
    "years_until_db", "years_until_hibpe", "years_until_mci"
]

# ✅ 정규화 방법들
scalers = {
    "minmax": MinMaxScaler(),
    "zscore": StandardScaler(),
    "robust": RobustScaler()
}

# ✅ 결과 저장
results = []

for name, scaler in scalers.items():
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    # ✅ X, y 분리
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

    # ✅ XGBoost 학습
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ✅ 평가
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(5)

    results.append({
        "정규화": name,
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4),
        "Top 5 Features": top_features.to_dict()
    })

# ✅ 출력
for res in results:
    print(f"\n🔹 정규화: {res['정규화']}")
    print(f"  - Train R² = {res['Train R²']}")
    print(f"  - Test  R² = {res['Test R²']}")
    print("  📌 중요 변수 Top 5:")
    for k, v in res["Top 5 Features"].items():
        print(f"    - {k}: {v:.6f}")

# 결과
# 🔹 정규화: minmax
#   - Train R² = 0.8937
#   - Test  R² = 0.0468
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.159547
#     - years_until_hibpe_missing: 0.114760
#     - is_low_edu: 0.084204
#     - years_until_db_missing: 0.082305
#     - years_until_hibpe: 0.064973

# 🔹 정규화: zscore
#   - Train R² = 0.8943
#   - Test  R² = 0.0507
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.162148
#     - years_until_hibpe_missing: 0.115305
#     - years_until_db_missing: 0.078644
#     - years_until_hibpe: 0.069960
#     - is_low_edu: 0.066251

# 🔹 정규화: robust
#   - Train R² = 0.8949
#   - Test  R² = 0.0505
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.162098
#     - years_until_hibpe_missing: 0.114355
#     - years_until_db_missing: 0.088263
#     - years_until_hibpe: 0.069347
#     - is_low_edu: 0.056644