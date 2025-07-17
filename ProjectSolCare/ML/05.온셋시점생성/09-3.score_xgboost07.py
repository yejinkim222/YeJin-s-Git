# 모델 성능이 너무 안나온다...
# 근데 정규화를 까먹었엉
# 정규화하기...
# 정규화 o: age, edu_yrs, risk_weighted_age, male_age, female_age, log_risk_weighted_age, age_group5, years_until_*
# 정규화 x: gender, edu_level, edu_is_low, is_old, high_risk_group, has_*, *_missing, years_until_ad
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import xgboost as xgb

# ✅ 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ -1을 0으로, 관련 missing 컬럼 제거
df['has_hibpe'] = df['has_hibpe'].replace(-1, 0)
if 'has_hibpe_missing' in df.columns:
    df.drop(columns=['has_hibpe_missing'], inplace=True)

# ✅ 정규화 대상 수치형 컬럼
numeric_cols = [
    "age", "edu_yrs", "risk_weighted_age",
    "male_age", "female_age", "log_risk_weighted_age",
    "age_group5", "years_until_db",
    "years_until_hibpe", "years_until_mci"
]

# ✅ 정규화 방법 정의
scalers = {
    "minmax": MinMaxScaler(),
    "zscore": StandardScaler(),
    "robust": RobustScaler()
}

results = []

# ✅ 정규화 방식별로 반복
for name, scaler in scalers.items():
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    # ✅ X, y 정의
    target_col = 'years_until_ad'
    exclude_cols = ['hhid', 'hhid_year', 'year']
    X = df_scaled.drop(columns=exclude_cols + [target_col])
    y = df_scaled[target_col]
    X = X.dropna()
    y = y.loc[X.index]

    # ✅ train/test 분할
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

    # ✅ 중요 변수
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(5)

    results.append({
        "정규화": name,
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4),
        "Top 5 Features": top_features.to_dict()
    })

# ✅ 결과 출력
for res in results:
    print(f"\n🔹 정규화: {res['정규화']}")
    print(f"  - Train R² = {res['Train R²']}")
    print(f"  - Test  R² = {res['Test R²']}")
    print("  📌 중요 변수 Top 5:")
    for k, v in res["Top 5 Features"].items():
        print(f"    - {k}: {v:.6f}")

# 결과
# 🔹 정규화: minmax
#   - Train R² = 0.8909
#   - Test  R² = 0.0425
#   📌 중요 변수 Top 5:
#     - years_until_hibpe_missing: 0.156282
#     - has_hibpe: 0.154433
#     - years_until_hibpe: 0.089754
#     - years_until_db_missing: 0.066739
#     - is_low_edu: 0.059296

# 🔹 정규화: zscore
#   - Train R² = 0.8907
#   - Test  R² = 0.0443
#   📌 중요 변수 Top 5:
#     - years_until_hibpe_missing: 0.168172
#     - has_hibpe: 0.161165
#     - years_until_hibpe: 0.089468
#     - years_until_db_missing: 0.067468
#     - is_low_edu: 0.041628

# 🔹 정규화: robust
#   - Train R² = 0.8901
#   - Test  R² = 0.0482
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.155915
#     - years_until_hibpe_missing: 0.149168
#     - years_until_hibpe: 0.094717
#     - years_until_db_missing: 0.070430
#     - is_low_edu: 0.048886