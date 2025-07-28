import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import xgboost as xgb

# ✅ 데이터 로드
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 정규화 대상 컬럼만 지정
numeric_cols = [
    "age", "edu_yrs", "risk_weighted_age",
    "male_age", "female_age", "log_risk_weighted_age",
    "age_group5", "years_until_db", "years_until_hibpe", "years_until_mci"
]

# ✅ 스케일러 목록
scalers = {
    "minmax": MinMaxScaler(),
    "zscore": StandardScaler(),
    "robust": RobustScaler()
}

# ✅ 결과 저장 리스트
results = []

for name, scaler in scalers.items():
    df_scaled = df.copy()

    # ❗ has_hibpe_missing = 1 → has_hibpe = -1로 강제치환
    df_scaled.loc[df_scaled['has_hibpe_missing'] == 1, 'has_hibpe'] = -1
    # ❗ 이후 missing 컬럼 제거
    df_scaled = df_scaled.drop(columns=[
        "has_hibpe_missing", "years_until_hibpe_missing",
        "years_until_db_missing", "years_until_mci_missing"
    ])

    # ✅ 정규화 적용
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    # ✅ 피처와 타겟 분리
    target_col = "years_until_ad"
    exclude_cols = ["hhid", "hhid_year", "year"]
    X = df_scaled.drop(columns=exclude_cols + [target_col])
    y = df_scaled[target_col]
    X = X.dropna()
    y = y.loc[X.index]

    # ✅ 데이터 분할 (train: 70%, test: 30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ✅ XGBoost 모델 정의 및 학습
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=30,
        verbose=False
    )

    # ✅ 예측 및 성능 평가
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # ✅ 중요 변수 상위 5개
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top5 = importances.sort_values(ascending=False).head(5)

    results.append({
        "정규화": name,
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4),
        "Top 5 Features": top5.to_dict()
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
#   - Train R² = 0.5704
#   - Test  R² = 0.2373
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.235416
#     - is_old: 0.204309
#     - years_until_hibpe: 0.045205
#     - male_age: 0.041783
#     - female_age: 0.040499

# 🔹 정규화: zscore
#   - Train R² = 0.5706
#   - Test  R² = 0.231
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.246397
#     - is_old: 0.126462
#     - years_until_hibpe: 0.050446
#     - male_age: 0.044798
#     - female_age: 0.043870

# 🔹 정규화: robust
#   - Train R² = 0.5694
#   - Test  R² = 0.2363
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.260846
#     - is_old: 0.125085
#     - years_until_hibpe: 0.047957
#     - risk_weighted_age: 0.043950
#     - is_low_edu: 0.043783