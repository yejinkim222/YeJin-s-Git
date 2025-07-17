# 로그화해서
# 스코어 확인하기
# 응 개망했쥬
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
import xgboost as xgb

# ✅ 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 조건 처리: has_hibpe_missing == 1이면 has_hibpe = -1, 그리고 컬럼 제거
df.loc[df['has_hibpe_missing'] == 1, 'has_hibpe'] = -1
df.drop(columns=['has_hibpe_missing'], inplace=True)

# ✅ 정규화 대상 컬럼
numeric_cols = [
    "age", "edu_yrs", "risk_weighted_age", "male_age", "female_age",
    "log_risk_weighted_age", "age_group5",
    "years_until_db", "years_until_hibpe", "years_until_mci"
]

# ✅ 정규화 방식들
scalers = {
    "minmax": MinMaxScaler(),
    "zscore": StandardScaler(),
    "robust": RobustScaler()
}

# ✅ 결과 저장
results = []

# ✅ 로그 타겟 정의
df['log_years_until_ad'] = np.log1p(df['years_until_ad'])

for name, scaler in scalers.items():
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    # ✅ X, y 정의
    exclude_cols = ['hhid', 'hhid_year', 'year', 'years_until_ad']
    X = df_scaled.drop(columns=exclude_cols + ['log_years_until_ad'])
    y_log = df_scaled['log_years_until_ad']
    X = X.dropna()
    y_log = y_log.loc[X.index]

    # ✅ train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    # ✅ XGBoost 모델 학습
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    # ✅ 예측 후 원래 값 복원
    y_train_pred = np.expm1(xgb_model.predict(X_train))
    y_test_pred = np.expm1(xgb_model.predict(X_test))
    train_r2 = r2_score(np.expm1(y_train), y_train_pred)
    test_r2 = r2_score(np.expm1(y_test), y_test_pred)

    importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(5)

    results.append({
        "정규화": name,
        "모델": "XGBoost",
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4),
        "Top 5 Features": top_features.to_dict()
    })

    # ✅ ElasticNet (로컬에서만 실행)
    elastic_model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
    elastic_model.fit(X_train, y_train)
    y_test_pred_en = np.expm1(elastic_model.predict(X_test))
    elastic_r2 = r2_score(np.expm1(y_test), y_test_pred_en)

    results.append({
        "정규화": name,
        "모델": "ElasticNet",
        "Train R²": round(elastic_model.score(X_train, y_train), 4),
        "Test R²": round(elastic_r2, 4),
        "Top 5 Features": dict(zip(X.columns, np.abs(elastic_model.coef_)))  # 영향력 기준
    })

# ✅ 출력
for res in results:
    print(f"\n🔹 정규화: {res['정규화']} / 모델: {res['모델']}")
    print(f"  - Train R² = {res['Train R²']}")
    print(f"  - Test  R² = {res['Test R²']}")
    print("  📌 중요 변수 Top 5:")
    for k, v in sorted(res["Top 5 Features"].items(), key=lambda item: -item[1])[:5]:
        print(f"    - {k}: {v:.6f}")

# 결과
# 🔹 정규화: minmax / 모델: XGBoost
#   - Train R² = 0.3408
#   - Test  R² = 0.1846
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.082618
#     - years_until_db_missing: 0.080038
#     - is_low_edu: 0.067366
#     - age_group5: 0.059589
#     - male_age: 0.057877

# 🔹 정규화: minmax / 모델: ElasticNet
#   - Train R² = 0.1029
#   - Test  R² = 0.1524
#   📌 중요 변수 Top 5:
#     - years_until_db_missing: 0.280325
#     - has_hibpe: 0.146698
#     - is_old: 0.089961
#     - has_db: 0.071378
#     - years_until_mci_missing: 0.023643

# 🔹 정규화: zscore / 모델: XGBoost
#   - Train R² = 0.3409
#   - Test  R² = 0.1842
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.082775
#     - years_until_db_missing: 0.078679
#     - is_low_edu: 0.066868
#     - age_group5: 0.058595
#     - male_age: 0.057791

# 🔹 정규화: zscore / 모델: ElasticNet
#   - Train R² = 0.1381
#   - Test  R² = 0.1846
#   📌 중요 변수 Top 5:
#     - years_until_db_missing: 0.179853
#     - is_old: 0.179135
#     - years_until_hibpe_missing: 0.177527
#     - has_hibpe: 0.146190
#     - age_group5: 0.056998

# 🔹 정규화: robust / 모델: XGBoost
#   - Train R² = 0.3401
#   - Test  R² = 0.1832
#   📌 중요 변수 Top 5:
#     - has_hibpe: 0.082683
#     - years_until_db_missing: 0.078995
#     - is_low_edu: 0.066809
#     - age_group5: 0.059299
#     - male_age: 0.057629

# 🔹 정규화: robust / 모델: ElasticNet
#   - Train R² = 0.1385
#   - Test  R² = 0.1837
#   📌 중요 변수 Top 5:
#     - years_until_hibpe_missing: 0.188945
#     - years_until_db_missing: 0.176090
#     - is_old: 0.167492
#     - has_hibpe: 0.139615
#     - age_group5: 0.067228