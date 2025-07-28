# 1. has_hibpe 제거, years_until_hibpe만 남기기
# 2. has_hibpe를 known 여부로 바꿔서 1/np.nan 처리 후 결측 행 제거
# 3. has_hibpe * is_low_edu, has_hibpe * age_group5 등 교차 피처 추가
# 4. risk_factor_sum도 정규화 대상에 포함해서 다시 비교
# 해보기
# 미친오버피팅....
# 으아아악
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# ✅ 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 정규화할 컬럼
numeric_cols = [
    "age", "edu_yrs", "risk_weighted_age", "male_age", "female_age",
    "log_risk_weighted_age", "age_group5", "years_until_db",
    "years_until_hibpe", "years_until_mci", "risk_factor_sum"
]

# ✅ 정규화
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ✅ 실험 구성
experiments = {}

# 실험 A: has_hibpe 제거
df_a = df.drop(columns=["has_hibpe"])
experiments['실험 A: has_hibpe 제거'] = df_a

# 실험 B: has_hibpe == 1만 남기고 나머지는 NaN → drop
df_b = df.copy()
df_b["has_hibpe"] = df_b["has_hibpe"].apply(lambda x: 1 if x == 1 else np.nan)
df_b = df_b.dropna(subset=["has_hibpe"])
experiments['실험 B: has_hibpe == 1만'] = df_b

# 실험 C: 교차 파생 변수 추가
df_c = df.copy()
df_c["hibpe_lowedu"] = df_c["has_hibpe"] * df_c["is_low_edu"]
df_c["hibpe_old"] = df_c["has_hibpe"] * df_c["age_group5"]
experiments['실험 C: 교차 파생 변수'] = df_c

# 실험 D: 기본 전체
experiments['실험 D: 전체 그대로'] = df.copy()

# ✅ 결과 저장 리스트
results = []

# ✅ 실험 반복
for name, data in experiments.items():
    target_col = 'years_until_ad'
    exclude_cols = ['hhid', 'hhid_year', 'year']

    X = data.drop(columns=exclude_cols + [target_col])
    y = data[target_col]
    X = X.dropna()
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(5)

    print(f"\n🔹 {name}")
    print(f"  - Train R² = {train_r2:.4f}")
    print(f"  - Test  R² = {test_r2:.4f}")
    print("  📌 중요 변수 Top 5:")
    for k, v in top_features.items():
        print(f"    - {k}: {v:.6f}")

# 결과
#  실험 A: has_hibpe 제거
#   - Train R² = 0.8937
#   - Test  R² = 0.0510
#   📌 중요 변수 Top 5:
#     - has_hibpe_missing: 0.214738
#     - years_until_hibpe_missing: 0.121130
#     - years_until_db_missing: 0.083383
#     - years_until_hibpe: 0.055566
#     - years_until_mci_missing: 0.039107

# 🔹 실험 B: has_hibpe == 1만
#   - Train R² = 0.9993
#   - Test  R² = -0.3426
#   📌 중요 변수 Top 5:
#     - years_until_hibpe_missing: 0.243109
#     - high_risk_group: 0.156548
#     - years_until_db_missing: 0.090963
#     - years_until_mci_missing: 0.086348
#     - years_until_hibpe: 0.080967

# 🔹 실험 C: 교차 파생 변수
#   - Train R² = 0.8961
#   - Test  R² = 0.0518
#   📌 중요 변수 Top 5:
#     - has_hibpe_missing: 0.155616
#     - has_hibpe: 0.120990
#     - years_until_hibpe_missing: 0.108886
#     - years_until_db_missing: 0.058182
#     - years_until_hibpe: 0.056128

# 🔹 실험 D: 전체 그대로
#   - Train R² = 0.8959
#   - Test  R² = 0.0559
#   📌 중요 변수 Top 5:
#     - has_hibpe_missing: 0.173546
#     - has_hibpe: 0.126214
#     - years_until_hibpe_missing: 0.120665
#     - years_until_db_missing: 0.072336
#     - years_until_hibpe: 0.046098