# LightGBM 실험
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 결측 마스킹 처리
df.loc[df['has_hibpe_missing'] == 1, 'has_hibpe'] = -1
df = df.drop(columns=['has_hibpe_missing'])

# ✅ 정규화 대상 수치형 컬럼
numeric_cols = [
    "age", "edu_yrs", "risk_weighted_age", "male_age", "female_age",
    "log_risk_weighted_age", "age_group5",
    "years_until_db", "years_until_hibpe", "years_until_mci"
]

# ✅ 정규화 (Z-score 사용)
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ✅ 입력 / 타겟 분리
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]
X = X.dropna()
y = y.loc[X.index]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ LightGBM 모델 정의 및 학습
model = lgb.LGBMRegressor(
    n_estimators=700,
    max_depth=3,
    learning_rate=0.015,
    reg_alpha=0.7,
    reg_lambda=3,
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

# ✅ 중요 변수 출력
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)

print(f"🔹 모델: LightGBM")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")
print("  📌 중요 변수 Top 5:")
for feature, score in top_features.items():
    print(f"    - {feature}: {score:.6f}")

# 결과
# 🔹 모델: LightGBM
#   - Train R² = 0.3853
#   - Test  R² = 0.2975
#   📌 중요 변수 Top 5:
#     - female_age: 551.000000
#     - age: 514.000000
#     - male_age: 471.000000
#     - edu_yrs: 464.000000
#     - risk_weighted_age: 442.000000