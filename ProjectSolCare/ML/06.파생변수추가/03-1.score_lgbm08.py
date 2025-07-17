# age_group5 10이상만 필터링해서 학습
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 조건: age_group5가 10 이상인 행만 선택
df = df[df["age_group5"] >= 10].copy()

# ✅ 예측 대상 변수
target_col = "years_until_ad"

# ✅ 제거할 컬럼 (타겟, 식별자 등)
drop_cols = ['hhid', 'hhid_year', 'year', target_col]

# ✅ 피처 / 타겟 분리
X = df.drop(columns=drop_cols)
y = df[target_col]

# ✅ 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 모델 정의 및 학습
model = lgb.LGBMRegressor(
    learning_rate=0.02,
    num_leaves=60,
    max_depth=6,
    reg_alpha=0.3,
    reg_lambda=1.5,
    random_state=42
)
model.fit(X_train, y_train)

# ✅ 예측 및 평가
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# ✅ 중요 변수 확인
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)

# ✅ 출력
print(f"🔹 모델: LightGBM (age_group5 ≥ 10)")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")
print("\n📌 중요 변수 Top 5:")
for col, score in top_features.items():
    print(f"  - {col}: {score:.6f}")

# 결과
# 🔹 모델: LightGBM (age_group5 ≥ 10)
#   - Train R² = 0.3262
#   - Test  R² = 0.3004

# 📌 중요 변수 Top 5:
#   - age: 275.000000
#   - male_age: 220.000000
#   - risk_weighted_age: 159.000000
#   - edu_yrs: 145.000000
#   - female_age: 136.000000