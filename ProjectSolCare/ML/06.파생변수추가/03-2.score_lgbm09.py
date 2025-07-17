# age_group5 10 이상
# 복잡한 모델
# 얼리스타핑
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 조건: age_group5가 10 이상인 행만 선택
df = df[df["age_group5"] >= 10].copy()

# ✅ 예측 대상 및 feature 분리
target_col = "years_until_ad"
drop_cols = ['hhid', 'hhid_year', 'year', target_col]

X = df.drop(columns=drop_cols)
y = df[target_col]

# ✅ 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 모델 정의 (복잡하게 설정)
model = lgb.LGBMRegressor(
    learning_rate=0.02,
    num_leaves=90,
    max_depth=8,
    reg_alpha=0.4,
    reg_lambda=2,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=1000,
    random_state=42
)

# ✅ 학습 with early stopping (콜백 방식)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

# ✅ 예측 및 평가
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# ✅ 중요 변수 상위 5개
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)

# ✅ 출력
print(f"🔹 모델: LightGBM (age_group5 ≥ 10 + 복잡 + early stopping)")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")
print("\n📌 중요 변수 Top 5:")
for col, score in top_features.items():
    print(f"  - {col}: {score:.6f}")

# 결과
# 🔹 모델: LightGBM (age_group5 ≥ 10 + 복잡 + early stopping)
#   - Train R² = 0.4194
#   - Test  R² = 0.3101

# 📌 중요 변수 Top 5:
#   - age: 503.000000
#   - risk_weighted_age: 421.000000
#   - edu_yrs: 411.000000
#   - male_age: 398.000000
#   - female_age: 335.000000