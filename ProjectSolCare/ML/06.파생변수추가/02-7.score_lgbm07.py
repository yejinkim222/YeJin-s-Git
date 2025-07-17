# has_hibpe 결측 제거하고
# 단순한 모델로 실험하기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv")

# ✅ 결측 제거: has_hibpe 결측 제거
df = df[df['has_hibpe'] != -1].copy()

# ✅ feature/target 분리
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

# ✅ train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 단순 모델 파라미터 설정
params = {
    'name': 'lgbm_simple',
    'learning_rate': 0.01,
    'num_leaves': 20,
    'max_depth': 3,
    'reg_alpha': 0.5,
    'reg_lambda': 2,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# ✅ 모델 학습
model = lgb.LGBMRegressor(
    learning_rate=params['learning_rate'],
    num_leaves=params['num_leaves'],
    max_depth=params['max_depth'],
    reg_alpha=params['reg_alpha'],
    reg_lambda=params['reg_lambda'],
    subsample=params['subsample'],
    colsample_bytree=params['colsample_bytree'],
    random_state=params['random_state'],
    n_estimators=1000
)
model.fit(X_train, y_train)

# ✅ 예측 및 평가
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(f"🔹 모델: {params['name']}")
print(f"  - Train R² = {r2_score(y_train, y_train_pred):.4f}")
print(f"  - Test  R² = {r2_score(y_test, y_test_pred):.4f}")

# ✅ 중요 변수 출력
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)
print("\n📌 중요 변수 Top 5:")
for k, v in top_features.items():
    print(f"  - {k}: {v:.6f}")

# 결과
# 🔹 모델: lgbm_simple
#   - Train R² = 0.4413
#   - Test  R² = -0.0444

# 📌 중요 변수 Top 5:
#   - age: 725.000000
#   - female_age: 511.000000
#   - edu_yrs: 483.000000
#   - risk_weighted_age: 448.000000
#   - years_until_db: 306.000000