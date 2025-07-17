# has_hibpe 결측 없는것만 써서 해보기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ has_hibpe 결측 (-1 마스킹된 값) 제거
df_filtered = df[df['has_hibpe'] != -1].copy()

# ✅ 타겟 및 피처 정의
target_col = 'years_until_ad'
drop_cols = ['hhid', 'hhid_year', 'year']
X = df_filtered.drop(columns=drop_cols + [target_col])
y = df_filtered[target_col]

# ✅ Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ LightGBM 모델 정의 (하이퍼파라미터는 이전 실험 기준에서 선택)
model = lgb.LGBMRegressor(
    learning_rate=0.015,
    num_leaves=80,
    max_depth=6,
    reg_alpha=0.3,
    reg_lambda=1,
    n_estimators=1000,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

# ✅ 학습
model.fit(X_train, y_train)

# ✅ 예측 및 평가
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# ✅ 결과 출력
print(f"\n🔹 모델: LightGBM (has_hibpe 결측 제거)")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")

# ✅ 중요 변수 출력
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)

print("\n📌 중요 변수 Top 5:")
for feature, importance in top_features.items():
    print(f"  - {feature}: {importance:.6f}")

# 결과
# 🔹 모델: LightGBM (has_hibpe 결측 제거)
#   - Train R² = 0.5338
#   - Test  R² = -0.0684

# 📌 중요 변수 Top 5:
#   - age: 858.000000
#   - edu_yrs: 619.000000
#   - risk_weighted_age: 576.000000
#   - female_age: 485.000000
#   - male_age: 340.000000