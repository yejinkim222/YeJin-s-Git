# 언더피팅 해결 위해 복잡도 증가
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# ✅ 1. 데이터 로드
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 2. 데이터 전처리
# 사용할 피처
drop_cols = ['hhid', 'year', 'hhid_year', 'AD_MCI_status']  # ID 및 타겟 외 변수
target_col = 'years_until_ad'
feature_cols = [col for col in df.columns if col not in drop_cols + [target_col]]

# 결측값 마스킹된 -1 유지
X = df[feature_cols].copy()
y = df[target_col].copy()

# ✅ 3. 복잡도 증가 실험
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.007,
    num_leaves=120,
    max_depth=9,
    reg_alpha=0.3,
    reg_lambda=1.5,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train, y_train)

# ✅ 4. 성능 평가
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("🔹 모델: LightGBM (복잡도 증가)")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")

# ✅ 5. 중요 변수 출력
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\n📌 중요 변수 Top 5:")
for i, (name, score) in enumerate(importance.head(5).items(), start=1):
    print(f"  {i}. {name}: {score}")

# 결과
# 🔹 모델: LightGBM (복잡도 증가)
#   - Train R² = 0.5500
#   - Test  R² = 0.2508

# 📌 중요 변수 Top 5:
#   1. age: 3895
#   2. female_age: 3300
#   3. edu_yrs: 3186
#   4. risk_weighted_age: 3052
#   5. male_age: 2511