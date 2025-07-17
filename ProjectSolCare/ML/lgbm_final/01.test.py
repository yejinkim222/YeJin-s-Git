import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 1. 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/model_storage/rf_data_final2.csv")

# ✅ 2. 피처/타겟 정의
X = df.drop(columns=["years_until_ad"])
y = df["years_until_ad"]

# ✅ 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 4. LGBM 모델 학습 (베이스라인)
model = lgb.LGBMRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ✅ 5. 성능 평가
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))

# ✅ 6. 결과 출력
print(f"✅ LGBM Train R²: {train_r2:.4f}")
print(f"✅ LGBM Test  R²: {test_r2:.4f}")

# 결과
# ✅ LGBM Train R²: 0.9035
# ✅ LGBM Test  R²: 0.6536