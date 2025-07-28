import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ✅ 1. 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/model_storage/xgb_data_final2.csv")

# ✅ 2. 입력/타겟 분리
target = "years_until_ad"
X = df.drop(columns=[target])
y = df[target]

# ✅ 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 4. 모델 정의 및 학습 (베이스라인용)
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ✅ 5. 성능 출력
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))

print("📌 Random Forest 베이스라인 성능")
print(f"Train R²: {train_r2:.4f}")
print(f"Test  R²:  {test_r2:.4f}")

# 결과
# 📌 Random Forest 베이스라인 성능
# Train R²: 0.8792
# Test  R²:  0.6859