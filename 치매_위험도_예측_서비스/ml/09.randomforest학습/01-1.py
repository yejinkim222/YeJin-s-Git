# 08에서 쓰던 데이터로
# randomforest 학습 베이스라인
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/10.AD_train_xgboost_filled.csv"
df = pd.read_csv(file_path)

# 📌 학습 제외 컬럼 정의
exclude_cols = [
    'hhid', 'year', 'hhid_year', 'AD_MCI_status',
    'edu_yrs', 'edu_level', 'years_until_mci', 'gender'
]
target_col = 'years_until_ad'

# 📌 Feature/Target 분리
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

# 📌 Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 📌 Random Forest 기본 모델
rf_model = RandomForestRegressor(
    n_estimators=100,        # 기본값
    random_state=42,
    n_jobs=-1
)

# 📌 학습
rf_model.fit(X_train, y_train)

# 📌 예측 및 평가
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"✅ RandomForest 베이스라인 Test R²: {r2:.4f}")

# 결과
# ✅ RandomForest 베이스라인 Test R²: 0.2669