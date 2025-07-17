# 결측치에 -1이 아닌 NaN 넣고 해보기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import lightgbm as lgb

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# 📌 필요 없는 결측 마스킹 컬럼 제거
df = df.drop(columns=[
    'has_hibpe_missing', 'years_until_db_missing', 
    'years_until_hibpe_missing', 'years_until_mci_missing'
])

# 📌 정규화할 수치형 컬럼
numeric_cols = [
    "age", "edu_yrs", "risk_weighted_age", "male_age", "female_age",
    "log_risk_weighted_age", "age_group5",
    "years_until_db", "years_until_hibpe", "years_until_mci"
]

# 📌 MinMax 정규화 (원래대로 유지)
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 📌 Feature / Target 분리
target_col = "years_until_ad"
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

# 📌 LightGBM은 NaN 그대로 둠
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 📌 LightGBM 모델 설정
model = lgb.LGBMRegressor(
    n_estimators=700,
    learning_rate=0.015,
    num_leaves=70,
    max_depth=8,
    reg_alpha=0.3,
    reg_lambda=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 📌 평가
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 📌 중요 변수 추출
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)

# 📌 결과 출력
print("\n✅ LightGBM 결측치 NaN 그대로 사용한 결과")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")
print("  📌 중요 변수 Top 5:")
for k, v in top_features.items():
    print(f"    - {k}: {v:.6f}")
