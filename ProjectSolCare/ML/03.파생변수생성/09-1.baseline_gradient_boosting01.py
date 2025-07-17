# Gradient Boosting score, 중요 피쳐 출력
# Baseline Boosting 모델로 사용
# 80/20 split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/03.AD_train_derived.csv"
df = pd.read_csv(file_path)

# 📌 사용 변수 정의
target = "years_until_ad"
feature_cols = [
    'age', 'gender', 'edu_yrs', 'has_db', 'AD_MCI_status', 'has_hibpe',
    'edu_level', 'years_until_mci', 'years_until_db', 'years_until_hibpe',
    'years_until_hibpe_missing', 'has_hibpe_missing', 'years_until_mci_missing',
    'years_until_db_missing', 'edu_yrs_missing','age_group5', 
    'risk_factor_sum', 'edu_is_low', 'risk_weighted_age',
    'male_age', 'female_age'
]

# 📌 학습/평가 데이터 분리 (80/20 split)
X = df[feature_cols]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 모델 학습 및 평가
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📌 R² 출력
r2 = r2_score(y_test, y_pred)
print(f"\n🔷 Gradient Boosting R² Score: {r2:.4f}")

# 📌 중요 피처 Top 5 출력
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n📌 중요 변수 Top 5:")
print(importance_df.head(5).to_string(index=False))

# 출력
#  Gradient Boosting R² Score: 0.2190
# 📌 중요 변수 Top 5:
#           feature  importance
# has_hibpe_missing    0.192350
#               age    0.154517
# risk_weighted_age    0.139795
#    years_until_db    0.122868
#           edu_yrs    0.099369