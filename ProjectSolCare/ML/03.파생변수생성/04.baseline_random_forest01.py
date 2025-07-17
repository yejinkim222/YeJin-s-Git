# 파생 변수 만들기 전 베이스라인 스코어 확인하기
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# ✅ 데이터 불러오기 (예진님이 직접 지정)
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/02.AD_train_check_missing.csv"
df = pd.read_csv(file_path)

# ✅ 입력 피처 & 타겟
feature_cols = [
    'age', 'gender', 'edu_level', 'years_until_mci', 'years_until_db', 'years_until_hibpe',
    'years_until_hibpe_missing', 'has_hibpe_missing', 'years_until_mci_missing',
    'years_until_db_missing', 'edu_yrs_missing', 'edu_level_missing', 'age_group5'
]
X = df[feature_cols]
y = df['years_until_ad']

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ✅ 결과 저장용
results = []

# ✅ depth별 모델 학습 및 R² 계산
for depth in [3, 5, 10]:
    model = RandomForestRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    results.append({'max_depth': depth, 'r2_score': r2})

# ✅ 결과 출력
results_df = pd.DataFrame(results)
print("✅ RandomForest 회귀 성능 (R²):")
print(results_df)

# 결과
# ✅ RandomForest 회귀 성능 (R²):
#    max_depth  r2_score
# 0          3  0.246945
# 1          5  0.303288
# 2         10  0.256490