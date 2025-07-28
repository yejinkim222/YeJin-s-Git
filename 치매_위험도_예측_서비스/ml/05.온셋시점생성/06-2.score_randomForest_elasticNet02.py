# randomForest 실험
# 여러번 훈련시키기
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 📌 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/06.AD_train_error_fixed.csv"
df = pd.read_csv(file_path)

# 📌 risk_weighted_age 다시 계산
df['risk_weighted_age'] = df['risk_factor_sum'].replace(-1, 0) * df['age_group5']

# 📌 변수 설정
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

# 📌 결측 제거
X = X.dropna()
y = y.loc[X.index]

# 📌 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 📌 여러 번 학습 후 예측 평균
n_runs = 10
test_preds = []

for i in range(n_runs):
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42 + i,  # seed 다르게 해서 다양성 확보
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    test_preds.append(preds)

# 📌 평균 예측값 계산
avg_preds = np.mean(test_preds, axis=0)

# 📌 성능 평가
r2 = r2_score(y_test, avg_preds)
print(f"\n📊 최종 평균 R² (테스트셋): {r2:.4f}")

# 결과
# 📊 최종 평균 R² (테스트셋): 0.2981