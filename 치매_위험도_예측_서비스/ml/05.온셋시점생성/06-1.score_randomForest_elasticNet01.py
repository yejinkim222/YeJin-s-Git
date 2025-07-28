# elasticNet 사용해서 변수 선택한 뒤
# randomForest 실험
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 1. 데이터 로드
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/06.AD_train_error_fixed.csv"
df = pd.read_csv(file_path)

# ✅ 2. X, y 분리
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]
X = X.dropna()
y = y.loc[X.index]

# ✅ 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 4. ElasticNet으로 변수 선택
enet = ElasticNet(alpha=0.1, l1_ratio=0.7, random_state=42)
enet.fit(X_train, y_train)

coef = pd.Series(enet.coef_, index=X.columns)
selected_features = coef[coef != 0].index.tolist()
print(f"\n📌 ElasticNet 선택된 변수 수: {len(selected_features)}")
print(f"선택된 변수:\n{selected_features}")

# ✅ 5. 선택된 변수만으로 RF 재학습
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_sel, y_train)

# ✅ 6. 성능 평가
train_pred = rf.predict(X_train_sel)
test_pred = rf.predict(X_test_sel)
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"\n✅ RandomForest (ElasticNet 변수만 사용) 결과:")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")

# ✅ 7. 중요 변수 출력
importances = pd.Series(rf.feature_importances_, index=selected_features)
print("\n📌 중요 변수 Top 5:")
print(importances.sort_values(ascending=False).head(5))

# 결과
# 📌 ElasticNet 선택된 변수 수: 16
# 선택된 변수:
# ['age', 'edu_yrs', 'AD_MCI_status', 'has_hibpe', 'edu_level', 'years_until_mci', 'years_until_db', 'years_until_hibpe', 'years_until_hibpe_missing', 'has_hibpe_missing', 'years_until_mci_missing', 'years_until_db_missing', 'edu_is_low', 'risk_weighted_age', 'male_age', 'female_age']

# ✅ RandomForest (ElasticNet 변수만 사용) 결과:
#   - Train R² = 0.3490
#   - Test  R² = 0.2960

# 📌 중요 변수 Top 5:
# age                  0.234419
# has_hibpe_missing    0.183981
# has_hibpe            0.150396
# female_age           0.093091
# male_age             0.062008
# dtype: float64