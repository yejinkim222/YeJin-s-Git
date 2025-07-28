# lgbm 학습 실험
# 전체 feature 이용해보기
# 로그 변환 회귀 -> 근데 스코어 나락갔다... 포기 ㅜ
# ✅ 로그 변환 기반 LightGBM 회귀
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/03.AD_train_derived.csv"
df = pd.read_csv(file_path)

# 📌 변수 정의
target = "years_until_ad"
feature_cols = [
    'age', 'gender', 'edu_yrs', 'has_db', 'AD_MCI_status', 'has_hibpe',
    'edu_level', 'years_until_mci', 'years_until_db', 'years_until_hibpe',
    'years_until_hibpe_missing', 'has_hibpe_missing', 'years_until_mci_missing',
    'years_until_db_missing', 'edu_yrs_missing','age_group5', 
    'risk_factor_sum', 'edu_is_low', 'risk_weighted_age',
    'male_age', 'female_age'
]

# 📌 스케일링
X = df[feature_cols]
y = df[target]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 🔄 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 🔁 ✅ 로그 변환
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# 🎯 LightGBM 모델 정의
model = lgb.LGBMRegressor(
    max_depth=6,
    learning_rate=0.03,
    n_estimators=1000,
    num_leaves=40,
    min_split_gain=1e-3,
    min_child_samples=5,
    random_state=42
)

# ⏳ early stopping 설정
early_stopping_cb = lgb.early_stopping(stopping_rounds=30, verbose=False)

# ✅ 모델 학습 (로그 변환된 y로)
model.fit(
    X_train, y_train_log,
    eval_set=[(X_test, y_test_log)],
    eval_metric="l2",
    callbacks=[early_stopping_cb]
)

# 📈 예측 후 다시 복원
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # log → 원래값으로 되돌림

# 🎯 성능 평가
score = r2_score(y_test, y_pred)
print(f"\n✅ LightGBM (로그 변환) R² Score: {score:.4f}")

# 📌 중요 피처 확인
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)
print("\n📌 중요 변수 Top 5:")
print(top_features)

# 결과
# ✅ LightGBM (로그 변환) R² Score: 0.1086

# 📌 중요 변수 Top 5:
# age                  308
# edu_yrs              265
# years_until_db       207
# risk_weighted_age    146
# male_age             117
# dtype: int32