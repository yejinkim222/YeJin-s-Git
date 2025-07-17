# 찬영씨가 1년뒤를 예측하는 게 가장 예측 잘한다고 해 주셔서
# 나도 그렇게 해서 스코어 확인하기..
# 10에서의 elasticNet 통해 선택된 featureSet 사용
# LGBM 학습 실험 (후처리로 1년 이내 예측 정확도 평가 추가)

# 📌 패키지 임포트
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/10.result/final_AD_train_long.csv"
df = pd.read_csv(file_path)

# 📌 변수 정의
target = "years_until_ad"
feature_cols = [
    'gender', 'edu_yrs', 'has_db', 'AD_MCI_status', 'has_hibpe', 
    'ad_year', 'edu_level'
]

# 📌 결측치 마킹 컬럼 생성 (-1 또는 np.nan)
for col in feature_cols:
    df[f"{col}_missing"] = df[col].apply(lambda x: 1 if pd.isna(x) or x == -1 else 0)

# 📌 결측치 채우기 (모델 학습을 위해 필요)
df = df.copy()
df[feature_cols] = df[feature_cols].replace(-1, np.nan)
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

# 📌 학습용 데이터 구성
X = df[feature_cols + [f"{col}_missing" for col in feature_cols]]
y = df[target]

# 📌 스케일링
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 📌 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 📌 LightGBM 회귀 모델 학습
model = lgb.LGBMRegressor(
    max_depth=6,
    learning_rate=0.03,
    n_estimators=1000,
    num_leaves=40,
    min_split_gain=1e-3,
    min_child_samples=5,
    random_state=42
)

early_stopping_cb = lgb.early_stopping(stopping_rounds=30, verbose=False)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="l2",
    callbacks=[early_stopping_cb]
)

# 📈 결과 출력
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"\n✅ LightGBM (회귀) R² Score: {score:.4f}")

importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)
print("\n📌 중요 변수 Top 5:")
print(top_features)
