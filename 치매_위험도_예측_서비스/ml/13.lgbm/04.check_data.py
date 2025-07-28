# 새로 저장한 데이터 이용해서
# 모델 성능 확인하는 코드
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_for_lgbm.csv"
df = pd.read_csv(file_path)

# 📌 타겟 변수
target_col = "years_until_ad"
y = df[target_col]

# 📌 모델 학습에서 제외할 컬럼
model_exclude_only = ["ad_year", "ad_year_missing", "year_missing", "years_until_ad_ratio"]

# 📌 학습용 feature 구성
non_features = ["hhid", "year", "hhid_year", target_col] + model_exclude_only
X = df.drop(columns=[col for col in non_features if col in df.columns])

# 📌 결측치 제거
X = X.dropna()
y = y.loc[X.index]

# 📌 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 📌 LGBM 학습
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# 📌 성능 평가
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))

# 📌 결과 출력
print(f"✅ Train R²: {train_r2:.4f}")
print(f"✅ Test  R²: {test_r2:.4f}")

# 결과
# ✅ Train R²: 0.9003
# ✅ Test  R²: 0.5838