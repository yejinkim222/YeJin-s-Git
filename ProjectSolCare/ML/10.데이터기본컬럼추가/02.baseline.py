# randomforest, xgboost, lgbm baseline score test
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings("ignore")

# 📌 데이터 로드
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(file_path)

# 📌 제외할 컬럼
exclude_cols = ['hhid', 'year', 'hhid_year', 'ad_year']
target_col = 'years_until_ad'

# 📌 입력(X), 타겟(y) 분리
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

# 📌 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 📌 모델 정의
models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42)
}

# 📌 결과 저장 및 출력
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ {name} Test R²: {r2:.4f}")

# 결과
# ✅ RandomForest Test R²: 0.6563
# ✅ XGBoost Test R²: 0.6563
# ✅ LightGBM Test R²: 0.5665