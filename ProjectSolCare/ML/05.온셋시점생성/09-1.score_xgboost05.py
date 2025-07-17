# 파생변수 2개 더 만들고 다시...
# xgBoost 실험
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ X, y 분리
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]
X = X.dropna()
y = y.loc[X.index]

# ✅ train/test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ XGBoost 모델 생성 및 학습
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ✅ 평가
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n🔹 XGBoost 결과:")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")

# ✅ 중요 변수 Top 5
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)
print("\n📌 중요 변수 Top 5:")
print(top_features.to_string())

# 결과
# 🔹 XGBoost 결과:
#   - Train R² = 0.7236
#   - Test  R² = 0.2005

# 📌 중요 변수 Top 5:
# has_hibpe                    0.198145
# has_hibpe_missing            0.177405
# years_until_db_missing       0.070928
# years_until_hibpe_missing    0.068839
# age_group5                   0.044133