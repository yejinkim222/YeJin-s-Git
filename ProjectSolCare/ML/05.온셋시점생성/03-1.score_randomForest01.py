# randomForest 실험
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/05.AD_train_add_row.csv"
df = pd.read_csv(file_path)

# ✅ 전처리
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]
X = X.dropna()
y = y.loc[X.index]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 랜덤포레스트 모델 정의
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,       # 필요하면 제한 가능
    random_state=42,
    n_jobs=-1             # 병렬처리
)
rf.fit(X_train, y_train)

# ✅ 예측 및 성능 평가
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 RandomForest 회귀 결과")
print(f"R² score: {r2:.4f}")

# ✅ 중요 변수 출력 (Top 5)
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)
print("\n📌 중요 변수 Top 5:")
print(top_features.to_string())
