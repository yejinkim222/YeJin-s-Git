# has_hibpe 영향력이 커서
# 1. 결측을 0으로 만들고
# 2. 관련 결측 마킹 변수 빼고,
# 3. 둘 다 같이 한 거랑
# 4. 원본 그대로 학습한 거
# 비교해보기
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 타겟 및 제외 컬럼 설정
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
y = df[target_col]

# ✅ 실험 1: 원본 그대로
X1 = df.drop(columns=exclude_cols + [target_col])

# ✅ 실험 2: has_hibpe의 -1을 0으로 변경
X2 = X1.copy()
X2['has_hibpe'] = X2['has_hibpe'].replace(-1, 0)

# ✅ 실험 3: missing 컬럼 제거
X3 = X1.drop(columns=['has_hibpe_missing', 'years_until_hibpe_missing'])

# ✅ 실험 4: 실험 2 + 실험 3 조합
X4 = X2.drop(columns=['has_hibpe_missing', 'years_until_hibpe_missing'])

X_sets = [X1, X2, X3, X4]
results = []

# ✅ 실험 루프
for i, X in enumerate(X_sets, 1):
    X = X.dropna()
    y_valid = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_valid, test_size=0.2, random_state=42
    )

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

    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))

    feature_imp = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_imp.sort_values(ascending=False).head(5)

    print(f"\n🔹 실험 {i}:")
    print(f"  - Train R² = {train_r2:.4f}")
    print(f"  - Test  R² = {test_r2:.4f}")
    print("  📌 중요 변수 Top 5:")
    print(top_features.to_string())
