# 모델 복잡도 다시 살짝 올리기
# 얼리스탑핑은 아직 안넣고 하기로함
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb

# ✅ 데이터 로드
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 기본 전처리 (필요 시 조정)
target_col = "years_until_ad"
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

# ✅ 학습/검증 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 실험할 하이퍼파라미터 목록
param_grid = [
    {"name": "lgbm_plus1", "learning_rate": 0.015, "num_leaves": 70, "max_depth": 6, "reg_alpha": 0.4, "reg_lambda": 2, "subsample": 0.9, "colsample_bytree": 0.9},
    {"name": "lgbm_plus2", "learning_rate": 0.02, "num_leaves": 80, "max_depth": 6, "reg_alpha": 0.3, "reg_lambda": 1.8, "subsample": 0.8, "colsample_bytree": 0.8},
    {"name": "lgbm_plus3", "learning_rate": 0.015, "num_leaves": 90, "max_depth": 7, "reg_alpha": 0.3, "reg_lambda": 1.5, "subsample": 0.7, "colsample_bytree": 0.8},
    {"name": "lgbm_plus4", "learning_rate": 0.02, "num_leaves": 100, "max_depth": 8, "reg_alpha": 0.2, "reg_lambda": 1, "subsample": 1.0, "colsample_bytree": 0.9}
]

# ✅ 결과 저장
results = []

for params in param_grid:
    model = lgb.LGBMRegressor(
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        n_estimators=1000,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top5 = importances.sort_values(ascending=False).head(5)

    print(f"\n🔹 모델: {params['name']}")
    print(f"  - Train R² = {train_r2:.4f}")
    print(f"  - Test  R² = {test_r2:.4f}")
    print(f"  📌 중요 변수 Top 5:")
    for feat, val in top5.items():
        print(f"    - {feat}: {val:.6f}")
    print(f"  🔧 Params: {params}")
    print("-" * 50)

# 결과
# 🔹 모델: lgbm_plus4
#   - Train R² = 0.6909
#   - Test  R² = 0.2006
#   📌 중요 변수 Top 5:
#     - age: 3249.000000
#     - edu_yrs: 3223.000000
#     - risk_weighted_age: 2891.000000
#     - female_age: 2890.000000
#     - male_age: 2039.000000
#   🔧 Params: {'name': 'lgbm_plus4', 'learning_rate': 0.02, 'num_leaves': 100, 'max_depth': 8, 'reg_alpha': 0.2, 'reg_lambda': 1, 'subsample': 1.0, 'colsample_bytree': 0.9}