# 도움 안 되던 파생변수 없애고
# 정규화한 거 취소하기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 제거할 중요도 낮은 파생 변수
low_importance_features = [
    "is_low_edu", "is_old", "log_risk_weighted_age",
    "years_until_hibpe_missing", "years_until_db_missing", "years_until_mci_missing"
]
df = df.drop(columns=low_importance_features, errors='ignore')

# ✅ 타겟 및 피처 분리
target_col = "years_until_ad"
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

# ✅ 결측 타겟 제거
X = X.loc[y.notna()]
y = y.loc[y.notna()]

# ✅ train/test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ LightGBM 모델 정의
params = {
    'name': 'lgbm_final',
    'learning_rate': 0.01,
    'num_leaves': 60,
    'max_depth': 6,
    'reg_alpha': 0.3,
    'reg_lambda': 1,
    'random_state': 42
}
model = lgb.LGBMRegressor(
    learning_rate=params['learning_rate'],
    num_leaves=params['num_leaves'],
    max_depth=params['max_depth'],
    reg_alpha=params['reg_alpha'],
    reg_lambda=params['reg_lambda'],
    random_state=params['random_state']
)
model.fit(X_train, y_train)

# ✅ 성능 평가
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# ✅ 중요 변수 출력
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)

print("🔹 모델: LightGBM (중요 변수 제거 후)")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")
print("  📌 중요 변수 Top 5:")
for k, v in top_features.items():
    print(f"    - {k}: {v:.6f}")
print(f"  🔧 Params: {params}")

# 결과
# 🔹 모델: LightGBM (중요 변수 제거 후)
#   - Train R² = 0.2376
#   - Test  R² = 0.2489
#   📌 중요 변수 Top 5:
#     - age: 345.000000
#     - female_age: 310.000000
#     - male_age: 221.000000
#     - edu_yrs: 213.000000
#     - years_until_db: 178.000000
#   🔧 Params: {'name': 'lgbm_final', 
#               'learning_rate': 0.01, 
#               'num_leaves': 60, 
#               'max_depth': 6, 
#               'reg_alpha': 0.3, 
#               'reg_lambda': 1, 
#               'random_state': 42}