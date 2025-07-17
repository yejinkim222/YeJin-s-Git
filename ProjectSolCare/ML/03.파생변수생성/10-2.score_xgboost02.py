# lasso로 중요 feature 남기고 xgboost 실험
# 실험 조건은 아까와 동일
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/03.AD_train_derived.csv"
df = pd.read_csv(file_path)

# 📌 기본 변수 정의
target = "years_until_ad"
feature_cols = [
    'age', 'gender', 'edu_yrs', 'has_db', 'AD_MCI_status', 'has_hibpe',
    'edu_level', 'years_until_mci', 'years_until_db', 'years_until_hibpe',
    'years_until_hibpe_missing', 'has_hibpe_missing', 'years_until_mci_missing',
    'years_until_db_missing', 'edu_yrs_missing','age_group5', 
    'risk_factor_sum', 'edu_is_low', 'risk_weighted_age',
    'male_age', 'female_age'
]

# 📌 Train/Test 분리
X = df[feature_cols]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔍 Step 1: Lasso로 중요 변수 선택
lasso = Lasso(alpha=0.05)
lasso.fit(X_train, y_train)
selected_features = X_train.columns[lasso.coef_ != 0].tolist()

print("✅ 선택된 피처 수:", len(selected_features))
print("✅ 선택된 피처 목록:", selected_features)

# 🎯 Step 2: 선택된 피처 기반으로 XGBoost 재학습 + 튜닝
params_grid = [
    {'max_depth': 3, 'learning_rate': 0.1},
    {'max_depth': 3, 'learning_rate': 0.03},
    {'max_depth': 3, 'learning_rate': 0.01},
    {'max_depth': 4, 'learning_rate': 0.1},
    {'max_depth': 4, 'learning_rate': 0.03},
    {'max_depth': 4, 'learning_rate': 0.01},
]

results = []

for params in params_grid:
    model = XGBRegressor(
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=500,
        early_stopping_rounds=30,  # ✅ 여기로 옮김
        random_state=42
    )
    
    model.fit(
        X_train[selected_features], y_train,
        eval_set=[(X_test[selected_features], y_test)],
        verbose=False
    )
    
    y_pred = model.predict(X_test[selected_features])
    score = r2_score(y_test, y_pred)
    
    results.append({
        'max_depth': params['max_depth'],
        'learning_rate': params['learning_rate'],
        'r2_score': score,
        'model': model
    })

# 🔍 상위 5개 실험 결과 출력
top_results = sorted(results, key=lambda x: x['r2_score'], reverse=True)[:5]

print("\n🔍 XGBoost (Lasso 기반 피처 선택 후 재학습) 상위 5개 실험 결과")
for i, res in enumerate(top_results, 1):
    print(f"\n🔷 {i}. max_depth={res['max_depth']}, learning_rate={res['learning_rate']} → R² = {res['r2_score']:.4f}")
    model = res['model']
    importance = pd.Series(model.feature_importances_, index=selected_features)
    top_features = importance.sort_values(ascending=False).head(5)
    print("   📌 중요 변수:")
    for feat, val in top_features.items():
        print(f"     - {feat}: {val:.4f}")

# 결과
# ✅ 선택된 피처 수: 13
# ✅ 선택된 피처 목록: ['age', 'edu_yrs', 'has_hibpe', 'edu_level', 'years_until_mci',
#  'years_until_db', 'years_until_hibpe', 'years_until_hibpe_missing', 'has_hibpe_missing', 'years_until_mci_missing', 'years_until_db_missing', 'risk_weighted_age', 'male_age']

# 🔍 XGBoost (Lasso 기반 피처 선택 후 재학습) 상위 5개 실험 결과

# 🔷 1. max_depth=4, learning_rate=0.1 → R² = 0.2404
#    📌 중요 변수:
#      - has_hibpe_missing: 0.5982
#      - years_until_hibpe: 0.0787
#      - years_until_db: 0.0708
#      - risk_weighted_age: 0.0626
#      - male_age: 0.0533

# 🔷 2. max_depth=3, learning_rate=0.01 → R² = 0.2402
#    📌 중요 변수:
#      - has_hibpe_missing: 0.4437
#      - years_until_db: 0.0989
#      - risk_weighted_age: 0.0913
#      - years_until_hibpe: 0.0902
#      - male_age: 0.0744

# 🔷 3. max_depth=3, learning_rate=0.1 → R² = 0.2385
#    📌 중요 변수:
#      - has_hibpe_missing: 0.5353
#      - years_until_hibpe: 0.0823
#      - years_until_db: 0.0791
#      - risk_weighted_age: 0.0717
#      - age: 0.0586

# 🔷 4. max_depth=3, learning_rate=0.03 → R² = 0.2384
#    📌 중요 변수:
#      - has_hibpe_missing: 0.4742
#      - years_until_db: 0.0980
#      - risk_weighted_age: 0.0855
#      - years_until_hibpe: 0.0837
#      - male_age: 0.0702

# 🔷 5. max_depth=4, learning_rate=0.01 → R² = 0.2319
#    📌 중요 변수:
#      - has_hibpe_missing: 0.5390
#      - years_until_db: 0.0846
#      - years_until_hibpe: 0.0791
#      - risk_weighted_age: 0.0715
#      - male_age: 0.0592