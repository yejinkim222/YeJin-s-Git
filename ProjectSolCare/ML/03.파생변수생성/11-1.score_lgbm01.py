# 10에서의 elasticNet 통해 선택된 featureSet 사용
# LGBM 학습 실험
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔍 Step 1: Lasso로 중요 변수 선택
lasso = Lasso(alpha=0.05)
lasso.fit(X_train, y_train)
selected_features = X_train.columns[lasso.coef_ != 0].tolist()

print("✅ 선택된 피처 수:", len(selected_features))
print("✅ 선택된 피처 목록:", selected_features)

# 🎯 Step 2: LGBM 실험
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
    model = LGBMRegressor(
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=500,
        random_state=42
    )
    
    model.fit(
        X_train[selected_features], y_train,
        eval_set=[(X_test[selected_features], y_test)],
        callbacks=[
            early_stopping(stopping_rounds=30),
            log_evaluation(period=0)
        ]
    )
    
    y_pred = model.predict(X_test[selected_features])
    score = r2_score(y_test, y_pred)
    
    results.append({
        'max_depth': params['max_depth'],
        'learning_rate': params['learning_rate'],
        'r2_score': score,
        'model': model
    })

# 🔍 상위 3개 실험 결과 출력
top_results = sorted(results, key=lambda x: x['r2_score'], reverse=True)[:3]

print("\n🔍 LGBM 상위 3개 실험 결과")
for i, res in enumerate(top_results, 1):
    print(f"\n🔷 {i}. max_depth={res['max_depth']}, learning_rate={res['learning_rate']} → R² = {res['r2_score']:.4f}")
    model = res['model']
    importance = pd.Series(model.feature_importances_, index=selected_features)
    top_features = importance.sort_values(ascending=False).head(5)
    print("   📌 중요 변수:")
    for feat, val in top_features.items():
        print(f"     - {feat}: {val:.4f}")

# 결과
# 🔍 LGBM 상위 3개 실험 결과

# 🔷 1. max_depth=3, learning_rate=0.03 → R² = 0.2452
#    📌 중요 변수:
#      - age: 195.0000
#      - edu_yrs: 147.0000
#      - years_until_db: 119.0000
#      - risk_weighted_age: 92.0000
#      - male_age: 85.0000

# 🔷 2. max_depth=3, learning_rate=0.01 → R² = 0.2430
#    📌 중요 변수:
#      - age: 552.0000
#      - edu_yrs: 389.0000
#      - years_until_db: 339.0000
#      - risk_weighted_age: 260.0000
#      - male_age: 229.0000

# 🔷 3. max_depth=3, learning_rate=0.1 → R² = 0.2398
#    📌 중요 변수:
#      - age: 68.0000
#      - edu_yrs: 57.0000
#      - years_until_db: 46.0000
#      - risk_weighted_age: 42.0000
#      - male_age: 37.0000