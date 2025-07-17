# elasticNet alpha 튜닝해서
# xgboost 재학습 실험
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/03.AD_train_derived.csv"
df = pd.read_csv(file_path)

# 📌 변수 설정
target = "years_until_ad"
feature_cols = [
    'age', 'gender', 'edu_yrs', 'has_db', 'AD_MCI_status', 'has_hibpe',
    'edu_level', 'years_until_mci', 'years_until_db', 'years_until_hibpe',
    'years_until_hibpe_missing', 'has_hibpe_missing', 'years_until_mci_missing',
    'years_until_db_missing', 'edu_yrs_missing', 'age_group5', 
    'risk_factor_sum', 'edu_is_low', 'risk_weighted_age',
    'male_age', 'female_age'
]

# 📌 Train/Test 분리
X = df[feature_cols]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔍 alpha 후보 리스트
alpha_list = [0.005, 0.01, 0.03, 0.05, 0.07, 0.1]

results = []

for alpha in alpha_list:
    # ElasticNet으로 피처 선택
    enet = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42)
    enet.fit(X_train, y_train)
    selected_features = X_train.columns[enet.coef_ != 0].tolist()

    # XGBoost 학습
    model = XGBRegressor(
        max_depth=3,
        learning_rate=0.01,
        n_estimators=500,
        early_stopping_rounds=30,
        random_state=42
    )
    model.fit(X_train[selected_features], y_train,
              eval_set=[(X_test[selected_features], y_test)],
              verbose=False)
    
    y_pred = model.predict(X_test[selected_features])
    score = r2_score(y_test, y_pred)

    # 중요 변수 정리
    importances = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False).head(5)

    results.append({
        'alpha': alpha,
        'r2_score': round(score, 4),
        'top_features': importances
    })

# 상위 3개 결과만 출력
top_results = sorted(results, key=lambda x: x['r2_score'], reverse=True)[:3]

for i, res in enumerate(top_results, 1):
    print(f"\n🔷 {i}. alpha={res['alpha']} → R² = {res['r2_score']}")
    print("   📌 중요 변수:")
    for feat, val in res['top_features'].items():
        print(f"     - {feat}: {round(val, 4)}")

# 결과
# 🔷 1. alpha=0.05 → R² = 0.257
#    📌 중요 변수:
#      - has_hibpe_missing: 0.3889
#      - AD_MCI_status: 0.0981
#      - years_until_db: 0.0929
#      - risk_weighted_age: 0.0901
#      - years_until_hibpe: 0.0807

# 🔷 2. alpha=0.07 → R² = 0.257
#    📌 중요 변수:
#      - has_hibpe_missing: 0.3889
#      - AD_MCI_status: 0.0981
#      - years_until_db: 0.0929
#      - risk_weighted_age: 0.0901
#      - years_until_hibpe: 0.0807

# 🔷 3. alpha=0.1 → R² = 0.257
#    📌 중요 변수:
#      - has_hibpe_missing: 0.3889
#      - AD_MCI_status: 0.0981
#      - years_until_db: 0.0929
#      - risk_weighted_age: 0.0901
#      - years_until_hibpe: 0.0807