# xgboost
# 여러 방식으로 모델 학습 후 상위 5개 모델 출력
# 학습 방식
# 학습/검증 데이터 분리 (80/20 → train/val/test 각각 분할)
# 실험 설정
# 1. depths = [3, 4, 5, 6] -> 얕은 깊이(3)가 효과적
# 2. learning_rates = [0.1, 0.05, 0.03, 0.01] -> 작은 학습률(0.03)
# 3. n_estimators = 300
# 4. early_stopping_rounds = 30
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/03.AD_train_derived.csv"
df = pd.read_csv(file_path)

# 📌 사용 변수 정의
target = "years_until_ad"
feature_cols = [
    'age', 'gender', 'edu_yrs', 'has_db', 'AD_MCI_status', 'has_hibpe',
    'edu_level', 'years_until_mci', 'years_until_db', 'years_until_hibpe',
    'years_until_hibpe_missing', 'has_hibpe_missing', 'years_until_mci_missing',
    'years_until_db_missing', 'edu_yrs_missing','age_group5',
    'risk_factor_sum', 'edu_is_low', 'risk_weighted_age',
    'male_age', 'female_age'
]

# 📌 학습/검증 데이터 분리 (80/20 → train/val/test 각각 분할)
X = df[feature_cols]
y = df[target]
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# 📌 실험 설정
depths = [3, 4, 5, 6]
learning_rates = [0.1, 0.05, 0.03, 0.01]
n_estimators = 300
early_stopping_rounds = 30

# 📌 결과 저장용
results = []

for depth in depths:
    for lr in learning_rates:
        model = xgb.XGBRegressor(
            max_depth=depth,
            learning_rate=lr,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            random_state=42,
            verbosity=0
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        importance = model.feature_importances_
        top_features = sorted(zip(X.columns, importance), key=lambda x: x[1], reverse=True)[:5]

        results.append({
            "max_depth": depth,
            "learning_rate": lr,
            "r2_score": round(score, 4),
            "top_features": top_features
        })

# 📌 결과 정렬
results_df = pd.DataFrame(results).sort_values(by="r2_score", ascending=False).reset_index(drop=True)

# 📌 상위 5개 모델 요약 출력
print("\n🔍 XGBoost 상위 5개 실험 결과")
print(results_df[['max_depth', 'learning_rate', 'r2_score']].head(5).to_string(index=False))

# 📌 상위 5개 모델 상세 출력
print("\n📊 중요 변수 Top 5 (상위 5개 모델)")
for idx, row in results_df.head(5).iterrows():
    print(f"\n🔷 {idx+1}. max_depth={row['max_depth']}, learning_rate={row['learning_rate']} → R² = {row['r2_score']}")
    print("   📌 중요 변수:")
    for f, imp in row['top_features']:
        print(f"     - {f}: {imp:.4f}")

# 결과
# 🔍 XGBoost 상위 5개 실험 결과
#  max_depth  learning_rate  r2_score
#          3           0.03    0.2339
#          3           0.10    0.2313
#          3           0.01    0.2270
#          4           0.03    0.2265
#          4           0.10    0.2239

# 📊 중요 변수 Top 5 (상위 5개 모델)

# 🔷 1. max_depth=3, learning_rate=0.03 → R² = 0.2339
#    📌 중요 변수:
#      - has_hibpe: 0.2247
#      - has_hibpe_missing: 0.1838
#      - risk_factor_sum: 0.0797
#      - years_until_hibpe: 0.0768
#      - AD_MCI_status: 0.0745

# 🔷 2. max_depth=3, learning_rate=0.1 → R² = 0.2313
#    📌 중요 변수:
#      - has_hibpe: 0.2454
#      - has_hibpe_missing: 0.2033
#      - years_until_hibpe: 0.0777
#      - AD_MCI_status: 0.0743
#      - years_until_db: 0.0608

# 🔷 3. max_depth=3, learning_rate=0.01 → R² = 0.227
#    📌 중요 변수:
#      - has_hibpe: 0.2347
#      - has_hibpe_missing: 0.1365
#      - years_until_hibpe: 0.1325
#      - years_until_db: 0.0808
#      - risk_weighted_age: 0.0754

# 🔷 4. max_depth=4, learning_rate=0.03 → R² = 0.2265
#    📌 중요 변수:
#      - has_hibpe: 0.2579
#      - has_hibpe_missing: 0.1512
#      - years_until_hibpe: 0.0884
#      - AD_MCI_status: 0.0692
#      - years_until_db: 0.0687

# 🔷 5. max_depth=4, learning_rate=0.1 → R² = 0.2239
#    📌 중요 변수:
#      - has_hibpe: 0.2430
#      - has_hibpe_missing: 0.1951
#      - years_until_hibpe: 0.0958
#      - years_until_db: 0.0692
#      - AD_MCI_status: 0.0664