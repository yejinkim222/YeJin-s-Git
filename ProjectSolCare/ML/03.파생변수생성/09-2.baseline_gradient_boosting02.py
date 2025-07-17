# Gradient Boosting score, 중요 피쳐 출력
# Baseline Boosting 모델로 사용
# 80/20 split
# 하이퍼 파라미터 튜닝
# overfitting 해결중
# 근데 더 안좋아졌다...
# 그냥 linear보다 이게 더 점수 잘나온다 정도로만 활용
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

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

# 📌 학습/평가 데이터 분리 (80/20 Split)
X = df[feature_cols]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 실험할 max_depth 리스트
max_depths = [4, 5, 6]

# 📌 실험 실행
for depth in max_depths:
    print(f"\n🔷 Gradient Boosting (max_depth={depth})")
    
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=depth,
        min_samples_split=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    
    print(f"  ✅ R² Score: {score:.4f}")
    
    # 중요 변수 출력
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    print("  📌 중요 변수 Top 5:")
    print(feature_importance.head(5).to_string(index=False))

# 결과
#   ✅ R² Score: 0.1661
#   📌 중요 변수 Top 5:
#           feature  importance
#               age    0.159374
# has_hibpe_missing    0.151197
# risk_weighted_age    0.130616
#           edu_yrs    0.122991
#    years_until_db    0.114985

# 🔷 Gradient Boosting (max_depth=5)
#   ✅ R² Score: 0.1178
#   📌 중요 변수 Top 5:
#           feature  importance
#               age    0.167031
#           edu_yrs    0.152595
# has_hibpe_missing    0.137125
# risk_weighted_age    0.123166
#    years_until_db    0.105412

# 🔷 Gradient Boosting (max_depth=6)
#   ✅ R² Score: 0.0901
#   📌 중요 변수 Top 5:
#           feature  importance
#           edu_yrs    0.158751
#               age    0.152599
# has_hibpe_missing    0.126965
# risk_weighted_age    0.125251
#    years_until_db    0.097594