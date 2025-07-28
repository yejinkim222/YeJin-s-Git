# 1. 파생변수, age 필터링 둘 다 없이
# 2. 파생변수 효과만 확인
# 3. age_group5 ≥ 10 필터링 효과만 확인
# 4. 둘 다 확인
# 순서대로 실험하는 코드
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# 📌 공통 설정
target_col = 'years_until_ad'
base_features = [
    'age', 'gender', 'edu_yrs', 'has_db', 'has_hibpe', 'edu_level',
    'years_until_mci', 'years_until_db', 'years_until_hibpe',
    'years_until_hibpe_missing', 'has_hibpe_missing',
    'years_until_mci_missing', 'years_until_db_missing', 'edu_yrs_missing'
]

engineered_features = [
    'age_group5', 'risk_factor_sum', 'edu_is_low', 'risk_weighted_age',
    'male_age', 'female_age', 'log_risk_weighted_age',
    'is_low_edu', 'is_old', 'high_risk_group'
]

# 📌 실험 조합 설정
experiments = {
    'exp_1_baseline': {
        'use_features': base_features,
        'apply_filter': False
    },
    'exp_2_engineered': {
        'use_features': base_features + engineered_features,
        'apply_filter': False
    },
    'exp_3_filtered': {
        'use_features': base_features,
        'apply_filter': True
    },
    'exp_4_both': {
        'use_features': base_features + engineered_features,
        'apply_filter': True
    },
}

results = []

# 📌 실험 실행
for name, config in experiments.items():
    df_exp = df.copy()
    if config['apply_filter']:
        df_exp = df_exp[df_exp['age_group5'] >= 10]

    X = df_exp[config['use_features']]
    y = df_exp[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMRegressor(
        learning_rate=0.01,
        num_leaves=60,
        max_depth=6,
        reg_alpha=0.3,
        reg_lambda=1,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results.append({
        '실험명': name,
        'Train R²': round(train_r2, 4),
        'Test R²': round(test_r2, 4)
    })

# 📌 결과 출력
results_df = pd.DataFrame(results)
print(results_df)

# 결과
#              실험명    Train R²  Test R²
# 0    exp_1_baseline    0.2252   0.2147
# 1  exp_2_engineered    0.2398   0.2420
# 2    exp_3_filtered    0.2248   0.2343
# 3        exp_4_both    0.2463   0.2370