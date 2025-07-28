# 정밀 하이퍼파라미터 튜닝 실험
import pandas as pd
import numpy as np
import xgboost as xgb
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/10.AD_train_xgboost_filled.csv"
df = pd.read_csv(file_path)

# 📌 제외 컬럼 및 타겟 설정
exclude_cols = ['hhid', 'year', 'hhid_year', 'AD_MCI_status', 'edu_yrs', 'edu_level', 'years_until_mci', 'gender']
target_col = 'years_until_ad'

X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 세밀 튜닝 파라미터 조합
param_grid = {
    'learning_rate': [0.02, 0.025, 0.03, 0.035],
    'max_depth': [2, 3, 4],
    'subsample': [0.75, 0.8, 0.85],
    'colsample_bytree': [0.75, 0.8, 0.85],
    'reg_alpha': [0.0, 0.2, 0.4],
    'reg_lambda': [0.8, 1.0, 1.2]
}

# 전체 조합 생성
all_params = list(product(
    param_grid['learning_rate'],
    param_grid['max_depth'],
    param_grid['subsample'],
    param_grid['colsample_bytree'],
    param_grid['reg_alpha'],
    param_grid['reg_lambda'],
))

print(f"🔍 정밀 튜닝 조합 수: {len(all_params)}")

results = []

for i, (lr, md, ss, cs, ra, rl) in enumerate(all_params):
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=lr,
        max_depth=md,
        subsample=ss,
        colsample_bytree=cs,
        reg_alpha=ra,
        reg_lambda=rl,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=30
    )

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'learning_rate': lr,
        'max_depth': md,
        'subsample': ss,
        'colsample_bytree': cs,
        'reg_alpha': ra,
        'reg_lambda': rl,
        'test_r2': r2
    })

    print(f"[{i+1:03d}/{len(all_params)}] R²: {r2:.4f} | lr={lr}, md={md}, ss={ss}, cs={cs}, ra={ra}, rl={rl}")

# 📊 결과 정리
results_df = pd.DataFrame(results)
results_df.sort_values(by='test_r2', ascending=False, inplace=True)

top_n = 10
print(f"\n✅ 정밀 튜닝 결과 상위 {top_n} 조합:")
print(results_df.head(top_n).to_string(index=False))

# 결과
# ✅ 정밀 튜닝 결과 상위 10 조합:
#  learning_rate  max_depth  subsample  colsample_bytree  reg_alpha  reg_lambda  test_r2        
#          0.030          3       0.85              0.75        0.4         0.8 0.349198        
#          0.030          3       0.85              0.85        0.2         0.8 0.348965        
#          0.030          3       0.85              0.80        0.2         0.8 0.348965        
#          0.030          3       0.85              0.75        0.0         1.2 0.347833        
#          0.035          3       0.80              0.80        0.2         1.0 0.347260        
#          0.035          3       0.80              0.85        0.2         1.0 0.347260        
#          0.030          3       0.85              0.75        0.2         0.8 0.346908        
#          0.035          3       0.80              0.85        0.4         1.0 0.346091        
#          0.035          3       0.80              0.80        0.4         1.0 0.346091        
#          0.030          3       0.85              0.75        0.4         1.2 0.345370