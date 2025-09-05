# 파생변수 만들어보고
# 성능 실험해보기
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 원본 파일 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/10.AD_train_xgboost_filled.csv"
df = pd.read_csv(file_path)

# 고정 제외 컬럼
exclude_cols = [
    "hhid",
    "year",
    "hhid_year",
    "AD_MCI_status",
    "edu_yrs",
    "edu_level",
    "years_until_mci",
    "gender",
]
target_col = "years_until_ad"

# 위험도 변수 생성
df["risk_factor_sum"] = df[["has_db", "has_hibpe", "AD_MCI_status"]].sum(axis=1)

# 실험용 파생변수
derived_vars = {
    "edu_x_risk": lambda d: d["edu_yrs"] * d["risk_factor_sum"],
    "risk_to_age_ratio": lambda d: d["risk_factor_sum"] / (d["age"] + 1),
}

results = []

for var_name, func in derived_vars.items():
    df_copy = df.copy()

    # 파생변수 생성 후 NaN/inf 정리
    try:
        df_copy[var_name] = func(df_copy)
        df_copy[var_name] = df_copy[var_name].replace([np.inf, -np.inf], np.nan).fillna(-1)
    except Exception as e:
        print(f"{var_name} 생성 중 오류 발생: {e}")
        continue

    # 학습용 데이터 분리
    X = df_copy.drop(columns=exclude_cols + [target_col])
    y = df_copy[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=30
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    results.append((var_name, r2, mse))
    print(f"추가한 변수: {var_name:<20} | Test R²: {r2:.4f} | Test MSE: {mse:.4f}")

# 결과 출력
results_df = pd.DataFrame(results, columns=['added_variable', 'test_r2', 'test_mse'])
results_df.sort_values(by='test_r2', ascending=False, inplace=True)

print("\nTest R² / MSE 향상 실험 결과:")
print(results_df.to_string(index=False))

# 결과
# 추가한 변수: edu_x_risk           | Test R²: 0.3290 | Test MSE: 16.2388
# 추가한 변수: risk_to_age_ratio    | Test R²: 0.3158 | Test MSE: 16.5587

# Test R² / MSE 향상 실험 결과:       
#    added_variable  test_r2  test_mse
#        edu_x_risk 0.328978 16.238817
# risk_to_age_ratio 0.315759 16.558697