# 09-5의 오버피팅을 해결하기 위한 몸부림..
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import xgboost as xgb
import itertools

# 데이터 로드
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv")

# 조건 처리
df.loc[df['has_hibpe_missing'] == 1, 'has_hibpe'] = -1
df = df.drop(columns=['has_hibpe_missing'])

# 정규화 대상
numeric_cols = [
    "age", "edu_yrs", "risk_weighted_age", "male_age", "female_age",
    "log_risk_weighted_age", "age_group5",
    "years_until_db", "years_until_hibpe", "years_until_mci"
]

# 스케일러 정의
scalers = {
    "minmax": MinMaxScaler(),
    "zscore": StandardScaler(),
    "robust": RobustScaler()
}

# 실험 파라미터 조합
param_grid = list(itertools.product(
    [500, 1000],           # n_estimators
    [3, 4],                # max_depth
    [0.01, 0.02],          # learning_rate
    [0, 0.5],              # reg_alpha
    [1],                  # reg_lambda
    ["zscore", "robust"]  # scaler
))

results = []

for n_estimators, max_depth, learning_rate, reg_alpha, reg_lambda, scaler_name in param_grid:
    df_scaled = df.copy()
    scaler = scalers[scaler_name]
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    # X, y
    target_col = 'years_until_ad'
    exclude_cols = ['hhid', 'hhid_year', 'year']
    X = df_scaled.drop(columns=exclude_cols + [target_col])
    y = df_scaled[target_col]
    X = X.dropna()
    y = y.loc[X.index]

    # split 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 평가
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results.append({
        "scaler": scaler_name,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4)
    })

# 상위 5개 결과 출력
top5 = sorted(results, key=lambda x: x["Test R²"], reverse=True)[:5]
for res in top5:
    print(f"\n🔹 정규화: {res['scaler']}")
    print(f"  - Train R² = {res['Train R²']}")
    print(f"  - Test  R² = {res['Test R²']}")
    print(f"  - Params: n_estimators={res['n_estimators']}, max_depth={res['max_depth']}, learning_rate={res['learning_rate']}, reg_alpha={res['reg_alpha']}, reg_lambda={res['reg_lambda']}")

# 결과
# 🔹 정규화: zscore
#   - Train R² = 0.4207
#   - Test  R² = 0.2486
#   - Params: n_estimators=500, max_depth=3, learning_rate=0.01, reg_alpha=0, reg_lambda=1

# 🔹 정규화: zscore
#   - Train R² = 0.4199
#   - Test  R² = 0.248
#   - Params: n_estimators=500, max_depth=3, learning_rate=0.01, reg_alpha=0.5, reg_lambda=1

# 🔹 정규화: robust
#   - Train R² = 0.42
#   - Test  R² = 0.2474
#   - Params: n_estimators=500, max_depth=3, learning_rate=0.01, reg_alpha=0.5, reg_lambda=1

# 🔹 정규화: robust
#   - Train R² = 0.4204
#   - Test  R² = 0.2472
#   - Params: n_estimators=500, max_depth=3, learning_rate=0.01, reg_alpha=0, reg_lambda=1

# 🔹 정규화: robust
#   - Train R² = 0.5181
#   - Test  R² = 0.2444
#   - Params: n_estimators=500, max_depth=3, learning_rate=0.02, reg_alpha=0, reg_lambda=1