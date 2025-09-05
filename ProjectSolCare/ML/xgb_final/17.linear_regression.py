import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

data_path = "C:/workspace/Project01/model_storage/xgb_data_final2.csv"
df = pd.read_csv(data_path)

# 타겟/피처 분리
y = df["years_until_ad"]
X = df.drop(columns=["years_until_ad"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# 예측
y_train_pred = lin_model.predict(X_train)
y_test_pred = lin_model.predict(X_test)

# 성능 계산
results = {
    "Linear Regression": {
        "Train R²": r2_score(y_train, y_train_pred),
        "Test R²": r2_score(y_test, y_test_pred),
        "Train MSE": mean_squared_error(y_train, y_train_pred),
        "Test MSE": mean_squared_error(y_test, y_test_pred),
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),
    }
}

# 저장된 XGBoost 모델 불러오기
xgb_model = joblib.load("C:/workspace/Project01/model_storage/xgb_best_model_final2.pkl")

# 예측
y_train_pred_xgb = xgb_model.predict(X_train)
y_test_pred_xgb = xgb_model.predict(X_test)

# 성능 계산
results["XGBoost"] = {
    "Train R²": r2_score(y_train, y_train_pred_xgb),
    "Test R²": r2_score(y_test, y_test_pred_xgb),
    "Train MSE": mean_squared_error(y_train, y_train_pred_xgb),
    "Test MSE": mean_squared_error(y_test, y_test_pred_xgb),
    "Train MAE": mean_absolute_error(y_train, y_train_pred_xgb),
    "Test MAE": mean_absolute_error(y_test, y_test_pred_xgb),
}

df_results = pd.DataFrame(results).T
print(df_results)



# ===== 변수별 영향도 (Linear Regression 계수 확인) =====
results = []

for col in X.columns:
    X_col = X[[col]].values
    y_val = y.values
    lr = LinearRegression()
    lr.fit(X_col, y_val)
    coef = lr.coef_[0]
    results.append((col, coef))

# 절댓값 기준 정렬
results.sort(key=lambda x: abs(x[1]), reverse=True)

# 출력
print("\n[Linear Regression: 변수별 기울기(계수)]")
for name, coef in results:
    sign = "양의 상관" if coef > 0 else "음의 상관"
    print(f"{name:<30}: {coef:>8.4f} ({sign})")

