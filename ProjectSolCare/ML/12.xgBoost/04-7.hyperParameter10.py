# 하이퍼 파라미터 조정해보기
# ✅ XGBoost GridSearchCV 실험 (정밀 하이퍼파라미터 탐색)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")

# ✅ 파생변수 생성
df["edu_level_bucket"] = pd.cut(df["edu_level"], bins=[-1, 1, 2, 3], labels=[0, 1, 2]).astype("Int64")
df["age_x_edu"] = df["age"] * df["edu_level"]
df["edu_level_squared"] = df["edu_level"] ** 2

# ✅ 사용할 변수 리스트 정의 (데이터에 실제로 존재하는 것만 반영)
candidate_features = [
    'age', 'gender', 'edu_level', 'has_hibpe_missing', 'edu_yrs_missing',
    'age_group5', 'risk_factor_sum', 'edu_is_low', 'risk_weighted_age',
    'male_age', 'female_age', 'log_risk_weighted_age', 'is_low_edu',
    'is_old', 'high_risk_group', 'edu_level_bucket',
    'age_x_edu', 'edu_level_squared'
]
feature_cols = [col for col in candidate_features if col in df.columns]
target_col = 'years_until_ad'

# ✅ 결측 제거
df_model = df[feature_cols + [target_col]].dropna()
X = df_model[feature_cols]
y = df_model[target_col]

# ✅ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 하이퍼파라미터 탐색 범위 (정밀 조정)
param_grid = {
    "n_estimators": [300, 400, 500, 600],
    "max_depth": [6, 7, 8],
    "learning_rate": [0.03, 0.05, 0.07],
    "subsample": [0.9, 1.0],
    "colsample_bytree": [0.7, 0.8],
    "reg_alpha": [0.05, 0.1, 0.2],
    "reg_lambda": [1.0, 1.5],
    "min_child_weight": [5, 7],
    "gamma": [0.5, 1],
    "max_delta_step": [0]
}

# ✅ XGBoost 모델 정의
xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    verbosity=0
)

# ✅ GridSearchCV with KFold(5)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='r2',
    n_jobs=-1,
    cv=cv,
    verbose=2
)

# ✅ 학습
grid_search.fit(X_train, y_train)

# ✅ 최적 모델 및 평가
best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# ✅ 결과 출력
print("📌 Best Params:")
print(grid_search.best_params_)

print(f"\n✅ Train R²: {r2_score(y_train, y_train_pred):.4f}")
print(f"✅ Test R²: {r2_score(y_test, y_test_pred):.4f}")

# 결과
# ✅ Train R²: 0.6263
# ✅ Test R²: 0.3276