# 하이퍼 파라미터 정밀하게 튜닝
# 26,244가지 조합 전부 해보고 상위 10개 출력하는 코드
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import product
from lightgbm import early_stopping

# 📌 데이터 로드
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/09.AD_train_lgbm_final_pruned.csv")

# ✅ 타겟과 제외할 기본 컬럼 지정
target_col = "years_until_ad"
exclude_cols = [
    "edu_yrs", "edu_level", "years_until_mci", "has_db", "gender",
    "hhid", "year", "hhid_year"
]

# ✅ 입력 / 타겟 분리
X = df.drop(columns=[target_col] + exclude_cols, errors="ignore")
y = df[target_col]

# 📌 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 하이퍼파라미터 그리드 정의 (총 26,244 조합)
param_grid = {
    "learning_rate": [0.005, 0.01, 0.02],
    "num_leaves": [31, 40, 50],
    "max_depth": [4, 5, 6],
    "reg_alpha": [0.0, 0.1, 0.3, 0.5],
    "reg_lambda": [1.0, 1.2, 1.5],
    "subsample": [0.8],
    "colsample_bytree": [0.9]
}

# 📌 모든 조합 생성
all_combinations = list(product(*param_grid.values()))
param_names = list(param_grid.keys())

# 📌 결과 저장용 리스트
results = []

# 📌 전체 조합 반복
for idx, values in enumerate(all_combinations):
    params = dict(zip(param_names, values))
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        **params,
        random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=50)],
    )
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    results.append({
        "실험명": f"model_{idx+1:05d}",
        "Train R²": train_r2,
        "Test R²": test_r2,
        **params
    })

# 📌 결과 정렬 및 상위 10개 출력
results_df = pd.DataFrame(results)
top10 = results_df.sort_values(by="Test R²", ascending=False).head(10)

# 📌 결과 출력
print("\n📊 정리된 상위 10개 성능 결과:")
print(top10[["실험명", "Train R²", "Test R²"]])

# 결과
# 📊 정리된 상위 10개 성능 결과:
#              실험명  Train R²   Test R²
# 237  model_00238  0.465700  0.355225
# 273  model_00274  0.465700  0.355225
# 309  model_00310  0.465700  0.355225
# 235  model_00236  0.465805  0.355137
# 307  model_00308  0.465805  0.355137
# 271  model_00272  0.465805  0.355137
# 230  model_00231  0.470829  0.354613
# 266  model_00267  0.470829  0.354613
# 302  model_00303  0.470829  0.354613
# 231  model_00232  0.473231  0.354484