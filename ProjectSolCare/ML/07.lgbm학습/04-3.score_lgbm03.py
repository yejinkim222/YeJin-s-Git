# 제거할 컬럼은 빼고
# 하이퍼파라미터 미세조정 실험
import pandas as pd
import lightgbm as lgb
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import early_stopping

# ✅ 데이터 로드
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/09.AD_train_lgbm_final_pruned.csv")

# ✅ 학습에서 제외할 기본 컬럼들
excluded_cols = [
    "gender", "edu_yrs", "edu_level", "years_until_mci", "has_db",
    "hhid", "year", "hhid_year"
]

# ✅ 타겟 분리 + 제외 컬럼 제거
target_col = "years_until_ad"
X = df.drop(columns=[target_col] + excluded_cols, errors="ignore")
y = df[target_col]

# ✅ 학습 / 테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 세밀한 하이퍼파라미터 그리드 정의
param_grid = {
    "learning_rate": [0.005, 0.0075, 0.01],
    "num_leaves": [70, 80, 90],
    "max_depth": [5, 6, 7],
    "reg_alpha": [0.2, 0.4, 0.6],
    "reg_lambda": [1.0, 1.5, 2.0],
    "subsample": [0.8, 0.85, 0.9],
    "colsample_bytree": [0.85, 0.9, 0.95]
}

# ✅ 조합 생성
keys, values = zip(*param_grid.items())
all_combinations = [dict(zip(keys, v)) for v in product(*values)]  # 총 2187개

# ✅ 결과 저장용 리스트
results = []

for idx, params in enumerate(all_combinations):
    model = lgb.LGBMRegressor(
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=42,
        n_estimators=1000
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=50)]
    )

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results.append({
        "실험명": f"model_{idx+1:05d}",
        "Train R²": round(r2_score(y_train, y_pred_train), 6),
        "Test R²": round(r2_score(y_test, y_pred_test), 6),
        **params
    })

# ✅ 결과 정리 및 상위 10개 출력
results_df = pd.DataFrame(results)
top10_df = results_df.sort_values(by="Test R²", ascending=False).head(10)
print("\n📊 상위 10개 성능 결과:")
print(top10_df.to_string(index=False))

# 결과
# 📊 상위 10개 성능 결과:
#         실험명  Train R²  Test R²  learning_rate  num_leaves  max_depth  reg_alpha  reg_lambda  subsample  colsample_bytree
# model_00772  0.471496 0.353371         0.0075          70          5        0.4          1.5       0.90              0.85
# model_01012  0.471496 0.353371         0.0075          80          5        0.4          1.5       0.85              0.85
# model_00766  0.471496 0.353371         0.0075          70          5        0.4          1.5       0.80              0.85
# model_00769  0.471496 0.353371         0.0075          70          5        0.4          1.5       0.85              0.85
# model_00770  0.471496 0.353371         0.0075          70          5        0.4          1.5       0.85              0.90
# model_01258  0.471496 0.353371         0.0075          90          5        0.4          1.5       0.90              0.85
# model_01259  0.471496 0.353371         0.0075          90          5        0.4          1.5       0.90              0.90
# model_00773  0.471496 0.353371         0.0075          70          5        0.4          1.5       0.90              0.90
# model_01009  0.471496 0.353371         0.0075          80          5        0.4          1.5       0.80              0.85
# model_01253  0.471496 0.353371         0.0075          90          5        0.4          1.5       0.80              0.90