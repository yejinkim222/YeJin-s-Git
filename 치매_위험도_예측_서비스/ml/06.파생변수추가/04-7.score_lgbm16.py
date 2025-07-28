# 파생변수 생성하고
# 기여도 낮은 파생변수 제거한 거에
# 하이퍼 파라미터 튜닝해보기
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import early_stopping

# ✅ 데이터 로드
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv")

# ✅ 파생변수 생성
df["age_edu_ratio"] = df["age"] / (df["edu_yrs"] + 1)
df["is_old_and_low_edu"] = ((df["age_group5"] >= 12) & (df["edu_yrs"] < 6)).astype(int)
df["has_db_and_hibpe"] = ((df["years_until_db"] != -1) & (df["years_until_hibpe"] != -1)).astype(int)
df["has_any_chronic"] = ((df["years_until_db"] != -1) | (df["years_until_hibpe"] != -1) | (df["has_hibpe"] == 1)).astype(int)
df["age_male_diff"] = df["male_age"] - df["age"]

# ✅ 필터링 (age_group5 ≥ 10)
df = df[df["age_group5"] >= 10].copy()

# ✅ 입력 변수 / 타겟 정의
target_col = "years_until_ad"
feature_cols = [
    'age', 'edu_yrs', 'male_age', 'female_age', 'risk_weighted_age',
    'years_until_db', 'years_until_hibpe', 'has_hibpe',
    'age_edu_ratio', 'is_old_and_low_edu', 'has_db_and_hibpe',
    'has_any_chronic', 'age_male_diff'
]
X = df[feature_cols]
y = df[target_col]

# ✅ 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 하이퍼파라미터 후보들
param_grid = [
    {"learning_rate": 0.01, "num_leaves": 60, "max_depth": 6, "reg_alpha": 0.3, "reg_lambda": 1.2, "subsample": 0.85, "colsample_bytree": 0.9},
    {"learning_rate": 0.007, "num_leaves": 80, "max_depth": 7, "reg_alpha": 0.5, "reg_lambda": 2.0, "subsample": 0.8, "colsample_bytree": 0.9},
    {"learning_rate": 0.005, "num_leaves": 100, "max_depth": 8, "reg_alpha": 0.6, "reg_lambda": 3.0, "subsample": 0.85, "colsample_bytree": 0.8},
    {"learning_rate": 0.008, "num_leaves": 90, "max_depth": 6, "reg_alpha": 0.4, "reg_lambda": 1.5, "subsample": 0.9, "colsample_bytree": 1.0},
    {"learning_rate": 0.01, "num_leaves": 70, "max_depth": 5, "reg_alpha": 0.3, "reg_lambda": 1.5, "subsample": 0.75, "colsample_bytree": 0.85}
]

results = []

for i, params in enumerate(param_grid, 1):
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=50)]
    )

    y_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, y_pred)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print(f"\n🔹 모델: lgbm_tuned{i}")
    print(f"  - Train R² = {train_r2:.4f}")
    print(f"  - Test  R² = {test_r2:.4f}")
    print("📌 중요 변수 Top 5:")
    print(importance_df.head(5))

    results.append({
        "실험명": f"lgbm_tuned{i}",
        "Train R²": train_r2,
        "Test R²": test_r2
    })

# ✅ 결과 DataFrame 저장 (선택)
results_df = pd.DataFrame(results)
results_df.sort_values(by="Test R²", ascending=False, inplace=True)
print("\n📊 정리된 성능 결과:")
print(results_df)

# 결과
# 🔹 모델: lgbm_tuned5
#   - Train R² = 0.4432
#   - Test  R² = 0.3200
# 📌 중요 변수 Top 5:
#              feature  importance
# 8      age_edu_ratio        1046
# 0                age         825
# 4  risk_weighted_age         654
# 3         female_age         577
# 5     years_until_db         571

# 📊 정리된 성능 결과:
#            실험명  Train R²   Test R²
# 0  lgbm_tuned1  0.471674  0.321502
# 4  lgbm_tuned5  0.443247  0.320000
# 1  lgbm_tuned2  0.453626  0.317192
# 2  lgbm_tuned3  0.489277  0.312059
# 3  lgbm_tuned4  0.412895  0.311408