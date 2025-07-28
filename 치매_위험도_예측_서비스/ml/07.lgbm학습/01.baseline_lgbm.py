# 위에서 만든 파생변수 저장한 데이터로
# lgbm 하이퍼 파라미터 튜닝
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import early_stopping

# ✅ 데이터 로드
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

# ✅ 학습 / 테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 튜닝할 하이퍼파라미터 목록
param_grid = [
    {"name": "lgbm_tuned_a", "learning_rate": 0.01, "num_leaves": 80, "max_depth": 6, "reg_alpha": 0.4, "reg_lambda": 1.0, "subsample": 0.9, "colsample_bytree": 0.9},
    {"name": "lgbm_tuned_b", "learning_rate": 0.008, "num_leaves": 100, "max_depth": 8, "reg_alpha": 0.6, "reg_lambda": 2.0, "subsample": 0.8, "colsample_bytree": 0.85},
    {"name": "lgbm_tuned_c", "learning_rate": 0.005, "num_leaves": 120, "max_depth": 9, "reg_alpha": 0.5, "reg_lambda": 3.0, "subsample": 0.85, "colsample_bytree": 0.8},
]

# ✅ 결과 저장용 리스트
results = []

for params in param_grid:
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

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # 성능 기록
    results.append({
        "실험명": params["name"],
        "Train R²": round(train_r2, 6),
        "Test R²": round(test_r2, 6)
    })

# ✅ 결과 출력
results_df = pd.DataFrame(results)
print("\n📊 하이퍼파라미터 튜닝 결과:")
print(results_df)

# 결과
# 📊 하이퍼파라미터 튜닝 결과:
#             실험명  Train R²   Test R²
# 0  lgbm_tuned_a  0.477689  0.332568
# 1  lgbm_tuned_b  0.418024  0.313849
# 2  lgbm_tuned_c  0.499925  0.316436