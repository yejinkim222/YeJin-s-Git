# 파생변수 추가해서
# 다시 하이퍼파라미터 조정
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import product
from lightgbm import early_stopping

# ✅ 데이터 로드
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/09.AD_train_lgbm_final_pruned.csv")

# ✅ 입력 / 타겟 분리
X = df.drop(columns=["years_until_ad", "hhid", "year", "hhid_year"], errors="ignore")
y = df["years_until_ad"]

# ✅ 학습 / 테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 그리드 조합 설정
learning_rates = [0.005, 0.008, 0.01]
num_leaves_list = [60, 80, 100]
max_depths = [5, 6, 7]
reg_alphas = [0.3, 0.5]
reg_lambdas = [1.0, 1.5]
subsamples = [0.85]
colsample_bytrees = [0.9]

param_grid = list(product(learning_rates, num_leaves_list, max_depths, reg_alphas, reg_lambdas, subsamples, colsample_bytrees))

# ✅ 결과 저장
results = []

for idx, (lr, nl, md, ra, rl, ss, csbt) in enumerate(param_grid):
    model = lgb.LGBMRegressor(
        learning_rate=lr,
        num_leaves=nl,
        max_depth=md,
        reg_alpha=ra,
        reg_lambda=rl,
        subsample=ss,
        colsample_bytree=csbt,
        n_estimators=1000,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=50)],
    )

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    results.append({
        "실험명": f"model_{idx+1:05d}",
        "Train R²": round(train_r2, 6),
        "Test R²": round(test_r2, 6),
        "lr": lr, "num_leaves": nl, "max_depth": md,
        "reg_alpha": ra, "reg_lambda": rl,
        "subsample": ss, "colsample_bytree": csbt
    })

# ✅ 상위 10개 결과 출력
results_df = pd.DataFrame(results).sort_values(by="Test R²", ascending=False).reset_index(drop=True)
print("\n📊 상위 10개 성능 결과:")
print(results_df.head(10))

# 결과
# 📊 상위 10개 성능 결과:
#            실험명  Train R²   Test R²  ...  reg_lambda  subsample  colsample_bytree0  model_00070  0.495630  0.342133  ...         1.5       0.85               0.9   
# 1  model_00058  0.495630  0.342133  ...         1.5       0.85               0.9   
# 2  model_00046  0.495630  0.342133  ...         1.5       0.85               0.9   
# 3  model_00094  0.512007  0.341573  ...         1.5       0.85               0.9   
# 4  model_00082  0.512007  0.341573  ...         1.5       0.85               0.9   
# 5  model_00106  0.512007  0.341573  ...         1.5       0.85               0.9   
# 6  model_00105  0.506389  0.340662  ...         1.0       0.85               0.9   
# 7  model_00081  0.506389  0.340662  ...         1.0       0.85               0.9   
# 8  model_00093  0.506389  0.340662  ...         1.0       0.85               0.9   
# 9  model_00057  0.503301  0.340652  ...         1.0       0.85               0.9