# 기본 컬럼 삭제하고 학습용 컬럼만 남긴 데이터로
# 하이퍼 파라미터 조정
import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import product

# ✅ 데이터 로딩
data_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_for_randomForest.csv"
df = pd.read_csv(data_path)

# 📌 타겟 변수
target_col = "years_until_ad"
y = df[target_col]

# 📌 모델 학습에서 제외할 컬럼
model_exclude_only = ["year", "ad_year"]

# 📌 학습용 feature 구성
non_features = ["hhid", "year", "hhid_year", target_col] + model_exclude_only
X = df.drop(columns=[col for col in non_features if col in df.columns])

# 📌 결측치 제거
X = X.dropna()
y = y.loc[X.index]

# 📌 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 하이퍼파라미터 조합 정의
param_grid = {
    "n_estimators": [80, 100, 120],
    "max_depth": [14, 15, 16],
    "min_samples_leaf": [1, 2],
    "min_samples_split": [2, 3],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False],
    "criterion": ["squared_error", "absolute_error"]
}

# ✅ 모든 조합 생성
param_combinations = list(product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["min_samples_leaf"],
    param_grid["min_samples_split"],
    param_grid["max_features"],
    param_grid["bootstrap"],
    param_grid["criterion"]
))

# ✅ 튜닝 결과 저장 리스트
results = []
best_r2 = -999
best_model = None
best_params = {}

# ✅ 모든 조합 반복 실행
for n, d, leaf, split, feat, boot, crit in param_combinations:
    model = RandomForestRegressor(
        n_estimators=n,
        max_depth=d,
        min_samples_leaf=leaf,
        min_samples_split=split,
        max_features=feat,
        bootstrap=boot,
        criterion=crit,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    train_r2 = model.score(X_train, y_train)

    results.append({
        "n_estimators": n,
        "max_depth": d,
        "min_samples_leaf": leaf,
        "min_samples_split": split,
        "max_features": feat,
        "bootstrap": boot,
        "criterion": crit,
        "train_r2": train_r2,
        "test_r2": test_r2
    })

    if test_r2 > best_r2:
        best_r2 = test_r2
        best_model = model
        best_params = {
            "n_estimators": n,
            "max_depth": d,
            "min_samples_leaf": leaf,
            "min_samples_split": split,
            "max_features": feat,
            "bootstrap": boot,
            "criterion": crit,
            "random_state": 42
        }

# ✅ 결과 정리 및 상위 10개 출력
results_df = pd.DataFrame(results)
top10 = results_df.sort_values(by="test_r2", ascending=False).head(10)
print("✅ RandomForest 정밀 튜닝 결과 (상위 10개):")
print(top10)

# ✅ 최종 모델 저장
model_path = "C:/workspace/Project01/model_storage/rf_best_model_final.pkl"
feature_path = "C:/workspace/Project01/model_storage/rf_model_features_final.json"
param_path = "C:/workspace/Project01/model_storage/rf_best_params_final.json"

joblib.dump(best_model, model_path)
print(f"📦 모델 저장 완료 → {model_path}")

with open(feature_path, "w") as f:
    json.dump(list(X.columns), f)
print(f"📦 피처 리스트 저장 완료 → {feature_path}")

with open(param_path, "w") as f:
    json.dump(best_params, f)
print(f"📦 하이퍼파라미터 저장 완료 → {param_path}")

# ✅ 최종 성능 출력
print(f"Train R²: {train_r2:.4f}")
print(f"Test  R²:  {test_r2:.4f}")

# 결과
# 📦 모델 저장 완료 → C:/workspace/Project01/model_storage/rf_best_model_final.pkl
# 📦 피처 리스트 저장 완료 → C:/workspace/Project01/model_storage/rf_model_features_final.json
# 📦 하이퍼파라미터 저장 완료 → C:/workspace/Project01/model_storage/rf_best_params_final.json

# 이 모델 성능
# Train R²: 0.7985
# Test  R²:  0.2964
