import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")

# ✅ 파생변수 생성
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)

# ✅ 모델 제외 컬럼 정의
exclude_cols = ["years_until_ad", "ad_year", "hhid_year", "hhid", "ad_year_missing", "year", "year_missing"]
target = "years_until_ad"

df_model = df.drop(columns=exclude_cols).copy()
df_model[target] = df[target]
df_model = df_model.dropna()

X = df_model.drop(columns=[target])
y = df_model[target]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 최적 파라미터 조합
best_params = {
    "n_estimators": 498,
    "max_depth": 12,
    "learning_rate": 0.14997745579336536,
    "subsample": 0.8989092608354626,
    "colsample_bytree": 0.9284142503510097,
    "reg_alpha": 1.4386504039680161,
    "reg_lambda": 0.03803761653911136,
    "min_child_weight": 1,
    "gamma": 9.219422177382182,
    "max_delta_step": 5,
    "random_state": 42
}

# ✅ 전체 모델 학습 및 중요도 출력
model = XGBRegressor(**best_params)
model.fit(X_train, y_train)

# ✅ 중요도 순 정렬
feature_importances = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=True)

# ✅ 중요도 시각화
plt.figure(figsize=(8, len(feature_importances) * 0.4))
plt.barh(feature_importances["feature"], feature_importances["importance"])
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.show()

# ✅ 성능 비교용: 원본 성능 기록
baseline_train_score = model.score(X_train, y_train)
baseline_test_score = model.score(X_test, y_test)

print(f"✅ Baseline ▶ Train R²: {baseline_train_score:.4f} | Test R²: {baseline_test_score:.4f}")

# ✅ Step 2: 한 컬럼씩 제거하며 성능 테스트
results = []
for col in feature_importances["feature"]:
    X_train_mod = X_train.drop(columns=[col])
    X_test_mod = X_test.drop(columns=[col])
    
    model_mod = XGBRegressor(**best_params)
    model_mod.fit(X_train_mod, y_train)
    
    train_r2 = model_mod.score(X_train_mod, y_train)
    test_r2 = model_mod.score(X_test_mod, y_test)
    
    results.append({
        "removed_feature": col,
        "train_r2": round(train_r2, 4),
        "test_r2": round(test_r2, 4)
    })

# ✅ 결과 테이블 출력
results_df = pd.DataFrame(results).sort_values(by="test_r2", ascending=False)
print("\n📊 Feature 제거 후 성능 비교:")
print(results_df.to_string(index=False))

# 결과
# ✅ Baseline ▶ Train R²: 0.9475 | Test R²: 0.6752