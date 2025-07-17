# 중요도 낮은 거 제거하고 학습
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(file_path)

# ✅ 파생변수 생성 (최종 기준)
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["edu_level_bucket"] = df["edu_level"] // 2

# ✅ 타겟 분리
y = df["years_until_ad"]

# ✅ 제거할 컬럼 정의
remove_features = [
    "edu_level", "risk_factor_sum", "has_db",
    "edu_yrs_missing", "ad_year_missing", "year_missing",
    "db_onset_after_missing", "mci_onset_after_missing",
    "hibpe_onset_after_missing"
]

# ✅ 입력 피처 정의
exclude_cols = ["years_until_ad", "ad_year", "hhid", "year", "hhid_year"] + remove_features
X = df.drop(columns=exclude_cols)

# ✅ 결측치 및 inf 처리
X = X.replace([np.inf, -np.inf], np.nan)
Xy = pd.concat([X, y], axis=1).dropna()
X = Xy.drop(columns=["years_until_ad"])
y = Xy["years_until_ad"]

# ✅ Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ XGBoost 최적 모델 정의 (예진님 조합)
model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.2,
    subsample=1.0,
    colsample_bytree=0.8,
    reg_alpha=0.01,
    reg_lambda=1.0,
    min_child_weight=5,
    gamma=0.5,
    max_delta_step=0
)

# ✅ 학습
model.fit(X_train, y_train)

# ✅ 성능 평가
print("✅ Train R²:", round(model.score(X_train, y_train), 4))
print("✅ Test R²:", round(model.score(X_test, y_test), 4))

# 결과
# ✅ Train R²: 0.9579
# ✅ Test R²: 0.6172