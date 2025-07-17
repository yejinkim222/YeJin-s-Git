# shap로 변수확인
# ✅ 라이브러리 임포트
import pandas as pd
import numpy as np
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(file_path)

# ✅ 파생변수 생성 (최종 모델 기준)
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["edu_level_bucket"] = df["edu_level"] // 2

# ✅ 타겟 분리
y = df["years_until_ad"]

# ✅ 학습에 사용할 피처만 선택 (기존 최적 모델 기준)
exclude_cols = ["years_until_ad", "ad_year", "hhid", "year", "hhid_year"]
X = df.drop(columns=exclude_cols)

# ✅ 결측치 및 inf 정리
X = X.replace([np.inf, -np.inf], np.nan)
Xy = pd.concat([X, y], axis=1).dropna()
X = Xy.drop(columns=["years_until_ad"])
y = Xy["years_until_ad"]

# ✅ Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 최적 XGBoost 모델 정의 (예진님이 찾은 최적 조합)
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

# ✅ 모델 학습
model.fit(X_train, y_train)

# ✅ SHAP 분석
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# ✅ 요약 플롯 출력 (중요도 + 방향성)
shap.summary_plot(shap_values, X_train, show=True)
