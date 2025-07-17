# 성능 제일 좋았던 모델 복원
import pandas as pd
# ✅ 데이터 로딩
file_path= "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(file_path)

# ✅ 파생변수 생성 동일하게 복원
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["edu_level_bucket"] = df["edu_level"] // 2

# ✅ 특징 및 타겟 정의 (실험과 동일하게)
X = df.drop(columns=["years_until_ad", "ad_year", "hhid_year"])
y = df["years_until_ad"]

# ✅ 결측치 제거 최소화
df_model = pd.concat([X, y], axis=1).dropna()
X = df_model.drop(columns=["years_until_ad"])
y = df_model["years_until_ad"]

# ✅ train/test 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 복원할 베스트 모델
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.2,
    subsample=1.0,
    colsample_bytree=0.8,
    reg_alpha=0.005,
    reg_lambda=0.5,
    min_child_weight=3,
    gamma=0,
    max_delta_step=0
)

model.fit(X_train, y_train)

# ✅ 성능 평가
print("✅ Train R²:", round(model.score(X_train, y_train), 4))
print("✅ Test R²:", round(model.score(X_test, y_test), 4))

# 결과
# ✅ Train R²: 0.9993
# ✅ Test R²: 0.8337