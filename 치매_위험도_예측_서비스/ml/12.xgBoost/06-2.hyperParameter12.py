# feature importance 확인해서
# 상하위 확인해보기
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(file_path)

# ✅ 파생변수 생성
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["edu_level_bucket"] = df["edu_level"] // 2

# ✅ 제외할 하위 중요도 피처
features_to_exclude = [
    "hibpe_onset_after_missing",
    "db_onset_after_missing",
    "edu_yrs_missing",
    "ad_year_missing",
    "year_missing"
]

# ✅ 타겟 정의
target_col = "years_until_ad"

# ✅ 특징 및 타겟 정의
X_full = df.drop(columns=["years_until_ad", "ad_year", "hhid_year"])
y_full = df["years_until_ad"]

# ✅ 결측치 제거 최소화
df_model = pd.concat([X_full, y_full], axis=1).dropna()
X_full = df_model.drop(columns=["years_until_ad"])
y_full = df_model["years_until_ad"]

# ✅ 학습/검증 분할 (공통)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

# ✅ 하위 5개 제외한 X 구성
X_train_reduced = X_train_full.drop(columns=features_to_exclude, errors="ignore")
X_test_reduced = X_test_full.drop(columns=features_to_exclude, errors="ignore")

# ✅ 공통 모델 설정
model_params = {
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.2,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.005,
    "reg_lambda": 0.5,
    "min_child_weight": 3,
    "gamma": 0,
    "max_delta_step": 0
}

# ✅ (1) 기존 모델 학습
model_full = XGBRegressor(**model_params)
model_full.fit(X_train_full, y_train)

# ✅ (2) 하위 5개 제외 모델 학습
model_reduced = XGBRegressor(**model_params)
model_reduced.fit(X_train_reduced, y_train)

# ✅ 성능 출력
print("✅ [기존 전체 피처 모델]")
print("✅ Train R²:", round(model_full.score(X_train_full, y_train), 4))
print("✅ Test R²:", round(model_full.score(X_test_full, y_test), 4))

print("\n✅ [하위 5개 제외 모델]")
print("✅ Train R²:", round(model_reduced.score(X_train_reduced, y_train), 4))
print("✅ Test R²:", round(model_reduced.score(X_test_reduced, y_test), 4))

# ✅ 중요도 비교용 출력
feature_names_reduced = X_train_reduced.columns.tolist()
importances = model_reduced.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": feature_names_reduced,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n🔼 Feature Importance 상위 5개 (하위 5개 제거 모델 기준):")
print(feature_importance_df.head(5).to_string(index=False))

print("\n🔽 Feature Importance 하위 5개 (하위 5개 제거 모델 기준):")
print(feature_importance_df.tail(5).to_string(index=False))

# 결과
# ✅ [기존 전체 피처 모델]
# ✅ Train R²: 0.9993
# ✅ Test R²: 0.8337

# ✅ [하위 5개 제외 모델]
# ✅ Train R²: 0.9993
# ✅ Test R²: 0.82

# 🔼 Feature Importance 상위 5개 (하위 5개 제거 모델 기준):
#                feature  importance
#      hibpe_onset_after    0.400944
#              has_hibpe    0.265415
# cognitive_decline_flag    0.157981
#                   year    0.060609
#      has_hibpe_missing    0.023481

# 🔽 Feature Importance 하위 5개 (하위 5개 제거 모델 기준):
#           feature  importance
#           edu_yrs    0.002608
# risk_weighted_age    0.002295
#            gender    0.001973
#            has_db    0.001343
#  edu_level_bucket    0.000000