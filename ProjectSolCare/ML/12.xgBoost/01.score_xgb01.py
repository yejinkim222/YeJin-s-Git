# xgboost 사용해서 변수, 점수 확인
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
data_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(data_path)

# ✅ 타겟 및 제외 컬럼 설정
target_col = "years_until_ad"
exclude_cols = ["ad_year", "hhid_year"]  # 고정 제외

# ✅ 피처 및 타겟 분리
X = df.drop(columns=[target_col] + exclude_cols)
y = df[target_col]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ✅ XGBoost 기본 모델 학습
model = XGBRegressor(random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ✅ 변수 중요도 확인
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# ✅ 상위 10개, 하위 10개 출력
print("✅ 상위 10개 변수:")
print(importances.head(10))

print("\n❌ 하위 10개 변수:")
print(importances.tail(10))

# ✅ 전체 성능 평가
train_r2_all = model.score(X_train, y_train)
test_r2_all = model.score(X_test, y_test)
print(f"\n📊 전체 피처 사용 - Train R²: {train_r2_all:.4f}, Test R²: {test_r2_all:.4f}")

# ✅ 하위 10개 제거
bottom_10 = importances.tail(10).index.tolist()
X_reduced = X.drop(columns=bottom_10)

# ✅ 재분할 및 학습
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reduced, y, random_state=42)
model_reduced = XGBRegressor(random_state=42, n_jobs=-1)
model_reduced.fit(X_train_r, y_train_r)

# ✅ 성능 평가 (하위 10개 제거 후)
train_r2_reduced = model_reduced.score(X_train_r, y_train_r)
test_r2_reduced = model_reduced.score(X_test_r, y_test_r)
print(f"\n📊 하위 10개 제거 후 - Train R²: {train_r2_reduced:.4f}, Test R²: {test_r2_reduced:.4f}")

# 결과
# ✅ 상위 10개 변수:
# hibpe_onset_after      0.772080
# year                   0.089202
# has_hibpe              0.036459
# mci_onset_after        0.029528
# AD_MCI_status          0.012596
# edu_is_low             0.010526
# db_onset_after         0.009808
# risk_factor_sum        0.007327
# risk_weighted_age      0.007076
# age_gender_interact    0.006161
# dtype: float32

# ❌ 하위 10개 변수:
# edu_yrs_missing              0.0
# year_missing                 0.0
# ad_year_missing              0.0
# db_onset_after_missing       0.0
# age_group5                   0.0
# mci_onset_after_missing      0.0
# has_hibpe_missing            0.0
# hibpe_onset_after_missing    0.0
# edu_level                    0.0
# cognitive_decline_flag       0.0
# dtype: float32

# 📊 전체 피처 사용 - Train R²: 0.9987, Test R²: 0.7955

# 📊 하위 10개 제거 후 - Train R²: 0.9987, Test R²: 0.7955