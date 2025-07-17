# 하위 5개가 중요도 0이라서 영향 없어서
# 그 다음 하위 5개 제거해보기
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")

# ✅ 타겟 컬럼
target_col = "years_until_ad"

# ✅ 미리 제거할 중요도 0 변수
pre_exclude = [
    "mci_onset_after_missing", "edu_yrs_missing",
    "db_onset_after_missing", "ad_year_missing", "year_missing"
]

# ✅ 비학습용 컬럼
non_features = ["hhid", "year", "hhid_year", target_col]

# ✅ X, y 분리 및 결측 제거
X = df.drop(columns=non_features + pre_exclude).dropna()
y = df.loc[X.index, target_col]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 1차 학습 (제거된 상태에서)
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# ✅ 중요도 상/하위 5개 출력
importances = pd.Series(model.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

top5 = importances_sorted.head(5)
bottom5 = importances_sorted.tail(5)

print("✅ Feature Importance 상위 5개:\n", top5)
print("\n✅ Feature Importance 하위 5개:\n", bottom5)

# ✅ 제거 전 성능
train_r2_before = r2_score(y_train, model.predict(X_train))
test_r2_before = r2_score(y_test, model.predict(X_test))

# ✅ 하위 5개 제거 후 학습
X_reduced = X.drop(columns=bottom5.index)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

model_reduced = LGBMRegressor(random_state=42)
model_reduced.fit(X_train_r, y_train_r)

train_r2_after = r2_score(y_train_r, model_reduced.predict(X_train_r))
test_r2_after = r2_score(y_test_r, model_reduced.predict(X_test_r))

print("\n📊 R² 비교")
print(f"제거 전 Train R²: {train_r2_before:.4f}")
print(f"제거 전 Test  R²: {test_r2_before:.4f}")
print(f"제거 후 Train R²: {train_r2_after:.4f}")
print(f"제거 후 Test  R²: {test_r2_after:.4f}")

# 결과
# ✅ Feature Importance 상위 5개:
#  age                    732
# edu_yrs                500
# ad_year                441
# risk_weighted_age      377
# age_gender_interact    287
# dtype: int32

# ✅ Feature Importance 하위 5개:
#  edu_level                    0
# age_group5                   0
# edu_is_low                   0
# hibpe_onset_after_missing    0
# has_hibpe_missing            0

# 📊 R² 비교
# 제거 전 Train R²: 0.9406
# 제거 전 Test  R²: 0.7518
# 제거 후 Train R²: 0.9406
# 제거 후 Test  R²: 0.7518