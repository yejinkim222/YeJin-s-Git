import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")

# ✅ 타겟 및 피처 설정
target_col = "years_until_ad"
exclude_cols = ["ad_year", "hhid_year"]
X = df.drop(columns=[target_col] + exclude_cols)
y = df[target_col]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ✅ 모델 학습 (전체 피처)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ✅ 변수 중요도 확인
importances = pd.Series(model.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

# ✅ 상위 10개, 하위 10개 출력
print("✅ 상위 10개 변수:")
print(importances_sorted.head(10))
print("\n❌ 하위 10개 변수:")
print(importances_sorted.tail(10))

# ✅ 하위 10개 제거 후 다시 학습
bottom_10 = importances_sorted.tail(10).index.tolist()
X_reduced = X.drop(columns=bottom_10)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reduced, y, random_state=42)
model_r = RandomForestRegressor(random_state=42)
model_r.fit(X_train_r, y_train_r)
y_pred_r = model_r.predict(X_test_r)
r2_reduced = r2_score(y_test_r, y_pred_r)

print(f"\n✅ 하위 10개 제거 후 Test R²: {r2_reduced:.4f}")

# 결과
# ✅ 상위 10개 변수:
# year                         0.375855
# hibpe_onset_after            0.292310
# hhid                         0.066490
# age                          0.036403
# edu_yrs                      0.030841
# has_hibpe                    0.024591
# hibpe_onset_after_missing    0.022477
# risk_weighted_age            0.021426
# has_hibpe_missing            0.020707
# db_onset_after               0.019480
# dtype: float64

# ❌ 하위 10개 변수:
# AD_MCI_status             0.004834
# cognitive_decline_flag    0.003967
# gender                    0.003591
# risk_factor_sum           0.001991
# has_db                    0.001819
# db_onset_after_missing    0.001700
# edu_is_low                0.001282
# edu_yrs_missing           0.000009
# ad_year_missing           0.000000
# year_missing              0.000000
# dtype: float64

# ✅ 하위 10개 제거 후 Test R²: 0.8062