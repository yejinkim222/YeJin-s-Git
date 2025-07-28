# 02 결과 토대로
# 중요도 상위 변수 순서대로 제거하며 성능 비교
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/10.AD_train_xgboost_filled.csv"
df = pd.read_csv(file_path)

# 📌 고정 제외 컬럼 + 학습 제외 컬럼 정의
exclude_cols = [
    'hhid', 'year', 'hhid_year',
    'AD_MCI_status', 'edu_yrs', 'edu_level', 'years_until_mci', 'gender'
]
target_col = 'years_until_ad'

# 📌 학습용 데이터 분리
X_all = df.drop(columns=[target_col])
y = df[target_col]

# 📌 실험 대상 feature 목록: 제외 컬럼을 뺀 나머지
feature_candidates = [col for col in X_all.columns if col not in exclude_cols]

# 📌 전체 결과 저장
results = []

# 📌 하나씩 제거하며 실험 반복
for removed_feature in feature_candidates:
    selected_features = [col for col in feature_candidates if col != removed_feature]

    X = X_all[selected_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=30,
        verbose=False
    )

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    results.append((removed_feature, r2))
    print(f"🚫 제외한 변수: {removed_feature:<20} | Test R²: {r2:.4f}")

# 📌 정리된 결과 출력
results_df = pd.DataFrame(results, columns=['excluded_feature', 'test_r2'])
results_df.sort_values(by='test_r2', ascending=False, inplace=True)

print("\n✅ Test R² 향상 실험 결과 (상위 10개):")
print(results_df.head(10).to_string(index=False))

# 결과
# 🚫 제외한 변수: age                  | Test R²: 0.3275
# 🚫 제외한 변수: has_db               | Test R²: 0.3179
# 🚫 제외한 변수: has_hibpe            | Test R²: 0.2327
# 🚫 제외한 변수: years_until_db       | Test R²: 0.2371
# 🚫 제외한 변수: risk_weighted_age    | Test R²: 0.3308
# 🚫 제외한 변수: male_age             | Test R²: 0.3212
# 🚫 제외한 변수: female_age           | Test R²: 0.3278
# 🚫 제외한 변수: age_edu_ratio        | Test R²: 0.2977
# 🚫 제외한 변수: edu_male_diff        | Test R²: 0.3029
# 🚫 제외한 변수: age_male_diff        | Test R²: 0.3271

# ✅ Test R² 향상 실험 결과 (상위 10개):
#  excluded_feature  test_r2
# risk_weighted_age 0.330796
#        female_age 0.327835
#               age 0.327549
#     age_male_diff 0.327144
#          male_age 0.321211
#            has_db 0.317924
#     edu_male_diff 0.302934
#     age_edu_ratio 0.297696
#    years_until_db 0.237060
#         has_hibpe 0.232662