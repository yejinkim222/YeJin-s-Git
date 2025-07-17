# XGBoost 전체 컬럼 학습 및 중요도 확인
# ✅ XGBoost 전체 컬럼 학습 (식별자 제거 후 중요도 확인)
# ✅ 필요한 라이브러리
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/10.AD_train_xgboost_filled.csv"
df = pd.read_csv(file_path)

# ✅ 학습 제외할 컬럼
exclude_cols = ['hhid', 'year', 'hhid_year']
target_col = 'years_until_ad'

# ✅ 입력(X), 타겟(y) 분리
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

# ✅ train/test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ XGBoost 모델 정의
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    verbosity=1
)

# ✅ 학습 (early stopping 포함)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=30,
    verbose=50
)

# ✅ 예측 및 평가
y_pred = xgb_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\n✅ XGBoost 베이스라인 Test R²: {r2:.4f}")

# ✅ XGBoost 모델에서 중요도 추출
booster = xgb_model.get_booster()
score_dict = booster.get_score(importance_type='gain')

# ✅ 중요도 DataFrame으로 정렬
importance_df = pd.DataFrame.from_dict(score_dict, orient='index', columns=['gain'])
importance_df.index.name = 'feature'
importance_df.reset_index(inplace=True)
importance_df.sort_values(by='gain', ascending=False, inplace=True)

# ✅ 출력 (상위 30, 하위 30)
print("\n📈 [상위 30개 중요 변수]")
print(importance_df.head(5).to_string(index=False))

print("\n📉 [하위 30개 중요 변수]")
print(importance_df.tail(5).to_string(index=False))

# 결과
# ✅ XGBoost 베이스라인 Test R²: 0.3419

# 📈 [상위 30개 중요 변수]
#        feature       gain
#      has_hibpe 245.270416
#       male_age  49.170044
#  edu_male_diff  40.327766
#     female_age  39.177586
# years_until_db  38.526020

# 📉 [하위 30개 중요 변수]
#         feature      gain
#   AD_MCI_status 30.148443
#         edu_yrs 27.481710
#       edu_level 22.371977
# years_until_mci 20.330746
#          gender 14.796475