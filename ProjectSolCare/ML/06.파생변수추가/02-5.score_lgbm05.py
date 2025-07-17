# 하이퍼 파라미터 세밀하게 조정
# 얼리스타핑 적용
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor, early_stopping

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 타겟 변수 및 제거 대상
target_col = "years_until_ad"
drop_cols = ['hhid', 'hhid_year', 'year']

# ✅ feature/target 분리
X = df.drop(columns=drop_cols + [target_col])
y = df[target_col]

# ✅ 결측 -1 마스킹된 상태 유지 (이미 전처리 되었다고 가정함)

# ✅ 학습/검증/테스트 분할
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

# ✅ LightGBM 모델 정의
model = LGBMRegressor(
    learning_rate=0.02,
    num_leaves=100,
    max_depth=8,
    reg_alpha=0.2,
    reg_lambda=1,
    subsample=1.0,
    colsample_bytree=0.9,
    random_state=42,
    n_estimators=1000
)

# ✅ 모델 학습 (early stopping 콜백 사용)
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='l2',
    callbacks=[early_stopping(stopping_rounds=50, verbose=True)]
)

# ✅ 예측 및 평가
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# ✅ 중요 변수 출력
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)

# ✅ 결과 출력
print(f"\n🔹 모델: LightGBM (EarlyStopping 적용)")
print(f"  - Train R² = {train_r2:.4f}")
print(f"  - Test  R² = {test_r2:.4f}")
print("  📌 중요 변수 Top 5:")
for k, v in top_features.items():
    print(f"    - {k}: {v:.6f}")

# 결과
# 🔹 모델: LightGBM (EarlyStopping 적용)
#   - Train R² = 0.5108
#   - Test  R² = 0.2186
#   📌 중요 변수 Top 5:
#     - edu_yrs: 829.000000
#     - age: 800.000000
#     - female_age: 684.000000
#     - risk_weighted_age: 656.000000
#     - male_age: 539.000000