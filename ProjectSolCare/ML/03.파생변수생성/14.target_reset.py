# 다시 years_until_ad를 타겟으로 변경
# LightGBM 학습 실험
# 결과 똑같아서 폐기...
import pandas as pd
import warnings
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/04.AD_train_years_until_ad_class.csv"
df = pd.read_csv(file_path)

# 📌 변수 정의
target = "years_until_ad"
feature_cols = [
    'age', 'gender', 'edu_yrs', 'has_db', 'AD_MCI_status', 'has_hibpe',
    'edu_level', 'years_until_mci', 'years_until_db', 'years_until_hibpe',
    'years_until_hibpe_missing', 'has_hibpe_missing', 'years_until_mci_missing',
    'years_until_db_missing', 'edu_yrs_missing', 'age_group5',
    'risk_factor_sum', 'edu_is_low', 'risk_weighted_age',
    'male_age', 'female_age'
]

# 📌 스케일링
X = df[feature_cols]
y = df[target]
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 🔍 ElasticNet 기반 피처 선택
enet = ElasticNet(alpha=0.05, l1_ratio=0.5)
enet.fit(X_train, y_train)
selected_features = X_train.columns[enet.coef_ != 0].tolist()
print("✅ 선택된 피처 수:", len(selected_features))
print("✅ 선택된 피처 목록:", selected_features)

# 🎯 LightGBM 회귀 모델 학습 (early stopping은 callbacks로 적용)
model = lgb.LGBMRegressor(
    max_depth=6,
    learning_rate=0.03,
    n_estimators=1000,
    num_leaves=40,
    min_split_gain=1e-3,
    min_child_samples=5,
    random_state=42
)

# ✅ early stopping → callback으로 전달
early_stopping_cb = lgb.early_stopping(stopping_rounds=30, verbose=False)

model.fit(
    X_train[selected_features], y_train,
    eval_set=[(X_test[selected_features], y_test)],
    eval_metric="l2",
    callbacks=[early_stopping_cb]
)

# 📈 평가 및 출력
y_pred = model.predict(X_test[selected_features])
score = r2_score(y_test, y_pred)
print(f"\n✅ LightGBM (회귀) R² Score: {score:.4f}")

importances = pd.Series(model.feature_importances_, index=selected_features)
top_features = importances.sort_values(ascending=False).head(5)
print("\n📌 중요 변수 Top 5:")
print(top_features)
