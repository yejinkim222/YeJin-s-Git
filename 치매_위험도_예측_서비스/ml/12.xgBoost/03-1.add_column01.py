# 계속 오버피팅 나와서
# 파생변수 추가 실험
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")

# ✅ 타겟/제외 컬럼
target_col = "years_until_ad"
exclude_cols = ["ad_year", "hhid_year"]

# ✅ 기본 피처 정의
base_features = [col for col in df.columns if col not in exclude_cols + [target_col]]

# ✅ 파생변수 생성 로직
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["edu_x_risk"] = df["edu_yrs"] * df["risk_factor_sum"]
df["female_age_x_edu"] = df.apply(lambda row: row["age"] * row["edu_yrs"] if row["gender"] == 0 else 0, axis=1)
df["age_x_AD_MCI_status"] = df["age"] * df["AD_MCI_status"]
df["age_group5_x_db"] = df["age_group5"] * df["has_db"]
df["risk_age_db_hibpe"] = df["age"] * (df["has_db"] + df["has_hibpe"])

# ✅ 파생변수 리스트
new_features = [
    "age_x_edu",
    "edu_x_risk",
    "female_age_x_edu",
    "age_x_AD_MCI_status",
    "age_group5_x_db",
    "risk_age_db_hibpe"
]

# ✅ 베이스라인 모델 학습
X_base = df[base_features]
y = df[target_col]
X_train_base, X_test_base, y_train, y_test = train_test_split(X_base, y, random_state=42)
base_model = xgb.XGBRegressor(
    n_estimators=260, max_depth=6, learning_rate=0.2,
    subsample=1.0, colsample_bytree=0.8,
    reg_alpha=0.01, reg_lambda=2,
    random_state=42, n_jobs=-1, verbosity=0
)
base_model.fit(X_train_base, y_train, early_stopping_rounds=20,
               eval_set=[(X_test_base, y_test)], verbose=False)
base_train_r2 = base_model.score(X_train_base, y_train)
base_test_r2 = base_model.score(X_test_base, y_test)

# ✅ 변수별 성능 비교
results = []
for f in new_features:
    X_temp = df[base_features + [f]]
    X_train, X_test, _, _ = train_test_split(X_temp, y, random_state=42)
    model = xgb.XGBRegressor(
        n_estimators=260, max_depth=6, learning_rate=0.2,
        subsample=1.0, colsample_bytree=0.8,
        reg_alpha=0.01, reg_lambda=2,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_train, y_train, early_stopping_rounds=20,
              eval_set=[(X_test, y_test)], verbose=False)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    results.append({
        "파생변수": f,
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4),
        "Δ Test R²": round(test_r2 - base_test_r2, 4)
    })

# ✅ 출력
results_df = pd.DataFrame(results).sort_values(by="Δ Test R²", ascending=False)
print("✅ 파생변수 개별 추가 효과 (Test R² 기준):")
print(results_df)

# 결과
# ✅ 파생변수 개별 추가 효과 (Test R² 기준):
#                   파생변수  Train R²  Test R²  Δ Test R²
# 0            age_x_edu    0.9995   0.8336     0.0006
# 1           edu_x_risk    0.9922   0.8310    -0.0020
# 2     female_age_x_edu    0.9981   0.8250    -0.0080
# 4      age_group5_x_db    0.9991   0.8176    -0.0154
# 5    risk_age_db_hibpe    0.9993   0.8132    -0.0198
# 3  age_x_AD_MCI_status    0.9657   0.8059    -0.0271