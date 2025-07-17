# 파생변수 추가실험
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 로드
data_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(data_path)

# ✅ 공통 설정
target_col = "years_until_ad"
exclude_cols = ["ad_year", "hhid_year"]

# ✅ 기본 변수 지정
base_features = [col for col in df.columns if col not in [target_col] + exclude_cols]
X_base = df[base_features]
y = df[target_col]

# ✅ 추천 파생변수 정의 (6종)
df["age_x_has_hibpe"] = df["age"] * df["has_hibpe"]
df["edu_level_x_AD_MCI_status"] = df["edu_level"] * df["AD_MCI_status"]
df["has_db_x_has_hibpe"] = df["has_db"] * df["has_hibpe"]
df["age_x_risk_weighted_age"] = df["age"] * df["risk_weighted_age"]
df["age_x_edu_level"] = df["age"] * df["edu_level"]
df["edu_level_x_has_db"] = df["edu_level"] * df["has_db"]

new_vars = [
    "age_x_has_hibpe",
    "edu_level_x_AD_MCI_status",
    "has_db_x_has_hibpe",
    "age_x_risk_weighted_age",
    "age_x_edu_level",
    "edu_level_x_has_db"
]

# ✅ 실험 결과 저장용 리스트
results = []

# ✅ 파생변수별로 성능 비교
for new_var in new_vars:
    X = X_base.copy()
    X[new_var] = df[new_var]  # 변수 추가

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=260,
        max_depth=6,
        learning_rate=0.2,
        subsample=1.0,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    results.append({
        "변수": new_var,
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4),
        "Δ Test R²": round(test_r2 - 0.8335, 4)  # 기존 최고점 기준
    })

# ✅ 결과 출력
results_df = pd.DataFrame(results)
print("✅ 추천 파생변수 실험 결과:")
print(results_df.sort_values(by="Test R²", ascending=False))

# 결과
# ✅ 추천 파생변수 실험 결과:
#                           변수  Train R²  Test R²  Δ Test R²
# 4            age_x_edu_level    0.9997   0.8334    -0.0001
# 2         has_db_x_has_hibpe    0.9996   0.8272    -0.0063
# 3    age_x_risk_weighted_age    0.9995   0.8255    -0.0080
# 5         edu_level_x_has_db    0.9996   0.8222    -0.0113
# 0            age_x_has_hibpe    0.9996   0.8190    -0.0145
# 1  edu_level_x_AD_MCI_status    0.9996   0.8161    -0.0174