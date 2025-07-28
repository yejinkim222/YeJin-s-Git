# 일단 로그를 버리고...
# 다시 xgboost 하이퍼파라미터 튜닝하기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ✅ 데이터 경로
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ has_hibpe 전처리
df.loc[df['has_hibpe_missing'] == 1, 'has_hibpe'] = -1
df = df.drop(columns=['has_hibpe_missing'])

# ✅ 정규화 대상 컬럼
numeric_cols = [
    "age", "edu_yrs", "risk_weighted_age", "male_age", "female_age",
    "log_risk_weighted_age", "age_group5",
    "years_until_db", "years_until_hibpe", "years_until_mci"
]

# ✅ Z-score 정규화
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ✅ X, y 분리
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]
X = X.dropna()
y = y.loc[X.index]

# ✅ 실험 조합 리스트
experiments = [
    {"name": "A", "n_estimators": 700, "max_depth": 4, "learning_rate": 0.02, "reg_alpha": 0.1, "reg_lambda": 1},
    {"name": "B", "n_estimators": 1000, "max_depth": 5, "learning_rate": 0.03, "reg_alpha": 0.1, "reg_lambda": 1},
    {"name": "C", "n_estimators": 700, "max_depth": 5, "learning_rate": 0.03, "reg_alpha": 0.3, "reg_lambda": 2},
    {"name": "D", "n_estimators": 1000, "max_depth": 6, "learning_rate": 0.01, "reg_alpha": 0, "reg_lambda": 1},
    {"name": "E", "n_estimators": 800, "max_depth": 4, "learning_rate": 0.02, "reg_alpha": 0.5, "reg_lambda": 2}
]

# ✅ 결과 저장용 리스트
results = []

# ✅ 실험 루프
for exp in experiments:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=exp["n_estimators"],
        max_depth=exp["max_depth"],
        learning_rate=exp["learning_rate"],
        reg_alpha=exp["reg_alpha"],
        reg_lambda=exp["reg_lambda"],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    top_feats = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)

    results.append({
        "실험": exp["name"],
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4),
        "Params": exp,
        "Top 5 Features": top_feats.to_dict()
    })

# ✅ 결과 출력
for res in results:
    print(f"\n🔹 실험 {res['실험']} - Params: {res['Params']}")
    print(f"  - Train R² = {res['Train R²']}")
    print(f"  - Test  R² = {res['Test R²']}")
    print("  📌 중요 변수 Top 5:")
    for feat, val in res["Top 5 Features"].items():
        print(f"    - {feat}: {val:.6f}")
