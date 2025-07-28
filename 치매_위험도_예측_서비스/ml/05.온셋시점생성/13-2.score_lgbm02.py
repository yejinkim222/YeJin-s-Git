# lgbm에서 하이퍼 파라미터 튜닝
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# ✅ 전처리: -1 마스킹 & 컬럼 제거
df.loc[df['has_hibpe_missing'] == 1, 'has_hibpe'] = -1
df = df.drop(columns=['has_hibpe_missing'])

# ✅ 정규화 대상
numeric_cols = [
    "age", "edu_yrs", "risk_weighted_age", "male_age", "female_age",
    "log_risk_weighted_age", "age_group5", "years_until_db", "years_until_hibpe", "years_until_mci"
]
df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

# ✅ feature/target 분리
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]
X = X.dropna()
y = y.loc[X.index]

# ✅ train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 하이퍼파라미터 조합 리스트
param_list = [
    {"name": "lgbm1", "learning_rate": 0.05, "num_leaves": 31, "max_depth": 5, "reg_alpha": 0.1, "reg_lambda": 1},
    {"name": "lgbm2", "learning_rate": 0.03, "num_leaves": 50, "max_depth": 6, "reg_alpha": 0.3, "reg_lambda": 1},
    {"name": "lgbm3", "learning_rate": 0.02, "num_leaves": 60, "max_depth": 7, "reg_alpha": 0.5, "reg_lambda": 1.5},
    {"name": "lgbm4", "learning_rate": 0.01, "num_leaves": 70, "max_depth": 8, "reg_alpha": 0.3, "reg_lambda": 2},
    {"name": "lgbm5", "learning_rate": 0.05, "num_leaves": 40, "max_depth": 5, "reg_alpha": 0, "reg_lambda": 0},
]

# ✅ 결과 저장
results = []

for params in param_list:
    model = lgb.LGBMRegressor(
        learning_rate=params['learning_rate'],
        num_leaves=params['num_leaves'],
        max_depth=params['max_depth'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        n_estimators=1000,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importance.sort_values(ascending=False).head(5)

    results.append({
        "모델": params['name'],
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4),
        "Top 5 Features": top_features.to_dict(),
        "Params": params
    })

# ✅ 상위 5개 결과 출력
top_results = sorted(results, key=lambda x: x['Test R²'], reverse=True)[:5]

print("✅ 상위 5개 LightGBM 결과\n")
for res in top_results:
    print(f"🔹 모델: {res['모델']}")
    print(f"  - Train R² = {res['Train R²']}")
    print(f"  - Test  R² = {res['Test R²']}")
    print("  📌 중요 변수 Top 5:")
    for k, v in res["Top 5 Features"].items():
        print(f"    - {k}: {v:.6f}")
    print(f"  🔧 Params: {res['Params']}")
    print("-" * 50)

# 결과
# ✅ 상위 5개 LightGBM 결과

# 🔹 모델: lgbm4
#   - Train R² = 0.5834
#   - Test  R² = 0.2407
#   📌 중요 변수 Top 5:
#     - age: 3413.000000
#     - edu_yrs: 3354.000000
#     - risk_weighted_age: 2952.000000
#     - female_age: 2836.000000
#     - male_age: 2333.000000
#   🔧 Params: {'name': 'lgbm4', 'learning_rate': 0.01, 'num_leaves': 70, 'max_depth': 8, 'reg_alpha': 
# 0.3, 'reg_lambda': 2}
# --------------------------------------------------
# 🔹 모델: lgbm3
#   - Train R² = 0.6623
#   - Test  R² = 0.2189
#   📌 중요 변수 Top 5:
#     - edu_yrs: 3079.000000
#     - age: 2938.000000
#     - risk_weighted_age: 2596.000000
#     - female_age: 2580.000000
#     - male_age: 1914.000000
#   🔧 Params: {'name': 'lgbm3', 'learning_rate': 0.02, 'num_leaves': 60, 'max_depth': 7, 'reg_alpha': 
# 0.5, 'reg_lambda': 1.5}
# --------------------------------------------------
# 🔹 모델: lgbm2
#   - Train R² = 0.6803
#   - Test  R² = 0.2043
#   📌 중요 변수 Top 5:
#     - edu_yrs: 2558.000000
#     - age: 2399.000000
#     - female_age: 2102.000000
#     - risk_weighted_age: 2014.000000
#     - male_age: 1449.000000
#   🔧 Params: {'name': 'lgbm2', 'learning_rate': 0.03, 'num_leaves': 50, 'max_depth': 6, 'reg_alpha': 
# 0.3, 'reg_lambda': 1}
# --------------------------------------------------
# 🔹 모델: lgbm1
#   - Train R² = 0.7093
#   - Test  R² = 0.1831
#   📌 중요 변수 Top 5:
#     - edu_yrs: 2010.000000
#     - age: 1791.000000
#     - risk_weighted_age: 1639.000000
#     - female_age: 1628.000000
#     - male_age: 1274.000000
#   🔧 Params: {'name': 'lgbm1', 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 5, 'reg_alpha': 
# 0.1, 'reg_lambda': 1}
# --------------------------------------------------
# 🔹 모델: lgbm5
#   - Train R² = 0.7163
#   - Test  R² = 0.1675
#   📌 중요 변수 Top 5:
#     - edu_yrs: 1993.000000
#     - age: 1867.000000
#     - risk_weighted_age: 1679.000000
#     - female_age: 1573.000000
#     - male_age: 1334.000000
#   🔧 Params: {'name': 'lgbm5', 'learning_rate': 0.05, 'num_leaves': 40, 'max_depth': 5, 'reg_alpha': 
# 0, 'reg_lambda': 0}