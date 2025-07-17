import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")

# ✅ 기존 파생 변수
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["age_edu_ratio"] = df["age"] / (df["edu_yrs"] + 1)

# ✅ 제외 컬럼 정의 및 모델 데이터 구성
exclude_cols = ["years_until_ad", "ad_year", "hhid_year", "hhid", "ad_year_missing", "year", "year_missing"]
target = "years_until_ad"
df_model = df.drop(columns=exclude_cols).copy()
df_model[target] = df[target]

# ✅ 결측치, 무한값 처리
df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()

# ✅ X, y 분할
X_base = df_model.drop(columns=[target])
y = df_model[target]

# ✅ train-test 분할
X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)

# ✅ Optuna 최적화용 objective 함수 정의
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 480, 510),
        'max_depth': trial.suggest_int('max_depth', 11, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.145, 0.155),
        'subsample': trial.suggest_float('subsample', 0.89, 0.91),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.92, 0.94),
        'reg_alpha': trial.suggest_float('reg_alpha', 1.3, 1.6),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.03, 0.04),
        'min_child_weight': 1,
        'gamma': trial.suggest_float('gamma', 9.0, 9.3),
        'max_delta_step': 5,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    return score

# ✅ Optuna 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# ✅ 상위 5개 결과 출력
top5 = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
for i, t in enumerate(top5, 1):
    print(f"# {i:02d} ▶ Test R²: {t.value:.4f} | params: {t.params}")

# 결과
# 01 ▶ Test R²: 0.7392 | params: {'n_estimators': 496, 'max_depth': 12, 'learning_rate': 0.14823095794483634, 'subsample': 0.8915652093256143, 'colsample_bytree': 0.9265601946981163, 'reg_alpha': 1.5238151913604103, 'reg_lambda': 0.032926218816428134, 'gamma': 9.125702083638783}
# 02 ▶ Test R²: 0.7370 | params: {'n_estimators': 500, 'max_depth': 11, 'learning_rate': 0.15268689273038202, 'subsample': 0.89327619282152, 'colsample_bytree': 0.9399872864296246, 'reg_alpha': 1.5470404148715826, 'reg_lambda': 0.032616532391834555, 'gamma': 9.146800784734411}
# 03 ▶ Test R²: 0.7368 | params: {'n_estimators': 494, 'max_depth': 11, 'learning_rate': 0.15132220461921225, 'subsample': 0.8907205370094764, 'colsample_bytree': 0.9388701316309714, 'reg_alpha': 1.5972417404902488, 'reg_lambda': 0.03742790467963277, 'gamma': 9.002166694735466}
# 04 ▶ Test R²: 0.7349 | params: {'n_estimators': 506, 'max_depth': 12, 'learning_rate': 0.15490814653363383, 'subsample': 0.8965957843452422, 'colsample_bytree': 0.927316577276962, 'reg_alpha': 1.5035174073622097, 'reg_lambda': 0.036018987124342025, 'gamma': 9.29221047863914}
# 05 ▶ Test R²: 0.7347 | params: {'n_estimators': 480, 'max_depth': 12, 'learning_rate': 0.14526927048478402, 'subsample': 0.9056756731498169, 'colsample_bytree': 0.9352712835224747, 'reg_alpha': 1.3054556998629863, 'reg_lambda': 0.03381785715057148, 'gamma': 9.112074599223625}