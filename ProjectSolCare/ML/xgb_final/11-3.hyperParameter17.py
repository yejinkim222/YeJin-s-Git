# ✅ 필요 라이브러리 설치 필요: pip install optuna xgboost scikit-learn pandas
import optuna
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(file_path)

# ✅ 파생변수 생성
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["edu_level_bucket"] = df["edu_level"] // 2

# ✅ 제외할 컬럼 정의
excluded_cols = [
    "years_until_ad", "ad_year", "hhid_year", "hhid",
    "ad_year_missing", "year", "year_missing",
    "years_until_ad_ratio"  # ❌ 예진님이 제거하신 변수
]
target_col = "years_until_ad"

# ✅ 학습용 데이터 구성
X = df.drop(columns=[col for col in excluded_cols if col in df.columns])
y = df[target_col]
df_model = pd.concat([X, y], axis=1).dropna()
X = df_model.drop(columns=[target_col])
y = df_model[target_col]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Optuna 목적 함수 정의
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 5, 11),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.15),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 3)
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)

# ✅ Optuna 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# ✅ 상위 10개 결과 확인
top_trials = sorted(study.trials, key=lambda x: x.value, reverse=True)[:5]
for i, trial in enumerate(top_trials, 1):
    model = XGBRegressor(**trial.params)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    print(f"# {i:02d} ▶ Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | params: {trial.params}")

# 결과
# 01 ▶ Train R²: 0.9648 | Test R²: 0.6229 | params: {'n_estimators': 420, 'max_depth': 10, 'learning_rate': 0.10802504819938909, 'subsample': 0.7766483009978119, 'colsample_bytree': 0.7010158501051186, 'reg_alpha': 0.8559584256631595, 'reg_lambda': 0.40739419357465245, 'min_child_weight': 3, 'gamma': 2.650656148431976, 'max_delta_step': 3}
# 02 ▶ Train R²: 0.9672 | Test R²: 0.6233 | params: {'n_estimators': 560, 'max_depth': 8, 'learning_rate': 0.08973209727273783, 'subsample': 0.8725271171828588, 'colsample_bytree': 0.6254530674584127, 'reg_alpha': 0.3377211024401068, 'reg_lambda': 0.3126429968598996, 'min_child_weight': 4, 'gamma': 1.4550647679462623, 'max_delta_step': 3}
# 03 ▶ Train R²: 0.9652 | Test R²: 0.6134 | params: {'n_estimators': 514, 'max_depth': 9, 'learning_rate': 0.08903494516947738, 'subsample': 0.8405202505959546, 'colsample_bytree': 0.6681871430419861, 'reg_alpha': 0.3446785213697191, 'reg_lambda': 0.34490228285196717, 'min_child_weight': 5, 'gamma': 1.9387398016606676, 'max_delta_step': 3}
# 04 ▶ Train R²: 0.9603 | Test R²: 0.6153 | params: {'n_estimators': 433, 'max_depth': 6, 'learning_rate': 0.13989502000240683, 'subsample': 0.7821094734904211, 'colsample_bytree': 0.7892589322925698, 'reg_alpha': 0.3233056879603493, 'reg_lambda': 0.7535013960484233, 'min_child_weight': 3, 'gamma': 2.254008317116174, 'max_delta_step': 1}
# 05 ▶ Train R²: 0.9655 | Test R²: 0.6387 | params: {'n_estimators': 418, 'max_depth': 10, 'learning_rate': 0.11155686732122766, 'subsample': 0.8307166901940066, 'colsample_bytree': 0.7258358746910889, 'reg_alpha': 0.563191452681913, 'reg_lambda': 0.3983428671403096, 'min_child_weight': 3, 'gamma': 2.6794084993311778, 'max_delta_step': 3}