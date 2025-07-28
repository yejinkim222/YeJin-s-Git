import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(file_path)

# ✅ 파생변수 생성
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["edu_level_bucket"] = df["edu_level"] // 2

# ✅ 피처/타겟 정의
target_col = "years_until_ad"
excluded_cols = ["years_until_ad", "ad_year", "hhid_year", "hhid", "ad_year_missing", "year", "year_missing", "years_until_ad_ratio"]
X = df.drop(columns=[col for col in excluded_cols if col in df.columns])
y = df[target_col]

# ✅ 결측치 제거
df_model = pd.concat([X, y], axis=1).dropna()
X = df_model.drop(columns=[target_col])
y = df_model[target_col]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Optuna 목적 함수
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 300, 600),
        "max_depth": trial.suggest_int("max_depth", 11, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.12, 0.20),
        "subsample": trial.suggest_float("subsample", 0.85, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.83, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 0.1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 3),
        "gamma": trial.suggest_float("gamma", 3.0, 10.0),
        "max_delta_step": trial.suggest_int("max_delta_step", 4, 7)
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)

# ✅ Optuna 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# ✅ 상위 10개 결과
top_trials = sorted(study.trials, key=lambda x: x.value, reverse=True)[:10]
for i, trial in enumerate(top_trials, 1):
    model = XGBRegressor(**trial.params)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    print(f"# {i:02d} ▶ Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | params: {trial.params}")

# 결과
# 01 ▶ Train R²: 0.9488 | Test R²: 0.6502 | params: {'n_estimators': 520, 'max_depth': 13, 'learning_rate': 0.13040329858012853, 'subsample': 0.9097324073341722, 'colsample_bytree': 0.9324917646764559, 'reg_alpha': 1.8489712380087058, 'reg_lambda': 0.0038474455009845643, 'min_child_weight': 1, 'gamma': 9.133025792648995, 'max_delta_step': 5}
# 02 ▶ Train R²: 0.9482 | Test R²: 0.6663 | params: {'n_estimators': 536, 'max_depth': 13, 'learning_rate': 0.129711539508846, 'subsample': 0.9050039729651138, 'colsample_bytree': 0.9179295168406376, 'reg_alpha': 1.3962639957517848, 'reg_lambda': 0.01808358868761678, 'min_child_weight': 1, 'gamma': 9.037313088446329, 'max_delta_step': 5}
# 03 ▶ Train R²: 0.9458 | Test R²: 0.6003 | params: {'n_estimators': 505, 'max_depth': 14, 'learning_rate': 0.15142662219118247, 'subsample': 0.8835571148830381, 'colsample_bytree': 0.8787334821788695, 'reg_alpha': 1.829545692692328, 'reg_lambda': 0.05516004108492921, 'min_child_weight': 1, 'gamma': 9.930803488706534, 'max_delta_step': 6}
# 04 ▶ Train R²: 0.9475 | Test R²: 0.6620 | params: {'n_estimators': 522, 'max_depth': 13, 'learning_rate': 0.12937692783665491, 'subsample': 0.8965204490149131, 'colsample_bytree': 0.9253782675588849, 'reg_alpha': 1.8544861764495315, 'reg_lambda': 0.019770113125567854, 'min_child_weight': 1, 'gamma': 9.367567920083557, 'max_delta_step': 5}
# 05 ▶ Train R²: 0.9503 | Test R²: 0.6542 | params: {'n_estimators': 533, 'max_depth': 13, 'learning_rate': 0.12836838302168915, 'subsample': 0.9032102816944186, 'colsample_bytree': 0.9233487653953351, 'reg_alpha': 1.0368840579325616, 'reg_lambda': 0.01653171811822606, 'min_child_weight': 1, 'gamma': 9.188503687749916, 'max_delta_step': 5}
# 06 ▶ Train R²: 0.9480 | Test R²: 0.6612 | params: {'n_estimators': 565, 'max_depth': 13, 'learning_rate': 0.1291008047332529, 'subsample': 0.9054440233284019, 'colsample_bytree': 0.9415374609289324, 'reg_alpha': 1.8783390145389622, 'reg_lambda': 0.017062629689209158, 'min_child_weight': 1, 'gamma': 8.640582457754252, 'max_delta_step': 5}
# 07 ▶ Train R²: 0.9474 | Test R²: 0.6601 | params: {'n_estimators': 523, 'max_depth': 13, 'learning_rate': 0.13057257603740474, 'subsample': 0.8974857297563981, 'colsample_bytree': 0.9220962782848257, 'reg_alpha': 1.389116822455388, 'reg_lambda': 0.026072280700508207, 'min_child_weight': 1, 'gamma': 9.770747664858808, 'max_delta_step': 5}
# 08 ▶ Train R²: 0.9595 | Test R²: 0.6377 | params: {'n_estimators': 429, 'max_depth': 13, 'learning_rate': 0.12477578483981902, 'subsample': 0.9555846270502704, 'colsample_bytree': 0.9044239277045741, 'reg_alpha': 1.0440317116031523, 'reg_lambda': 0.04521323570615277, 'min_child_weight': 2, 'gamma': 4.690901808611965, 'max_delta_step': 5}
# 09 ▶ Train R²: 0.9483 | Test R²: 0.6378 | params: {'n_estimators': 544, 'max_depth': 13, 'learning_rate': 0.1349151389912429, 'subsample': 0.8824910130773391, 'colsample_bytree': 0.9274591767542807, 'reg_alpha': 1.8278001451445425, 'reg_lambda': 0.008244850798983405, 'min_child_weight': 1, 'gamma': 9.387263838766424, 'max_delta_step': 5}
# 10 ▶ Train R²: 0.9458 | Test R²: 0.6459 | params: {'n_estimators': 475, 'max_depth': 13, 'learning_rate': 0.14169502256509267, 'subsample': 0.87515560704451, 'colsample_bytree': 0.9660691075792218, 'reg_alpha': 1.796651183547369, 'reg_lambda': 0.00447174131631875, 'min_child_weight': 1, 'gamma': 9.975970334886991, 'max_delta_step': 5}