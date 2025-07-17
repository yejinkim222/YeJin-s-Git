import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_for_lgbm.csv")
target_col = "years_until_ad"
non_features = ["hhid", "hhid_year", "year", "ad_year", "ad_year_missing", "year_missing", "years_until_ad_ratio"]
df_model = df.drop(columns=[col for col in non_features if col in df.columns])

# ✅ 데이터 분할
X = df_model.drop(columns=["years_until_ad"])
y = df_model["years_until_ad"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# ✅ Optuna 목적 함수
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 450, 550),
        "max_depth": trial.suggest_int("max_depth", 11, 13),
        "learning_rate": trial.suggest_float("learning_rate", 0.13, 0.16),
        "subsample": trial.suggest_float("subsample", 0.91, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.91, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.015, 0.05),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 2),
        "gamma": trial.suggest_float("gamma", 8.5, 9.5),
        "max_delta_step": 5,
        "random_state": 42
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)

# ✅ Optuna 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# ✅ 상위 10개 모델 성능 출력
print("\n📈 Top 10 Trials by Test R²:")
top_trials = sorted(study.trials, key=lambda x: x.value, reverse=True)[:10]
for i, trial in enumerate(top_trials, 1):
    best_model = XGBRegressor(**trial.params)
    best_model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, best_model.predict(X_train))
    test_r2 = r2_score(y_test, best_model.predict(X_test))
    print(f"# {i:02d} ▶ Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | params: {trial.params}")

# 결과
# 📈 Top 10 Trials by Test R²:
# 01 ▶ Train R²: 0.9412 | Test R²: 0.6567 | params: {'n_estimators': 469, 'max_depth': 12, 'learning_rate': 0.14768356727751106, 'subsample': 0.9348039742075571, 'colsample_bytree': 0.935745821168931, 'reg_alpha': 1.2826648963433511, 'reg_lambda': 0.025145433591828213, 'min_child_weight': 1, 'gamma': 8.619987738200251}
# 02 ▶ Train R²: 0.9371 | Test R²: 0.6475 | params: {'n_estimators': 545, 'max_depth': 13, 'learning_rate': 0.13440812045416534, 'subsample': 0.9152513223090047, 'colsample_bytree': 0.9128024453773628, 'reg_alpha': 1.5331681093064915, 'reg_lambda': 0.03509209198735791, 'min_child_weight': 1, 'gamma': 9.093233545001349}
# 03 ▶ Train R²: 0.9376 | Test R²: 0.6362 | params: {'n_estimators': 459, 'max_depth': 11, 'learning_rate': 0.14802918439346263, 'subsample': 0.917703092177863, 'colsample_bytree': 0.9427771370353047, 'reg_alpha': 1.2702446944131096, 'reg_lambda': 0.03540775912392529, 'min_child_weight': 1, 'gamma': 9.294879176250458}
# 04 ▶ Train R²: 0.9387 | Test R²: 0.6434 | params: {'n_estimators': 479, 'max_depth': 12, 'learning_rate': 0.147880403180401, 'subsample': 0.9352110889839303, 'colsample_bytree': 0.933705050313638, 'reg_alpha': 1.2354673555167806, 'reg_lambda': 0.021433281522923783, 'min_child_weight': 1, 'gamma': 8.868834109849708}
# 05 ▶ Train R²: 0.9406 | Test R²: 0.6389 | params: {'n_estimators': 478, 'max_depth': 12, 'learning_rate': 0.1480563708660293, 'subsample': 0.9345821744341565, 'colsample_bytree': 0.9420175537287632, 'reg_alpha': 1.340052254026049, 'reg_lambda': 0.022150431894959206, 'min_child_weight': 1, 'gamma': 8.737865207552485}
# 06 ▶ Train R²: 0.9365 | Test R²: 0.6366 | params: {'n_estimators': 478, 'max_depth': 12, 'learning_rate': 0.15079195837216816, 'subsample': 0.9384692980832983, 'colsample_bytree': 0.920229035483863, 'reg_alpha': 1.5263511923763529, 'reg_lambda': 0.01900570140299104, 'min_child_weight': 1, 'gamma': 9.187040602692154}
# 07 ▶ Train R²: 0.9386 | Test R²: 0.6521 | params: {'n_estimators': 478, 'max_depth': 12, 'learning_rate': 0.1479079731237662, 'subsample': 0.9346459290749458, 'colsample_bytree': 0.9411551783017821, 'reg_alpha': 1.3860179382811109, 'reg_lambda': 0.02235695958189395, 'min_child_weight': 1, 'gamma': 8.724013045092367}
# 08 ▶ Train R²: 0.9395 | Test R²: 0.6257 | params: {'n_estimators': 471, 'max_depth': 11, 'learning_rate': 0.15102156928205773, 'subsample': 0.9127014656077445, 'colsample_bytree': 0.9116932161869391, 'reg_alpha': 1.1433064772111805, 'reg_lambda': 0.038961343232969875, 'min_child_weight': 1, 'gamma': 9.233560859657079}
# 09 ▶ Train R²: 0.9388 | Test R²: 0.6346 | params: {'n_estimators': 502, 'max_depth': 11, 'learning_rate': 0.14738556947234924, 'subsample': 0.9179311889712984, 'colsample_bytree': 0.9227489530970648, 'reg_alpha': 1.1151341116796583, 'reg_lambda': 0.04059820219650996, 'min_child_weight': 1, 'gamma': 9.127883220034642}
# 10 ▶ Train R²: 0.9354 | Test R²: 0.6492 | params: {'n_estimators': 492, 'max_depth': 13, 'learning_rate': 0.1483058721180753, 'subsample': 0.9475951645015661, 'colsample_bytree': 0.9207479921017364, 'reg_alpha': 1.7146719165629944, 'reg_lambda': 0.017130506129480503, 'min_child_weight': 1, 'gamma': 8.94292276714065}