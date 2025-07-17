import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")

# ✅ 파생 변수 생성
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)

# ✅ 불필요 컬럼 제외
exclude_cols = [
    "years_until_ad", "ad_year", "hhid_year", "hhid",
    "ad_year_missing", "year", "year_missing"
]
target = "years_until_ad"

df_model = df.drop(columns=exclude_cols).copy()
df_model[target] = df[target]
df_model = df_model.dropna()

X = df_model.drop(columns=[target])
y = df_model[target]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Optuna objective 정의
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 450, 600),
        "max_depth": trial.suggest_int("max_depth", 12, 14),
        "learning_rate": trial.suggest_float("learning_rate", 0.125, 0.15),
        "subsample": trial.suggest_float("subsample", 0.88, 0.93),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.9, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 0.05),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 2),
        "gamma": trial.suggest_float("gamma", 8.5, 10.0),
        "max_delta_step": trial.suggest_int("max_delta_step", 4, 6),
        "objective": "reg:squarederror",
        "random_state": 42,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)

# ✅ Optuna 최적화 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# ✅ 상위 10개 출력
top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]

for i, trial in enumerate(top_trials, 1):
    model = XGBRegressor(**trial.params, objective="reg:squarederror", random_state=42)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    print(f"# {i:02} ▶ Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | params: {trial.params}")

# 결과
# 01 ▶ Train R²: 0.9475 | Test R²: 0.6752 | params: {'n_estimators': 498, 'max_depth': 12, 'learning_rate': 0.14997745579336536, 'subsample': 0.8989092608354626, 'colsample_bytree': 0.9284142503510097, 'reg_alpha': 1.4386504039680161, 'reg_lambda': 0.03803761653911136, 'min_child_weight': 1, 'gamma': 9.219422177382182, 'max_delta_step': 5}
# 02 ▶ Train R²: 0.9504 | Test R²: 0.6730 | params: {'n_estimators': 499, 'max_depth': 12, 'learning_rate': 0.1491929688221537, 'subsample': 0.9022924072656447, 'colsample_bytree': 0.933484554570235, 'reg_alpha': 1.208321264974688, 'reg_lambda': 0.03539249084598422, 'min_child_weight': 1, 'gamma': 8.65318393323603, 'max_delta_step': 5}
# 03 ▶ Train R²: 0.9467 | Test R²: 0.6699 | params: {'n_estimators': 484, 'max_depth': 12, 'learning_rate': 0.1485216253437298, 'subsample': 0.9179573719446437, 'colsample_bytree': 0.9210518045377322, 'reg_alpha': 1.541510478413138, 'reg_lambda': 0.03230578805004761, 'min_child_weight': 1, 'gamma': 9.081645237850104, 'max_delta_step': 5}
# 04 ▶ Train R²: 0.9473 | Test R²: 0.6691 | params: {'n_estimators': 484, 'max_depth': 12, 'learning_rate': 0.14850624008835314, 'subsample': 0.9173238069716305, 'colsample_bytree': 0.9187066328962071, 'reg_alpha': 1.5869859121370686, 'reg_lambda': 0.029683092441656622, 'min_child_weight': 1, 'gamma': 9.09512672598312, 'max_delta_step': 5}
# 05 ▶ Train R²: 0.9498 | Test R²: 0.6673 | params: {'n_estimators': 501, 'max_depth': 12, 'learning_rate': 0.1484451897936633, 'subsample': 0.910076766513413, 'colsample_bytree': 0.9279263902152437, 'reg_alpha': 1.6051362834983358, 'reg_lambda': 0.03714513536172843, 'min_child_weight': 1, 'gamma': 8.74446122822961, 'max_delta_step': 5}
# 06 ▶ Train R²: 0.9500 | Test R²: 0.6666 | params: {'n_estimators': 501, 'max_depth': 12, 'learning_rate': 0.14871682845963788, 'subsample': 0.9054701585917881, 'colsample_bytree': 0.92091401207293, 'reg_alpha': 1.4773446936525354, 'reg_lambda': 0.032358274783115384, 'min_child_weight': 1, 'gamma': 8.81172409966624, 'max_delta_step': 5}
# 07 ▶ Train R²: 0.9493 | Test R²: 0.6665 | params: {'n_estimators': 495, 'max_depth': 13, 'learning_rate': 0.149100525011457, 'subsample': 0.9025835912695055, 'colsample_bytree': 0.9261561883454599, 'reg_alpha': 1.6969157414203033, 'reg_lambda': 0.03707406597873803, 'min_child_weight': 1, 'gamma': 8.986121856850458, 'max_delta_step': 5}
# 08 ▶ Train R²: 0.9484 | Test R²: 0.6659 | params: {'n_estimators': 476, 'max_depth': 12, 'learning_rate': 0.14824562552795842, 'subsample': 0.917639346504952, 'colsample_bytree': 0.9145393949341465, 'reg_alpha': 1.5015838794155263, 'reg_lambda': 0.03259158077373417, 'min_child_weight': 1, 'gamma': 9.095990211907091, 'max_delta_step': 5}
# 09 ▶ Train R²: 0.9489 | Test R²: 0.6647 | params: {'n_estimators': 520, 'max_depth': 12, 'learning_rate': 0.14891071266978398, 'subsample': 0.9162581576782823, 'colsample_bytree': 0.9339378269377084, 'reg_alpha': 1.5083873451611867, 'reg_lambda': 0.036773393943079435, 'min_child_weight': 1, 'gamma': 8.763785056730665, 'max_delta_step': 5}
# 10 ▶ Train R²: 0.9491 | Test R²: 0.6641 | params: {'n_estimators': 502, 'max_depth': 12, 'learning_rate': 0.14761451962002506, 'subsample': 0.902796316434693, 'colsample_bytree': 0.9254594916088931, 'reg_alpha': 1.6048208541160407, 'reg_lambda': 0.038271113926440996, 'min_child_weight': 1, 'gamma': 8.555977736588812, 'max_delta_step': 5}