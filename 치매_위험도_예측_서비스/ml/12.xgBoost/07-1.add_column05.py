# 파생변수 추가해보기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ✅ 데이터 로딩 및 기본 파생변수 생성
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv")
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["edu_level_bucket"] = df["edu_level"] // 2
df["edu_level_squared"] = df["edu_level"] ** 2  # 만약 포함하는 구조일 경우

# ✅ 고정 피처셋 제외 목록
exclude_cols = ["years_until_ad", "ad_year", "hhid_year"]

# ✅ 실험용 파생변수 정의
def apply_new_features(df):
    df = df.copy()
    df["is_old"] = (df["age"] >= 80).astype(int)
    df["edu_is_low"] = (df["edu_yrs"] < 6).astype(int)
    df["male_age"] = df["age"] * (df["gender"] == 0)
    df["female_age"] = df["age"] * (df["gender"] == 1)
    df["risk_factor_sum"] = df["has_db"] + df["has_hibpe"]
    df["edu_yrs_per_risk"] = df["edu_yrs"] / (1 + df["risk_factor_sum"])
    df["is_risk_and_old"] = ((df["risk_factor_sum"] >= 2) & (df["age"] >= 75)).astype(int)
    df["AD_MCI_status_onehot_1"] = (df["AD_MCI_status"] == 1).astype(int)
    df["age_group5_squared"] = df["age_group5"] ** 2
    df["edu_yrs_log"] = np.log(df["edu_yrs"] + 1)
    df["db_hibpe_interaction"] = df["has_db"] * df["has_hibpe"]
    return df

df = apply_new_features(df)

# ✅ 전체 파생변수 실험 후보
test_features = [
    "is_old", "edu_is_low", "male_age", "female_age",
    "edu_yrs_per_risk", "is_risk_and_old", "AD_MCI_status_onehot_1",
    "age_group5_squared", "edu_yrs_log", "db_hibpe_interaction"
]

# ✅ 고정 피처
base_features = [
    "age", "gender", "edu_yrs", "has_db", "AD_MCI_status",
    "has_hibpe", "edu_level", "age_x_edu",
    "hibpe_onset_delay_ratio", "edu_level_bucket"
]

# ✅ 결과 저장
results = []

# ✅ 반복 실험
for i, test_feature in enumerate(test_features, 1):
    use_features = base_features + [test_feature]
    df_model = df[use_features + ["years_until_ad"]].dropna()
    X = df_model.drop(columns=["years_until_ad"])
    y = df_model["years_until_ad"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ 추가: 결측값 및 inf 제거
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.dropna()
    X_test = X_test.dropna()
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]


    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=200,
        learning_rate=0.2,
        max_depth=6,
        subsample=1.0,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.0,
        min_child_weight=5,
        gamma=0.5,
        max_delta_step=0
    )

    model.fit(X_train, y_train)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    results.append({
        "experiment": f"{i}/10",
        "feature_added": test_feature,
        "train_r2": round(train_r2, 4),
        "test_r2": round(test_r2, 4),
        "delta_test_r2": round(test_r2 - 0.8356, 4)  # 기준 성능 대비 변화량
    })

    print(f"{i}/10")

# ✅ 결과 출력
results_df = pd.DataFrame(results).sort_values(by="test_r2", ascending=False)
print("\n✅ Test R² 기준 상위 순서:")
print(results_df.to_string(index=False))

# 결과
# ✅ Test R² 기준 상위 순서:
# experiment          feature_added  train_r2  test_r2  delta_test_r2
#       5/10       edu_yrs_per_risk    0.8980   0.4497        -0.3859
#       9/10            edu_yrs_log    0.8647   0.3036        -0.5320
#       4/10             female_age    0.8734   0.2979        -0.5377
#       7/10 AD_MCI_status_onehot_1    0.8571   0.2865        -0.5491
#       2/10             edu_is_low    0.8659   0.2814        -0.5542
#       6/10        is_risk_and_old    0.8659   0.2814        -0.5542
#      10/10   db_hibpe_interaction    0.8689   0.2809        -0.5547
#       3/10               male_age    0.8690   0.2808        -0.5548
#       8/10     age_group5_squared    0.8618   0.2591        -0.5765
#       1/10                 is_old    0.8681   0.2418        -0.5938