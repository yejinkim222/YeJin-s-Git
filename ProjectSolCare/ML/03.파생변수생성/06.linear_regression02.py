# 각 변수 linear regression하고
# baseline score 확인
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 데이터 로드
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/03.AD_train_derived.csv"
df = pd.read_csv(file_path)

# 타겟 변수
target = 'years_until_ad'

# 입력 피처 목록 (제외할 컬럼은 뺌)
exclude_cols = ['hhid', 'hhid_year', 'ad_year', 'year', target]
features = [col for col in df.columns if col not in exclude_cols]

# 리니어 리그레션 계수 분석
results = []
for col in features:
    X_col = df[[col]].values  # 2D array
    y = df[target].values
    model = LinearRegression()
    model.fit(X_col, y)
    coef = model.coef_[0]
    results.append((col, coef))

# 절댓값 기준 정렬 후 출력
results.sort(key=lambda x: abs(x[1]), reverse=True)
print("Linear Regression 계수 (기울기)")
for name, coef in results:
    print(f"{name:<30}: 기울기 = {coef:.4f}")

# 출력
# Linear Regression 계수 (기울기)
# edu_yrs_missing               : 기울기 = -6.2243
# has_hibpe                     : 기울기 = 4.7050
# has_hibpe_missing             : 기울기 = -4.6696
# years_until_db_missing        : 기울기 = -3.4322
# AD_MCI_status                 : 기울기 = -2.6685
# risk_factor_sum               : 기울기 = 0.9566
# edu_is_low                    : 기울기 = 0.7904
# years_until_mci_missing       : 기울기 = 0.7437
# gender                        : 기울기 = 0.7425
# has_db                        : 기울기 = -0.4749
# years_until_db                : 기울기 = 0.4632
# years_until_hibpe             : 기울기 = 0.4524
# age_group5                    : 기울기 = -0.3452
# edu_level                     : 기울기 = -0.2166
# age                           : 기울기 = -0.0709
# years_until_mci               : 기울기 = 0.0479
# edu_yrs                       : 기울기 = -0.0415
# male_age                      : 기울기 = -0.0144
# risk_weighted_age             : 기울기 = 0.0105
# female_age                    : 기울기 = 0.0077
# years_until_hibpe_missing     : 기울기 = 0.0053