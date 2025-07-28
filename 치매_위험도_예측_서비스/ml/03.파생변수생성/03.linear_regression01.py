# linear regression 통해서 변수 간 관계 파악하기
# 여기서의 계수(기울기): weight
# 절댓값 크면 중요한 변수, 작으면 영향 작음
# 양수면 양의 상관관계, 음수면 음의 상관관계
from sklearn.linear_model import LinearRegression
import pandas as pd

input_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/02.AD_train_check_missing.csv"
df = pd.read_csv(input_path)

# 변수 목록
features = [
    'hhid','year', 'age', 
    'hhid_year', 'gender', 'edu_yrs', 
    'has_db', 'AD_MCI_status', 'has_hibpe', 
    'ad_year', 'edu_level', 'years_until_mci', 
    'years_until_db', 'years_until_hibpe', 
    'years_until_hibpe_missing', 'has_hibpe_missing', 
    'years_until_mci_missing', 'years_until_db_missing', 
    'edu_yrs_missing', 'edu_level_missing', 'age_group5']

results = []

for col in features:
    X_col = df[[col]].values  # 2D array 형태로
    y_val = df["years_until_ad"].values
    lr = LinearRegression()
    lr.fit(X_col, y_val)
    coef = lr.coef_[0]
    results.append((col, coef))

# 절댓값 기준으로 정렬하여 출력
results.sort(key=lambda x: abs(x[1]), reverse=True)

# 출력
for name, coef in results:
    print(f"{name:<30}: 기울기 = {coef:.4f}")

# 결과
# edu_yrs_missing               : 기울기 = -6.2243
# edu_level_missing             : 기울기 = -6.2243
# has_hibpe_missing             : 기울기 = -4.6696
# years_until_db_missing        : 기울기 = -3.4322
# AD_MCI_status                 : 기울기 = -2.6685
# has_hibpe                     : 기울기 = 2.4315
# years_until_mci_missing       : 기울기 = 0.7437
# gender                        : 기울기 = 0.7425
# year                          : 기울기 = -0.6984
# ad_year                       : 기울기 = 0.5166
# has_db                        : 기울기 = -0.4749
# years_until_db                : 기울기 = 0.4632
# years_until_hibpe             : 기울기 = 0.4524
# age_group5                    : 기울기 = -0.3452
# edu_level                     : 기울기 = -0.2166
# age                           : 기울기 = -0.0709
# years_until_mci               : 기울기 = 0.0479
# edu_yrs                       : 기울기 = -0.0415
# years_until_hibpe_missing     : 기울기 = 0.0053
# hhid                          : 기울기 = -0.0000
# hhid_year                     : 기울기 = -0.0000