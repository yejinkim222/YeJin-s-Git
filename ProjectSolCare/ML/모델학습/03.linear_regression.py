# linear regression 통해서 변수 간 관계 파악하기
# 여기서의 계수(기울기): weight
# 절댓값 크면 중요한 변수, 작으면 영향 작음
# 양수면 양의 상관관계, 음수면 음의 상관관계
from sklearn.linear_model import LinearRegression
import pandas as pd

input_path = "C:/workspace/Project01/data/hrs/selected_data/10.result/final_AD_train_long_최초시점_filled_derived.csv"
df = pd.read_csv(input_path)

# 변수 목록
features = [
    'age', 'gender', 'edu_yrs', 'has_db', 'AD_MCI_status', 'has_hibpe',
    'edu_level', 'db_onset_after', 'hibpe_onset_after', 'mci_onset_after',
    'age_group5', 'risk_factor_sum', 'edu_is_low', 'risk_weighted_age',
    'age_gender_interact', 'hibpe_onset_after_missing', 'has_hibpe_missing',
    'mci_onset_after_missing', 'edu_yrs_missing', 'db_onset_after_missing',
    'ad_year_missing', 'year_missing'
]

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
# hibpe_onset_after_missing     : 기울기 = -4.8796
# has_hibpe_missing             : 기울기 = -4.6696
# db_onset_after_missing        : 기울기 = -3.4322
# AD_MCI_status                 : 기울기 = -2.6685
# has_hibpe                     : 기울기 = 2.4315
# risk_factor_sum               : 기울기 = 1.3301
# mci_onset_after               : 기울기 = -0.7437
# mci_onset_after_missing       : 기울기 = 0.7437
# gender                        : 기울기 = 0.7425
# hibpe_onset_after             : 기울기 = 0.4958
# has_db                        : 기울기 = -0.4749
# db_onset_after                : 기울기 = 0.4632
# age_group5                    : 기울기 = -0.3452
# edu_level                     : 기울기 = -0.2166
# edu_is_low                    : 기울기 = 0.1333
# age                           : 기울기 = -0.0709
# edu_yrs                       : 기울기 = -0.0415
# risk_weighted_age             : 기울기 = 0.0173
# age_gender_interact           : 기울기 = 0.0077
# ad_year_missing               : 기울기 = 0.0000
# year_missing                  : 기울기 = 0.0000