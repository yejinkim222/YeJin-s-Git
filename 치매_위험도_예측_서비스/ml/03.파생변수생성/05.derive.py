# 파생 변수 만들기
# risk_factor_sum: 각 질병이 있으면 1씩 더해서 질병이 여러개면 가중치.
# edu_is_low: 교육 수준이 낮은 경우만 마킹(~초졸).
# risk_weighted_age: 나이와 위험요소를 곱함.
# age_gender_interact: 남자, 여자 각각 나이와 곱한 변수 2개 생성.
import pandas as pd

# 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/02.AD_train_check_missing.csv"
df = pd.read_csv(file_path)

# 결측치 코드인 -1을 0으로 대체 (질병 없음을 의미)
df['has_db'] = df['has_db'].replace(-1, 0)
df['has_hibpe'] = df['has_hibpe'].replace(-1, 0)
df['AD_MCI_status'] = df['AD_MCI_status'].replace(-1, 0)

# ✅ 1. risk_factor_sum: 당뇨, 고혈압, MCI 여부 합산
df['risk_factor_sum'] = df['has_db'] + df['has_hibpe'] + df['AD_MCI_status']

# ✅ 2. edu_is_low: 교육 수준이 초졸 이하인 경우
df['edu_is_low'] = df['edu_level'].apply(lambda x: 1 if x != -1 and x <= 1 else 0)

# ✅ 3. risk_weighted_age: 나이 * risk_factor_sum
df['risk_weighted_age'] = df['age'] * df['risk_factor_sum']

# ✅ 4. age_gender_interact → 성별별 나이 변수 2개 생성
df['male_age'] = df.apply(lambda row: row['age'] if row['gender'] == 0 else 0, axis=1)
df['female_age'] = df.apply(lambda row: row['age'] if row['gender'] == 1 else 0, axis=1)

# 컬럼 삭제
df = df.drop(columns=['edu_level_missing', 'ad_year'])

# (선택) 저장하고 싶다면 아래 코드 사용
df.to_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/03.AD_train_derived.csv", index=False)
