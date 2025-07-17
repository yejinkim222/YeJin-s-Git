# risk_factor_sum -> risk_weighted_age(로그변환)
# high_risk_group: 고졸 이하 & 70세 이상 조합
import pandas as pd
import numpy as np

# ✅ 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/06.AD_train_error_fixed.csv"
df = pd.read_csv(file_path)

# ✅ 로그변환된 risk_weighted_age
df['log_risk_weighted_age'] = df['risk_weighted_age'].apply(lambda x: np.log1p(x) if x > 0 else 0)

# ✅ 교육 수준 + 나이 그룹 조합 변수 생성
df['is_low_edu'] = (df['edu_level'] <= 1).astype(int)  # 고졸 이하
df['is_old'] = (df['age'] >= 70).astype(int)           # 70세 이상
df['high_risk_group'] = (df['is_low_edu'] & df['is_old']).astype(int)

# ✅ 확인용 출력
print(df[['age', 'edu_level', 'risk_factor_sum', 'risk_weighted_age', 'log_risk_weighted_age',
          'is_low_edu', 'is_old', 'high_risk_group']].head())

# ✅ 저장
save_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df.to_csv(save_path, index=False)
print(f"\n✅ 저장 완료: {save_path}")
