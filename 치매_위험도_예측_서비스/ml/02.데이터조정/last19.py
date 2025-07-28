# 파생 변수 및 결측 여부 컬럼 추가
import pandas as pd

# 파일 경로
input_path = "C:/workspace/ProjectData/hrs/selected_data/10.result/final_AD_train_long_최초시점_filled.csv"
output_path = "C:/workspace/ProjectData/hrs/selected_data/10.result/final_AD_train_long_최초시점_filled_derived.csv"

# 데이터 불러오기
df = pd.read_csv(input_path)

# 1. 나이 5살 단위로 그룹핑 (예: 65세 → 그룹 13)
df['age_group5'] = pd.cut(df['age'], bins=range(0, 121, 5), right=False, labels=False)

# 2. 위험요인 합산: has_db + has_hibpe (결측값은 0으로 간주)
df['risk_factor_sum'] = df[['has_db', 'has_hibpe']].replace(-1, 0).sum(axis=1)

# 3. 교육 수준 낮은 사람 여부: edu_level == 0
df['edu_is_low'] = (df['edu_level'] == 0).astype(int)

# 4. 고위험자 고령일 때 영향 강조: (has_db + has_hibpe) × age
df['risk_weighted_age'] = df[['has_db', 'has_hibpe']].replace(-1, 0).sum(axis=1) * df['age']

# 5. 성별에 따른 나이 효과: age × gender
df['age_gender_interact'] = df['age'] * df['gender']

# ✅ 6. 결측 여부 파생 변수 추가 (-1이면 결측으로 간주)
df['hibpe_onset_after_missing'] = (df['hibpe_onset_after'] == -1).astype(int)
df['has_hibpe_missing'] = (df['has_hibpe'] == -1).astype(int)
df['mci_onset_after_missing'] = (df['mci_onset_after'] == -1).astype(int)

# 7. 누락 건 결측 여부 변수 추가
for col in ['edu_yrs', 'db_onset_after', 'ad_year', 'year']:
    missing_col = f"{col}_missing"
    if missing_col not in df.columns:
        df[missing_col] = (df[col] == -1).astype(int)

# 저장
df.to_csv(output_path, index=False)
print(f"✅ 파생 변수 + 결측 여부 컬럼 추가 및 저장 완료: {output_path}")
