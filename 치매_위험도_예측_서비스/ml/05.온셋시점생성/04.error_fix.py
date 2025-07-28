# 질병 제대로 표기되었는지 확인하고 
# 합산도 확인
# 오류 있으면 고치는 코드
import pandas as pd

# ✅ 1. onset 오류 수정 함수
def fix_onset_errors(df, disease_col, onset_col):
    """
    onset > 0인 경우, (year + onset)에 해당하는 row에서 disease_col을 1로 수정
    이미 1인 경우는 건너뜀
    """
    valid_df = df[(df[onset_col] > 0) & (df[onset_col] != -1)].copy()
    fix_count = 0

    for idx, row in valid_df.iterrows():
        hhid = row['hhid']
        target_year = row['year'] + row[onset_col]
        target_index = df[(df['hhid'] == hhid) & (df['year'] == target_year)].index

        if len(target_index) > 0 and df.loc[target_index, disease_col].iloc[0] != 1:
            df.loc[target_index, disease_col] = 1
            fix_count += 1

    print(f"🔧 {onset_col} → {disease_col}: 총 {fix_count}건 수정됨")
    return df

# ✅ 2. onset 오류 검사 함수
def verify_onset_logic(df, disease_col, onset_col):
    """
    onset > 0인 경우, (year + onset)의 row에서 disease_col == 1인지 확인
    오류만 집계함
    """
    errors = []
    valid_df = df[(df[onset_col] > 0) & (df[onset_col] != -1)]

    for idx, row in valid_df.iterrows():
        hhid = row['hhid']
        expected_year = row['year'] + row[onset_col]
        future_row = df[(df['hhid'] == hhid) & (df['year'] == expected_year)]

        if future_row.empty or future_row[disease_col].iloc[0] != 1:
            errors.append((hhid, row['year'], expected_year))

    print(f"\n📌 {onset_col} → {disease_col} 검사 결과:")
    print(f"❗ 오류 수: {len(errors)}")
    if errors:
        print("예: (hhid, 현재 year, 기대 발병 year):")
        for e in errors[:5]:  # 처음 5개만 예시로 출력
            print(f"  - hhid: {e[0]}, year: {e[1]} → expected: {e[2]}")
    return errors

# ✅ 3. 파일 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/05.AD_train_add_row.csv"
df = pd.read_csv(file_path)

# ✅ 4. onset 오류 확인 및 수정
for disease_col, onset_col in [
    ('has_db', 'years_until_db'),
    ('has_hibpe', 'years_until_hibpe'),
    ('AD_MCI_status', 'years_until_mci')
]:
    errors_before = verify_onset_logic(df, disease_col, onset_col)
    if errors_before:
        df = fix_onset_errors(df, disease_col, onset_col)
        verify_onset_logic(df, disease_col, onset_col)

# ✅ 5. risk_factor_sum 검사 및 수정
# -1은 결측으로 취급하여 0으로 변환 후 합산
df['risk_factor_sum_check'] = (
    df['has_db'].replace(-1, 0) +
    df['has_hibpe'].replace(-1, 0) +
    df['AD_MCI_status'].replace(-1, 0)
)

mismatch = df[df['risk_factor_sum'] != df['risk_factor_sum_check']]
print(f"\n📌 risk_factor_sum 불일치 개수: {len(mismatch)}")

if not mismatch.empty:
    df.loc[mismatch.index, 'risk_factor_sum'] = df.loc[mismatch.index, 'risk_factor_sum_check']
    print(f"🔧 risk_factor_sum 수정 완료 ({len(mismatch)}건)")

# ✅ 6. 정수 변환 (소수점 제거)
for col in ['year', 'age', 'risk_factor_sum']:
    if col in df.columns:
        df[col] = df[col].astype(int)

# ✅ 7. 임시 컬럼 제거
df.drop(columns=['risk_factor_sum_check'], inplace=True)

# ✅ 8. 저장
save_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/06.AD_train_error_fixed.csv"
df.to_csv(save_path, index=False)
print(f"\n✅ 전체 수정 완료: {save_path}")
