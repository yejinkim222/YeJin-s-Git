import pandas as pd

# ✅ 1. 원본 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/03.AD_train_derived.csv"
df = pd.read_csv(file_path)

# ✅ 💡 여기 추가!
df.loc[df['has_hibpe_missing'] == 1, 'has_hibpe'] = -1

# ✅ 2. 설정: onset 관련 컬럼들과 상태 컬럼 매핑
onset_cols = ['years_until_db', 'years_until_hibpe', 'years_until_mci']
has_cols = ['has_db', 'has_hibpe', 'AD_MCI_status']
has_map = dict(zip(onset_cols, has_cols))

# ✅ 3. 기본 컬럼 & 유지할 나머지 컬럼 정의
base_cols = [
    'hhid', 'year', 'age', 'gender', 'edu_yrs', 'edu_level',
    'has_db', 'has_hibpe', 'AD_MCI_status', 'risk_factor_sum',
    'edu_is_low', 'risk_weighted_age', 'male_age', 'female_age',
    'hhid_year', 'age_group5'
]
# 남은 컬럼 자동 추출
remaining_cols = [col for col in df.columns if col not in onset_cols + ['risk_factor_sum', 'risk_weighted_age', 'age_group5']]
final_cols = base_cols + onset_cols + [col for col in remaining_cols if col not in base_cols + onset_cols]

# ✅ 4. 결과 저장용 리스트 생성
new_rows = []

# ✅ 5. 각 응답자별로 처리
for _, row in df.iterrows():
    new_rows.append(row.to_dict())  # 원래 row 추가

    # 양수 onset 값만 필터링
    onset_valid = [(col, int(row[col])) for col in onset_cols if row[col] > 0]
    if not onset_valid:
        continue

    # 발생 시점 기준으로 정렬 (가장 먼저 생긴 질병부터)
    onset_valid.sort(key=lambda x: x[1])

    # 상태 누적 변수 초기화
    has_status = {
        'has_db': row['has_db'],
        'has_hibpe': row['has_hibpe'],
        'AD_MCI_status': row['AD_MCI_status']
    }
    risk_factor_sum = row['risk_factor_sum']
    onset_original = {col: int(row[col]) for col in onset_cols}

    # ✅ 6. 질병 발생 시점별로 row 생성
    for onset_col, onset_val in onset_valid:
        new_row = row.copy()

        # year, age 업데이트 (치매 기준 year/age에 onset 값 더하기)
        new_row['year'] = int(row['year']) + onset_val
        new_row['age'] = int(row['age']) + onset_val
        new_row['hhid_year'] = f"{str(new_row['hhid']).zfill(6)}_{int(new_row['year'])}"

        # onset 값 조정
        for col in onset_cols:
            if col == onset_col:
                new_row[col] = 0
            elif onset_original[col] > onset_val:
                new_row[col] = onset_original[col] - onset_val
            else:
                new_row[col] = -1

        # 질병 상태 누적 반영
        has_key = has_map[onset_col]
        has_status[has_key] = 1
        for key in has_status:
            new_row[key] = has_status[key]

        # MCI인 경우 AD_MCI_status = 1
        if onset_col == 'years_until_mci':
            new_row['AD_MCI_status'] = 1

        # risk_factor_sum 업데이트
        risk_factor_sum += 1
        new_row['risk_factor_sum'] = risk_factor_sum

        # risk_weighted_age 계산
        new_row['risk_weighted_age'] = new_row['age'] * risk_factor_sum

        # age_group5 갱신
        new_row['age_group5'] = new_row['age'] // 5

        # 필요한 컬럼만 저장
        new_rows.append(new_row[final_cols].to_dict())

# ✅ 7. 최종 데이터프레임 생성
df_expanded = pd.DataFrame(new_rows)

# ✅ 8. 저장 (선택)
save_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/05.AD_train_add_row.csv"
df_expanded.to_csv(save_path, index=False)

print(f"✅ 저장 완료! 총 행 수: {len(df_expanded)}")
