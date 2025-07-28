import pandas as pd

# 📂 파일 경로
input_path = "C:/workspace/ProjectData/hrs/selected_data/10.result/final_AD_train_long.csv"
output_path = "C:/workspace/ProjectData/hrs/selected_data/10.result/final_AD_train_long_치매발생자_filled.csv"

# 📌 데이터 불러오기
df = pd.read_csv(input_path)

# 📌 year 기준 정렬
df = df.sort_values(by=["hhid", "year"])

# 📌 치매 발생자만 추출 (years_until_ad 존재하는 경우)
ad_group = df[df["years_until_ad"].notna()]

# 📌 치매 발생 이전 데이터만 추출 (years_until_ad > 0)
before_ad_rows = df[df["hhid"].isin(ad_group["hhid"]) & (df["years_until_ad"] > 0)]

# 📌 onset 계산 함수
def compute_onset_years(df, col):
    onset_dict = {}
    for hhid, group in df.groupby('hhid'):
        group_sorted = group.sort_values('year')
        first_year = group_sorted['year'].iloc[0]
        onset_row = group_sorted[group_sorted[col] == 1]
        if not onset_row.empty:
            onset_year = onset_row['year'].iloc[0]
            onset_dict[hhid] = onset_year - first_year
        else:
            onset_dict[hhid] = None
    return onset_dict

# ✅ onset 시점 계산
db_onset = compute_onset_years(before_ad_rows, 'has_db')
hibpe_onset = compute_onset_years(before_ad_rows, 'has_hibpe')
mci_onset = compute_onset_years(before_ad_rows[before_ad_rows['AD_MCI_status'] == 1], 'AD_MCI_status')

# ✅ 최초 관측 시점 추출
first_obs = before_ad_rows.groupby("hhid").first().reset_index()

# ✅ onset 정보 추가
first_obs['db_onset_after'] = first_obs['hhid'].map(db_onset)
first_obs['hibpe_onset_after'] = first_obs['hhid'].map(hibpe_onset)
first_obs['mci_onset_after'] = first_obs['hhid'].map(mci_onset)

# ✅ 결측치 모두 -1로 채우기
first_obs.fillna(-1, inplace=True)

# ✅ 저장
first_obs.to_csv(output_path, index=False)
print(f"✅ 저장 완료: {output_path}")
