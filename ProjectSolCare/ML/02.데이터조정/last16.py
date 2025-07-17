# 최초 관측 시점 남기면서 onset 정보 추가하기
import pandas as pd

# 파일 경로
input_path = "C:/workspace/ProjectData/hrs/selected_data/10.result/final_AD_train_long.csv"
output_path = "C:/workspace/ProjectData/hrs/selected_data/10.result/final_AD_train_long_최초시점.csv"

# 데이터 불러오기
df = pd.read_csv(input_path)

# year 기준 정렬
df = df.sort_values(by=['hhid', 'year'])

# 최초 관측 시점 추출
first_obs = df.groupby('hhid').first().reset_index()

# 치매 발생까지 걸리는 햇수 계산
first_obs['years_until_ad'] = first_obs['ad_year'] - first_obs['year']

# 0→1로 변한 시점 계산용 함수
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

# 각 변수의 최초 발생 시점 계산
db_onset = compute_onset_years(df, 'has_db')
hibpe_onset = compute_onset_years(df, 'has_hibpe')
mci_onset = compute_onset_years(df[df['AD_MCI_status'] == 1], 'AD_MCI_status')

# 최초 관측 시점 데이터에 onset 정보 추가
first_obs['db_onset_after'] = first_obs['hhid'].map(db_onset)
first_obs['hibpe_onset_after'] = first_obs['hhid'].map(hibpe_onset)
first_obs['mci_onset_after'] = first_obs['hhid'].map(mci_onset)

# 필요하면 year 컬럼 삭제
# first_obs = first_obs.drop(columns=['year', 'ad_year'])

# 저장
first_obs.to_csv(output_path, index=False)
print(f"저장 완료: {output_path}")
