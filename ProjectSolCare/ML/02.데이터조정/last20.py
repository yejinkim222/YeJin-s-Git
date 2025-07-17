# onset 계산 mci에서 잘못 했던 거 깨달아서
# 급하게 다시 만들기...
import pandas as pd

# 파일 경로 (예진님이 직접 수정)
file_path = "C:/workspace/ProjectData/hrs/selected_data/10.result/final_AD_train_long.csv"
df = pd.read_csv(file_path)

# 1. 기본 정렬
df.sort_values(by=['hhid', 'year'], inplace=True)

# 2. onset 계산 함수 정의
def compute_onset(df, status_col, new_col):
    df = df.copy()
    df['prev'] = df.groupby('hhid')[status_col].shift(1, fill_value=0)
    df['onset_flag'] = (df['prev'] == 0) & (df[status_col] == 1)
    
    # onset year 구하기
    onset_years = df[df['onset_flag']].groupby('hhid')['year'].first()
    
    # 각 row에서 onset_year - 현재 row year 계산
    df[new_col] = df['hhid'].map(onset_years) - df['year']
    df.drop(columns=['prev', 'onset_flag'], inplace=True)
    return df

# 3. onset 계산 적용
df = compute_onset(df, 'AD_MCI_status', 'years_until_mci')
df = compute_onset(df, 'has_db', 'years_until_db')
df = compute_onset(df, 'has_hibpe', 'years_until_hibpe')

# 4. hhid별 최초 관측 시점만 남기기
df_first = df.sort_values(by=['hhid', 'year']).groupby('hhid').first().reset_index()

# 5. 저장 (선택)
df_first.to_csv("C:/workspace/ProjectData/hrs/selected_data/10.result/final_AD_train_long_최초시점_onset.csv", index=False)

print("처리 완료! 결과 shape:", df_first.shape)
print(df_first[['hhid', 'year', 'years_until_mci', 'years_until_db', 'years_until_hibpe']].head())

