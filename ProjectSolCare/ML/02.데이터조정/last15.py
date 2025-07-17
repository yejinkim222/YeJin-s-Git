# 치매 발생 예정인 사람의 데이터만 남기기
# long으로 전환 완료
# edu_level 추가 완료
import pandas as pd
import os

# 1. 파일 경로 리스트
file_paths = [
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r16_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r15_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r14_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r13_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r12_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r11_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r10_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r9_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r8_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r7_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r6_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r5_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r4_ad_mci.csv",
]

# 2. 모든 데이터 병합
df_list = [pd.read_csv(path) for path in file_paths]
all_data = pd.concat(df_list, ignore_index=True)

# 3. 치매가 발생한 사람(hhid)만 추출
hhid_with_ad = all_data[all_data['AD_MCI_status'] == 2]['hhid'].unique()
filtered_data = all_data[all_data['hhid'].isin(hhid_with_ad)].copy()

# 4. 각 hhid별 최초 치매 발생 연도(ad_year) 계산
ad_year_map = (
    filtered_data[filtered_data['AD_MCI_status'] == 2]
    .groupby('hhid')['year']
    .min()
    .rename('ad_year')
)

# 5. ad_year 병합
filtered_data = filtered_data.merge(ad_year_map, on='hhid', how='left')

# 6. 치매 발생 이전 시점만 유지 (발생 시점과 이후는 제거)
filtered_data = filtered_data[filtered_data['year'] < filtered_data['ad_year']].copy()

# 7. 회귀 타겟 생성: 치매까지 남은 년 수
filtered_data['years_until_ad'] = filtered_data['ad_year'] - filtered_data['year']

# 8. 결과 확인
print("최종 데이터 shape:", filtered_data.shape)
print("year 범위:", sorted(filtered_data['year'].unique()))
print("y값 예시 (years_until_ad):", filtered_data['years_until_ad'].describe())
print("포함된 치매 대상자 수:", filtered_data['hhid'].nunique())

# 9. edu_level 컬럼 추가
def categorize_edu(edu_yrs):
    if pd.isnull(edu_yrs):
        return -1  # 결측값 처리
    elif edu_yrs <= 5:
        return 0  # 기타/무학
    elif edu_yrs <= 8:
        return 1  # 초졸
    elif edu_yrs <= 11:
        return 2  # 중졸
    elif edu_yrs <= 13:
        return 3  # 고졸
    else:
        return 4  # 대졸 이상

filtered_data['edu_level'] = filtered_data['edu_yrs'].apply(categorize_edu).astype(int)

# 9. (선택) 저장
filtered_data.to_csv("C:/workspace/ProjectData/hrs/selected_data/10.result/final_AD_train_long.csv", index=False)
