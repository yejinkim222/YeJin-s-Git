# 확인용
import pandas as pd
import os

# 예시 파일 경로 (r4부터 r16까지 순차적으로 파일 목록을 나열)
file_paths = [
    # "C:/workspace/ProjectData/hrs/selected_data/r4-r16_ad_mci/rand_filtered_r4_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_ad_mci/rand_filtered_r5_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_ad_mci/rand_filtered_r6_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_ad_mci/rand_filtered_r7_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_ad_mci/rand_filtered_r8_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_ad_mci/rand_filtered_r9_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_ad_mci/rand_filtered_r10_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_ad_mci/rand_filtered_r11_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_ad_mci/rand_filtered_r12_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_ad_mci/rand_filtered_r13_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_ad_mci/rand_filtered_r14_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_ad_mci/rand_filtered_r15_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_ad_mci/rand_filtered_r16_ad_mci.csv",
]

# 결과를 저장할 폴더 설정
output_folder = "C:/workspace/ProjectData/hrs/result"

# 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 각 파일 처리
for file_path in file_paths:
    # 데이터 로드
    rand = pd.read_csv(file_path)
    
    # 1. hhid_year 컬럼에서 6자리 숫자만 추출하여 hhid 컬럼으로 추가
    rand['hhid'] = rand['hhid_year'].apply(lambda x: str(x).split('_')[0])  # "_" 앞의 6자리 숫자 추출
    
    # 2. 결과를 새로운 파일로 저장 (파일명에 '_hhid' 추가)
    file_name = os.path.basename(file_path)  # 원본 파일명 추출
    output_file = os.path.join(output_folder, file_name.replace(".csv", "_hhid.csv"))
    
    rand.to_csv(output_file, index=False)
    print(f"파일 처리 완료, 저장됨: {output_file}")
