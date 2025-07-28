# 확인용
# 행 수 세기
import pandas as pd
import os

# 예시 파일 경로 (이미 'r4'부터 'r16'까지 처리한 파일들이 존재)
file_paths = [
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r4_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r5_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r6_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r7_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r8_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r9_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r10_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r11_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r12_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r13_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r14_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r15_ad_mci.csv",
    "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci/rand_filtered_r16_ad_mci.csv",
]

# 각 파일에 대해 컬럼 수와 행 수 세기
total_rows = 0

for file_path in file_paths:
    # 데이터 로드
    rand = pd.read_csv(file_path)
    
    # 파일 이름에서 r4, r5, ..., r16 추출
    file_name = os.path.basename(file_path)
    
    # 컬럼 수와 행 수 출력
    num_rows = len(rand)

    total_rows += num_rows
    
    print(f"파일: {file_name}, 행 수: {num_rows}")

print(total_rows)
