# year 컬럼 추가하기
import pandas as pd
import os
import re  # 정규 표현식 추가

# 1. 파일 경로 설정 (r4부터 r16까지의 파일 경로)
file_paths = [
    "C:/workspace/ProjectData/hrs/selected_data/03.r4-r16_exclude/rand_filtered_r4_결측제거.csv",
    "C:/workspace/ProjectData/hrs/selected_data/03.r4-r16_exclude/rand_filtered_r5_결측제거.csv",
    "C:/workspace/ProjectData/hrs/selected_data/03.r4-r16_exclude/rand_filtered_r6_결측제거.csv",
    "C:/workspace/ProjectData/hrs/selected_data/03.r4-r16_exclude/rand_filtered_r7_결측제거.csv",
    "C:/workspace/ProjectData/hrs/selected_data/03.r4-r16_exclude/rand_filtered_r8_결측제거.csv",
    "C:/workspace/ProjectData/hrs/selected_data/03.r4-r16_exclude/rand_filtered_r9_결측제거.csv",
    "C:/workspace/ProjectData/hrs/selected_data/03.r4-r16_exclude/rand_filtered_r10_결측제거.csv",
    "C:/workspace/ProjectData/hrs/selected_data/03.r4-r16_exclude/rand_filtered_r11_결측제거.csv",
    "C:/workspace/ProjectData/hrs/selected_data/03.r4-r16_exclude/rand_filtered_r12_결측제거.csv",
    "C:/workspace/ProjectData/hrs/selected_data/03.r4-r16_exclude/rand_filtered_r13_결측제거.csv",
    "C:/workspace/ProjectData/hrs/selected_data/03.r4-r16_exclude/rand_filtered_r14_결측제거.csv",
    "C:/workspace/ProjectData/hrs/selected_data/03.r4-r16_exclude/rand_filtered_r15_결측제거.csv",
    "C:/workspace/ProjectData/hrs/selected_data/03.r4-r16_exclude/rand_filtered_r16_결측제거.csv",
]

# 2. 각 파일에 해당하는 year 매핑
year_mapping = {
    "r4": 1995,
    "r5": 1996,
    "r6": 1998,
    "r7": 2000,
    "r8": 2002,
    "r9": 2004,
    "r10": 2006,
    "r11": 2008,
    "r12": 2010,
    "r13": 2012,
    "r14": 2014,
    "r15": 2016,
    "r16": 2018,
}

# 3. 저장할 폴더 설정
output_folder = "C:/workspace/ProjectData/hrs/selected_data/04.r4-r16_year"

# 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 4. 각 파일 처리
for file_path in file_paths:
    # 데이터 로드
    rand = pd.read_csv(file_path)
    
    # 파일명에서 r4~r16에 해당하는 부분 추출 (정규표현식 사용)
    file_name = os.path.basename(file_path)
    match = re.search(r'r(\d+)', file_name)  # "r" 뒤에 숫자가 있는 부분을 추출
    
    if match:
        file_id = "r" + match.group(1)  # r4, r5, ..., r16 추출
        year = year_mapping.get(file_id, None)
        
        if year is not None:
            # 5. year 컬럼 추가
            rand['year'] = year

            # 6. 결과를 새로운 파일로 저장 (파일명 뒤에 _year 추가)
            output_file = os.path.join(output_folder, file_name.replace("_결측제거.csv", "_year.csv"))
            
            rand.to_csv(output_file, index=False, encoding='utf-8')

            print(f"year 컬럼을 추가한 데이터가 저장되었습니다: {output_file}")
        else:
            print(f"파일 {file_name}에 대해 유효한 year이 없습니다.")
    else:
        print(f"파일 {file_name}에서 'r' 뒤에 숫자를 추출할 수 없습니다.")
