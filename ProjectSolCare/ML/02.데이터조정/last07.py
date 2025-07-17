# age, hhid_year 컬럼 생성
# 'rabyear', 'radyear', 'hhid'는 손으로 삭제
import pandas as pd
import os
import re  # 정규 표현식 사용

# 1. 파일 경로 설정 (r4부터 r16까지의 파일 경로)
file_paths = [
    "C:/workspace/ProjectData/hrs/selected_data/04.r4-r16_year/rand_filtered_r4_year.csv",  # r4
    "C:/workspace/ProjectData/hrs/selected_data/04.r4-r16_year/rand_filtered_r5_year.csv",  # r5
    "C:/workspace/ProjectData/hrs/selected_data/04.r4-r16_year/rand_filtered_r6_year.csv",  # r6
    "C:/workspace/ProjectData/hrs/selected_data/04.r4-r16_year/rand_filtered_r7_year.csv",  # r7
    "C:/workspace/ProjectData/hrs/selected_data/04.r4-r16_year/rand_filtered_r8_year.csv",  # r8
    "C:/workspace/ProjectData/hrs/selected_data/04.r4-r16_year/rand_filtered_r9_year.csv",  # r9
    "C:/workspace/ProjectData/hrs/selected_data/04.r4-r16_year/rand_filtered_r10_year.csv",  # r10
    "C:/workspace/ProjectData/hrs/selected_data/04.r4-r16_year/rand_filtered_r11_year.csv",  # r11
    "C:/workspace/ProjectData/hrs/selected_data/04.r4-r16_year/rand_filtered_r12_year.csv",  # r12
    "C:/workspace/ProjectData/hrs/selected_data/04.r4-r16_year/rand_filtered_r13_year.csv",  # r13
    "C:/workspace/ProjectData/hrs/selected_data/04.r4-r16_year/rand_filtered_r14_year.csv",  # r14
    "C:/workspace/ProjectData/hrs/selected_data/04.r4-r16_year/rand_filtered_r15_year.csv",  # r15
    "C:/workspace/ProjectData/hrs/selected_data/04.r4-r16_year/rand_filtered_r16_year.csv",  # r16
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
output_folder = "C:/workspace/ProjectData/hrs/selected_data/05.r4-r16_hhid_year"

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

            # 6. age 컬럼 추가 (year - rabyear)
            rand['age'] = year - rand['rabyear']
            rand['age'] = rand['age'].astype(int)  # 정수형으로 변환

            # 7. hhid_year 컬럼 추가
            rand['hhid_year'] = rand['hhid'].apply(lambda x: str(x).zfill(6)) + "_" + rand['year'].astype(str)

            # 8. 결과를 새로운 파일로 저장 (파일명 뒤에 _year, _age, _hhid_year 추가)
            output_file = os.path.join(output_folder, file_name.replace("_year.csv", "_hhid_year.csv"))
            
            rand.to_csv(output_file, index=False, encoding='utf-8')

            print(f"year, age, hhid_year 컬럼을 추가한 데이터가 저장되었습니다: {output_file}")
        else:
            print(f"파일 {file_name}에 대해 유효한 year이 없습니다.")
    else:
        print(f"파일 {file_name}에서 'r' 뒤에 숫자를 추출할 수 없습니다.")
