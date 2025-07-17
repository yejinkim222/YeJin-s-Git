# ragender -> gender: "1.male"은 0, "2.female"은 1로 변환
# raedyrs -> edu_yrs: "0.none" → 0, "17.17+ yrs" → 17
# raedegrm 컬럼 삭제
# r(숫자)로 된 컬럼은 뒤에 붙은 부분만 남기기
import pandas as pd
import os
import re  # 정규 표현식 사용

# 예시 파일 경로 (이미 'r4'부터 'r16'까지 처리한 파일들이 존재)
file_paths = [
    "C:/workspace/ProjectData/hrs/selected_data/05.r4-r16_hhid_year/rand_filtered_r4_hhid_year.csv",
    "C:/workspace/ProjectData/hrs/selected_data/05.r4-r16_hhid_year/rand_filtered_r5_hhid_year.csv",
    "C:/workspace/ProjectData/hrs/selected_data/05.r4-r16_hhid_year/rand_filtered_r6_hhid_year.csv",
    "C:/workspace/ProjectData/hrs/selected_data/05.r4-r16_hhid_year/rand_filtered_r7_hhid_year.csv",
    "C:/workspace/ProjectData/hrs/selected_data/05.r4-r16_hhid_year/rand_filtered_r8_hhid_year.csv",
    "C:/workspace/ProjectData/hrs/selected_data/05.r4-r16_hhid_year/rand_filtered_r9_hhid_year.csv",
    "C:/workspace/ProjectData/hrs/selected_data/05.r4-r16_hhid_year/rand_filtered_r10_hhid_year.csv",
    "C:/workspace/ProjectData/hrs/selected_data/05.r4-r16_hhid_year/rand_filtered_r11_hhid_year.csv",
    "C:/workspace/ProjectData/hrs/selected_data/05.r4-r16_hhid_year/rand_filtered_r12_hhid_year.csv",
    "C:/workspace/ProjectData/hrs/selected_data/05.r4-r16_hhid_year/rand_filtered_r13_hhid_year.csv",
    "C:/workspace/ProjectData/hrs/selected_data/05.r4-r16_hhid_year/rand_filtered_r14_hhid_year.csv",
    "C:/workspace/ProjectData/hrs/selected_data/05.r4-r16_hhid_year/rand_filtered_r15_hhid_year.csv",
    "C:/workspace/ProjectData/hrs/selected_data/05.r4-r16_hhid_year/rand_filtered_r16_hhid_year.csv",
]

# 2. 저장할 폴더 설정 (변경된 파일 저장 폴더)
output_folder = "C:/workspace/ProjectData/hrs/selected_data/06.r4-r16_gender_edu"

# 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 3. 각 파일 처리
for file_path in file_paths:
    # 데이터 로드
    rand = pd.read_csv(file_path)
    
    # 4. 'ragender'를 'gender'로 바꾸고 1.male은 0, 2.female은 1로 변경
    rand['gender'] = rand['ragender'].apply(lambda x: 0 if "1" in str(x) else (1 if "2" in str(x) else None))
    
    # 5. 'raedyrs'를 'edu_yrs'로 바꾸고 "0.none"은 0, "17.17+ yrs"는 17로 변경
    # 0.none과 17.17+ yrs만 변환하고, 나머지 값은 그대로 둡니다.
    rand['edu_yrs'] = rand['raedyrs'].apply(lambda x: 0 if str(x) == "0.none" else (17 if "17" in str(x) else x))
    
    # 6. 'raedegrm', 'ragender', 'raedyrs' 컬럼 삭제
    rand.drop(columns=['raedegrm', 'ragender', 'raedyrs'], inplace=True)
    
    # 7. 'r(숫자)'로 시작하는 컬럼들에서 'r4', 'r5', ..., 'r16'을 삭제하고 영어 부분만 남기기
    r_columns = [col for col in rand.columns if re.match(r'r\d+', col)]
    for col in r_columns:
        new_col_name = re.sub(r'r\d+', '', col)  # 'r4depyr' -> 'depyr'
        rand.rename(columns={col: new_col_name}, inplace=True)
    
    # 8. 결과를 새로운 파일로 저장 (파일명 뒤에 _gen_edu 추가)
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_folder, file_name.replace("_hhid_year.csv", "_gen_edu.csv"))
    
    rand.to_csv(output_file, index=False, encoding='utf-8')

    print(f"데이터가 처리되고 저장되었습니다: {output_file}")
