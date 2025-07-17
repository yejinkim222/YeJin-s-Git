# 우울증 변수삭제..
import pandas as pd
import os

# 예시 파일 경로 (이미 'r4'부터 'r16'까지 처리한 파일들이 존재)
file_paths = [
    "C:/workspace/ProjectData/hrs/selected_data/06.r4-r16_gender_edu/rand_filtered_r4_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/06.r4-r16_gender_edu/rand_filtered_r5_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/06.r4-r16_gender_edu/rand_filtered_r6_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/06.r4-r16_gender_edu/rand_filtered_r7_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/06.r4-r16_gender_edu/rand_filtered_r8_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/06.r4-r16_gender_edu/rand_filtered_r9_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/06.r4-r16_gender_edu/rand_filtered_r10_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/06.r4-r16_gender_edu/rand_filtered_r11_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/06.r4-r16_gender_edu/rand_filtered_r12_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/06.r4-r16_gender_edu/rand_filtered_r13_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/06.r4-r16_gender_edu/rand_filtered_r14_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/06.r4-r16_gender_edu/rand_filtered_r15_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/06.r4-r16_gender_edu/rand_filtered_r16_gen_edu.csv",
]

# 2. 저장할 폴더 설정 (변경된 파일 저장 폴더)
output_folder = "C:/workspace/ProjectData/hrs/selected_data/07.r4-r16_del_dep"

# 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 3. 각 파일 처리
for file_path in file_paths:
    # 데이터 로드
    rand = pd.read_csv(file_path)

    # 4. 'dep'로 시작하는 모든 컬럼 삭제
    dep_columns = [col for col in rand.columns if col.startswith('dep')]
    rand.drop(columns=dep_columns, inplace=True)
    
    # 5. 결과를 새로운 파일로 저장 (파일명 뒤에 _processed 추가)
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_folder, file_name.replace("gen_edu.csv", "del_dep.csv"))
    
    rand.to_csv(output_file, index=False, encoding='utf-8')

    print(f"'dep' 컬럼들이 삭제되고 저장되었습니다: {output_file}")
