# dborlmed -> has_db: dborlmed에 0이 있으면 0, 1이 있으면 1, 결측이면 삭제
import pandas as pd
import os

# 예시 파일 경로 (이미 'r4'부터 'r16'까지 처리한 파일들이 존재)
file_paths = [
    "C:/workspace/ProjectData/hrs/selected_data/07.r4-r16_del_dep/rand_filtered_r4_del_dep.csv",
    "C:/workspace/ProjectData/hrs/selected_data/07.r4-r16_del_dep/rand_filtered_r5_del_dep.csv",
    "C:/workspace/ProjectData/hrs/selected_data/07.r4-r16_del_dep/rand_filtered_r6_del_dep.csv",
    "C:/workspace/ProjectData/hrs/selected_data/07.r4-r16_del_dep/rand_filtered_r7_del_dep.csv",
    "C:/workspace/ProjectData/hrs/selected_data/07.r4-r16_del_dep/rand_filtered_r8_del_dep.csv",
    "C:/workspace/ProjectData/hrs/selected_data/07.r4-r16_del_dep/rand_filtered_r9_del_dep.csv",
    "C:/workspace/ProjectData/hrs/selected_data/07.r4-r16_del_dep/rand_filtered_r10_del_dep.csv",
    "C:/workspace/ProjectData/hrs/selected_data/07.r4-r16_del_dep/rand_filtered_r11_del_dep.csv",
    "C:/workspace/ProjectData/hrs/selected_data/07.r4-r16_del_dep/rand_filtered_r12_del_dep.csv",
    "C:/workspace/ProjectData/hrs/selected_data/07.r4-r16_del_dep/rand_filtered_r13_del_dep.csv",
    "C:/workspace/ProjectData/hrs/selected_data/07.r4-r16_del_dep/rand_filtered_r14_del_dep.csv",
    "C:/workspace/ProjectData/hrs/selected_data/07.r4-r16_del_dep/rand_filtered_r15_del_dep.csv",
    "C:/workspace/ProjectData/hrs/selected_data/07.r4-r16_del_dep/rand_filtered_r16_del_dep.csv",
]

# 2. 저장할 폴더 설정 (변경된 파일 저장 폴더)
output_folder = "C:/workspace/ProjectData/hrs/selected_data/08.r4-r12_db"

# 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 3. 각 파일 처리
for file_path in file_paths:
    # 데이터 로드
    rand = pd.read_csv(file_path)
    # 4. 결측값이 있는 행 삭제 (dborlmed 컬럼에 결측이 있으면 해당 행 삭제)
    rand.dropna(subset=['dborlmed'], inplace=True)
                
    # 5. 'has_db' 컬럼 추가: dborlmed에 "0"이면 0, "1"이면 1
    rand['has_db'] = rand['dborlmed'].apply(lambda x: 1 if "1" in str(x) else (0 if "0" in str(x) else None))
    
    # 6. 'dborlmed' 컬럼 삭제
    rand.drop(columns=['dborlmed'], inplace=True)

    # 7. 결과를 새로운 파일로 저장 (파일명 뒤에 _processed 추가)
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_folder, file_name.replace("_del_dep.csv", "_db.csv"))
    
    rand.to_csv(output_file, index=False, encoding='utf-8')

    print(f"'has_db' 컬럼이 추가되고 'dborlmed' 컬럼이 삭제된 파일이 저장되었습니다: {output_file}")
