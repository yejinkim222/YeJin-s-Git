# 확인용
# 우울증 수 확인
# 우울감 느낀 사람도 아무도 없네...
# 눈물 머금고 우울증 포기하기..
import pandas as pd
import os
import re  # 정규 표현식 사용

# 예시 파일 경로 (이미 'r4'부터 'r16'까지 처리한 파일들이 존재)
file_paths = [
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_gender_edu/rand_filtered_r4_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_gender_edu/rand_filtered_r5_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_gender_edu/rand_filtered_r6_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_gender_edu/rand_filtered_r7_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_gender_edu/rand_filtered_r8_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_gender_edu/rand_filtered_r9_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_gender_edu/rand_filtered_r10_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_gender_edu/rand_filtered_r11_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_gender_edu/rand_filtered_r12_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_gender_edu/rand_filtered_r13_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_gender_edu/rand_filtered_r14_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_gender_edu/rand_filtered_r15_gen_edu.csv",
    "C:/workspace/ProjectData/hrs/selected_data/r4-r16_gender_edu/rand_filtered_r16_gen_edu.csv",
]

# 2. 저장할 폴더 설정 (변경된 파일 저장 폴더)
output_folder = "C:/workspace/ProjectData/hrs/selected_data/r4-r16_dep"

# 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 3. 각 파일 처리
for file_path in file_paths:
    # 데이터 로드
    rand = pd.read_csv(file_path)
    
    # 4. 우울증 점수 계산 및 feel_dep 컬럼 추가
    def calculate_feel_dep(row):
        depression_score = 0
        
        # 부정문항 (1이면 1점)
        negative_questions = ['depyr', 'deplos', 'deptir', 'depnoap', 'depsle', 'depnit', 'depdown']
        for q in negative_questions:
            if pd.notna(row[q]) and row[q] == 1:  # NaN이 아닌 값만 확인
                depression_score += 1
        
        # 긍정문항 (0이면 1점)
        positive_questions = ['dephun', 'depcon', 'deptho']
        for q in positive_questions:
            if pd.notna(row[q]) and row[q] == 0:  # NaN이 아닌 값만 확인
                depression_score += 1

        # 2점 이상이면 우울증
        return 1 if depression_score >= 1 else 0

    rand['feel_dep'] = rand.apply(calculate_feel_dep, axis=1)

    # 5. 'feel_dep' 컬럼의 1과 0의 개수 계산
    feel_dep_counts = rand['feel_dep'].value_counts()
    print(f"File: {file_path}")
    print(f"feel_dep - 1: {feel_dep_counts.get(1, 0)}")
    print(f"feel_dep - 0: {feel_dep_counts.get(0, 0)}")
    print("-" * 40)

    # 5. 결과를 새로운 파일로 저장 (파일명 뒤에 _dep 추가)
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_folder, file_name.replace("gen_edu.csv", "dep.csv"))
    
    rand.to_csv(output_file, index=False, encoding='utf-8')

    print(f"우울증 관련 컬럼이 처리되고 저장되었습니다: {output_file}")
