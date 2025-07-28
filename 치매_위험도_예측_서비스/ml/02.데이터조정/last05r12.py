import pandas as pd
import os

# 1. 파일 경로 설정 (r12에 해당하는 파일 경로)
file_paths = [
    "C:/workspace/ProjectData/hrs/selected_data/02.r4-r12_devided/rand_filtered_r12.csv",  # r12에 해당하는 파일만
]

# 2. 제외할 컬럼명 설정 (삭제되지 않아야 할 컬럼들)
exclude_columns = ['hhid', 'rabyear', 'radyear', 'ragender', 'raedyrs', 'raedegrm']

# 3. r12 컬럼들
r12_columns = [
    "r12dborlmed", "r12depyr", "r12deplos", "r12deptir", "r12depnoap", "r12dephun", 
    "r12depsle", "r12depnit", "r12depcon", "r12depdown", "r12deptho", "r12demene2", "r12alzhee2"
]

# 4. 저장할 폴더 설정
output_folder = "C:/workspace/ProjectData/hrs/selected_data/03.r4-r16_exclude"

# 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 각 파일 처리
for file_path in file_paths:
    # 데이터 로드
    rand = pd.read_csv(file_path)

    # 5. 결측값이 모두 있는 행만 삭제 (r12 컬럼에서 결측값이 모두 있는 행만 삭제)
    filtered_df = rand[exclude_columns + r12_columns]  # 필요한 컬럼만 필터링
    
    # 6. r12 컬럼에서 모든 값이 결측인 행만 삭제
    filtered_df = filtered_df.dropna(subset=r12_columns, how='all')

    # 7. 결과를 새로운 파일로 저장 (파일명 뒤에 _결측제거 추가)
    file_name = os.path.basename(file_path)  # 원본 파일명 추출
    output_file = os.path.join(output_folder, file_name.replace(".csv", "_결측제거.csv"))
    
    filtered_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"결측값이 모두 있는 행을 삭제한 데이터가 저장되었습니다: {output_file}")
