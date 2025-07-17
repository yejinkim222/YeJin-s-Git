# hhid를 6자리 숫자로 추가
# memrye2 -> AD_MCI_status: 없었으면 0(normal) 있었으면 1(mci)로 추가
# memrye2는 r4~r9까지만 해당
import pandas as pd
import os

# 예시 파일 경로 (이미 'r4'부터 'r12'까지 처리한 파일들이 존재)
file_paths = [
    "C:/workspace/ProjectData/hrs/selected_data/08.r4-r12_db/rand_filtered_r4_db.csv",
    "C:/workspace/ProjectData/hrs/selected_data/08.r4-r12_db/rand_filtered_r5_db.csv",
    "C:/workspace/ProjectData/hrs/selected_data/08.r4-r12_db/rand_filtered_r6_db.csv",
    "C:/workspace/ProjectData/hrs/selected_data/08.r4-r12_db/rand_filtered_r7_db.csv",
    "C:/workspace/ProjectData/hrs/selected_data/08.r4-r12_db/rand_filtered_r8_db.csv",
    "C:/workspace/ProjectData/hrs/selected_data/08.r4-r12_db/rand_filtered_r9_db.csv"
]

# 2. 저장할 폴더 설정 (변경된 파일 저장 폴더)
output_folder = "C:/workspace/ProjectData/hrs/selected_data/09.r4-r16_ad_mci"

# 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 3. 각 파일 처리
for file_path in file_paths:
    # 데이터 로드
    rand = pd.read_csv(file_path)
    
    # 4. 'memrye2' 컬럼을 'AD_MCI_status'로 변환
    rand['AD_MCI_status'] = rand['memrye2'].apply(lambda x: 1 if "1" in str(x) else (0 if "0" in str(x) else None))
    
    # 5. 기존 'memrye2' 컬럼 삭제
    rand.drop(columns=['memrye2', 'rabyear', 'radyear'], inplace=True)
    
    # 8. 결과를 새로운 파일로 저장 (파일명 뒤에 _processed 추가)
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_folder, file_name.replace("_db.csv", "_ad_mci.csv"))
    
    rand.to_csv(output_file, index=False, encoding='utf-8')

    print(f"파일이 처리되고 저장되었습니다: {output_file}")
