# r10부터 r16까지
# demene2, alzhee2 -> AD_MCI_status: 둘 중 하나라도 1(yes)면 2(ad)로 추가
import pandas as pd
import os

# 예시 파일 경로 (이미 'r10'부터 'r16'까지 처리한 파일들이 존재)
file_paths = [
    "C:/workspace/ProjectData/hrs/selected_data/08.r4-r12_db/rand_filtered_r10_db.csv",
    "C:/workspace/ProjectData/hrs/selected_data/08.r4-r12_db/rand_filtered_r11_db.csv",
    "C:/workspace/ProjectData/hrs/selected_data/08.r4-r12_db/rand_filtered_r12_db.csv",
    "C:/workspace/ProjectData/hrs/selected_data/08.r4-r12_db/rand_filtered_r13_db.csv",
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
    
    # 4. 'AD_MCI_status' 컬럼 추가: "demene2"와 "alzhee2"에서 하나라도 "1"이면 2, 두 값이 모두 "0"이면 0
    rand['AD_MCI_status'] = rand.apply(
        lambda row: 2 if "1" in str(row['demene2']) or "1" in str(row['alzhee2']) else 0,
        axis=1
    )
    
    # 5. 기존 'demene2','alzhee2'  컬럼 삭제
    rand.drop(columns=['demene2', 'alzhee2', 'rabyear', 'radyear'], inplace=True)

    # 5. 결과를 새로운 파일로 저장 (파일명 뒤에 _processed 추가)
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_folder, file_name.replace("_db.csv", "_ad_mci.csv"))
    
    rand.to_csv(output_file, index=False, encoding='utf-8')

    print(f"'AD_MCI_status' 컬럼이 추가된 파일이 저장되었습니다: {output_file}")
