# r4부터 r16까지 다른 파일로 저장
import pandas as pd

# 데이터 로드
file_path = "C:/workspace/ProjectData/hrs/selected_data/01.column_filtered_data/rand_filtered.csv"
rand = pd.read_csv(file_path)

# 1. 제외할 컬럼명 설정
exclude_columns = ['hhid', 'rabyear', 'radyear', 'ragender', 'raedyrs', 'raedegrm']

# 2. r4부터 r16까지 각각 필터링하여 저장
for i in range(4, 17):
    # r{i}에 해당하는 컬럼만 필터링 (ex: r1 컬럼만)
    r_columns = [col for col in rand.columns if f'r{i}' in col]  # r{i} 포함된 컬럼 추출
    
    # 제외 컬럼과 r{i} 컬럼을 합쳐서 필터링된 데이터 생성
    columns_to_keep = exclude_columns + r_columns
    
    # 새로운 데이터프레임 생성 (r1부터 r16까지 연도별로)
    filtered_df = rand[columns_to_keep]
    
    # 저장할 파일 경로 설정
    output_file = f"C:/workspace/ProjectData/hrs/selected_data/02.r4-r12_devided/rand_filtered_r{i}.csv"
    
    # 데이터 저장
    filtered_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"r{i} 데이터가 저장되었습니다: {output_file}")
