# rand에서 사용할 컬럼 중 실제로 존재하는 컬럼만 저장하고
# 나머지 저장되지 않은, 없는 컬럼은 txt로 저장하기
import pandas as pd

# 데이터 로드 (경로 설정 후 사용)
data_path =  "C:/workspace/ProjectData/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"
df = pd.read_stata(data_path)

# 제공된 컬럼 목록
rand_columns = {
    "id": ["hhid", "pn", "hhidpn"],
    "나이": ["rabyear", "radyear"],
    "성별": ["ragender"],
    "교육 수준": ["raedyrs", "raedegrm"],
    "당뇨": [f"r{i}dborlmed" for i in range(1, 17)],
    "고혈압": [f"r{i}hibpe" for i in range(14, 17)],
    # "우울증": [f"r{i}depyr" for i in range(3, 17)] + 
    #           [f"r{i}deplos" for i in range(3, 17)] + 
    #           [f"r{i}deptir" for i in range(3, 17)] + 
    #           [f"r{i}depnoap" for i in range(3, 17)] + 
    #           [f"r{i}dephun" for i in range(3, 17)] + 
    #           [f"r{i}depsle" for i in range(3, 17)] + 
    #           [f"r{i}depnit" for i in range(3, 17)] + 
    #           [f"r{i}depcon" for i in range(3, 17)] + 
    #           [f"r{i}depdown" for i in range(3, 17)] + 
    #           [f"r{i}deptho" for i in range(3, 17)],
    "AD, MCI": [f"r{w}demene2" for w in range(10, 17)] + 
               [f"r{w}alzhee2" for w in range(10, 17)] + 
               [f"r{w}memrye2" for w in range(4, 10)],
    "치매 onset 시점": [f"r{w}demens" for w in range(11, 17)]
}

# 연도별 치매 관련 컬럼 리스트 (예시)
# '치매 onset 시점' 컬럼들을 연도별로 나누어서 사용
year_columns = [f"r{w}demens" for w in range(11, 17)]  # 예시로 치매 onset 시점 연도별 컬럼

# 1. wide에서 long으로 변환
df_long = df.melt(id_vars=[col for col in df.columns if col not in year_columns],
                  value_vars=year_columns,
                  var_name='year',
                  value_name='치매')

# 2. hhid_year 컬럼 추가 (응답자의 ID와 연도를 합친 새로운 식별자)
df_long['hhid_year'] = df_long['hhid'].astype(str) + '_' + df_long['year'].astype(str)

# 3. 연도에 관련 없는 컬럼 복제 (나머지 컬럼을 각 연도에 대해 복제)
non_year_columns = [col for col in df.columns if col not in year_columns]
df_long[non_year_columns] = df[non_year_columns].iloc[0]  # 첫 번째 행을 사용해서 복제

# 4. 결과 확인
print(df_long.head())

# 5. 변환된 데이터를 Stata 파일로 저장
output_path = "your_output_path_here.dta"  # 실제 경로로 변경
df_long.to_stata(output_path, write_index=False)

