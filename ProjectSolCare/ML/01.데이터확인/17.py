import os
import pandas as pd

# 파일 경로
cog_path = "C:/workspace/ProjectData/hrs/cogfinalimp_9520wide/cogfinalimp_9520wide.dta"
rand_path = "C:/workspace/ProjectData/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"

#  원본 데이터 로드
df_cog = pd.read_stata(cog_path)
df_rand = pd.read_stata(rand_path)

#  연도 리스트 생성 (1995부터 2025까지 홀수년)
years = list(range(1995, 2026, 2))  # 홀수년

#  cog 데이터에서 연도별 컬럼만 추출
cog_columns = ["cogfunction", "cogtot27_imp", "imrc_imp", "dlrc_imp", "ser7_imp", "bwc20_imp", "memoryp_imp", "numiadl_imp"]
cog_year_columns = [f"{col}{year}" for col in cog_columns for year in years]

#  cog 데이터에서 존재하는 컬럼만 필터링
existing_cog_columns = [col for col in cog_year_columns if col in df_cog.columns]

#  없는 연도별 컬럼을 출력
missing_cog_columns = [col for col in cog_year_columns if col not in df_cog.columns]
with open("missing_columns_cog.txt", "w", encoding="utf-8") as file:
    file.write("Cog 데이터에서 없는 연도별 컬럼:\n")
    for col in missing_cog_columns:
        file.write(f'{col}\n')

#  필요한 컬럼만 남기기
df_cog_filtered = df_cog[existing_cog_columns]

#  NaN 값을 'missing'으로 처리 (Categorical 변수의 경우 'missing'을 카테고리로 추가)
# 먼저 모든 컬럼에 대해 'missing'을 추가할 수 있도록 범주 설정
for col in df_rand.select_dtypes(include='category').columns:
    df_rand[col] = df_rand[col].cat.add_categories('missing')

for col in df_cog.select_dtypes(include='category').columns:
    df_cog[col] = df_cog[col].cat.add_categories('missing')

# NaN 값을 'missing'으로 처리 (object 타입 컬럼은 object로 변환 후 처리)
df_cog_filtered = df_cog_filtered.apply(lambda x: x.fillna('missing') if x.dtype == 'O' else x.fillna(pd.NA))

# Rand 데이터 처리
rand_columns = {
    "id": ["hhid", "pn", "hhidpn"],
    "나이": ["rabyear", "radyear"],
    "성별": ["ragender", "gender"],
    "교육 수준": ["raedyrs", "raedegrm"],
    "당뇨": [f"r{i}dborlmed" for i in range(1, 17)],
    "고혈압": [f"r{i}hibpe" for i in range(14, 17)],
    "우울증": [f"r{i}depyr" for i in range(3, 17)] + [f"r{i}deplos" for i in range(3, 17)] + [f"r{i}deptir" for i in range(3, 17)] + [f"r{i}depnoap" for i in range(3, 17)] + [f"r{i}dephun" for i in range(3, 17)] + [f"r{i}depsle" for i in range(3, 17)] + [f"r{i}depnit" for i in range(3, 17)] + [f"r{i}depcon" for i in range(3, 17)] + [f"r{i}depdown" for i in range(3, 17)] + [f"r{i}deptho" for i in range(3, 17)],
    "AD, MCI": [f"r{w}demene2" for w in range(10, 17)] + [f"r{w}alzhee2" for w in range(4, 17)] + [f"r{w}memrye2" for w in range(4, 17)],
    "치매 onset 시점": [f"r{w}demens" for w in range(11, 17)]
}

# rand 데이터에서 사용할 컬럼만 남기기
# 먼저 존재하는 컬럼만 확인
existing_rand_columns = [col for col in sum(rand_columns.values(), []) if col in df_rand.columns]

# 누락된 컬럼을 파일로 저장
missing_rand_columns = [col for col in sum(rand_columns.values(), []) if col not in df_rand.columns]
with open("missing_columns_rand.txt", "w", encoding="utf-8") as file:
    file.write("Rand 데이터에서 없는 wave 컬럼:\n")
    for col in missing_rand_columns:
        file.write(f'{col}\n')

# rand 데이터에서 존재하는 컬럼만 필터링
df_rand_filtered = df_rand[existing_rand_columns]

# NaN 값을 'missing'으로 처리 (inplace=True 대신 새로운 할당)
df_rand_filtered = df_rand_filtered.apply(lambda x: x.fillna('missing') if x.dtype == 'O' else x.fillna(pd.NA))

# 파일로 저장: cog와 rand 데이터셋 저장
df_cog_filtered.to_stata("C:/workspace/ProjectData/hrs/selected_data/cog_filtered.dta", write_index=False)
df_rand_filtered.to_stata("C:/workspace/ProjectData/hrs/selected_data/rand_filtered.dta", write_index=False)

print("필요한 컬럼만 추출하여 저장 완료")
