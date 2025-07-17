import pandas as pd

# 로컬 경로에 맞게 수정 필요
path_cog = "G:/내 드라이브/코딩/Project01/data/hrs/cogfinalimp_9520wide/cogfinalimp_9520wide.dta"
output_path = "G:/내 드라이브/코딩/Project01/data/hrs/cogfinalimp_9520wide/cog_selected.dta"

# 데이터 불러오기
df_cog = pd.read_stata(path_cog)

# 사용할 변수 접두사 및 연도 리스트
prefixes = [
    "cogfunction", "cogtot27_imp", "imrc_imp", "dlrc_imp",
    "ser7_imp", "bwc20_imp", "memoryp_imp", "numiadl_imp"
]
years = [1995, 1996, 1998, 2000, 2002, 2004, 2006, 2008,
         2010, 2012, 2014, 2016, 2018, 2020]

# 사용할 컬럼 리스트 구성
selected_columns = ['hhid']
for prefix in prefixes:
    for year in years:
        col = f"{prefix}{year}"
        if col in df_cog.columns:
            selected_columns.append(col)

# 선택한 컬럼만 추출
df_selected = df_cog[selected_columns]

# 저장
df_selected.to_stata(output_path, write_index=False)

# 확인용 출력
print("저장된 컬럼 수:", len(df_selected.columns))
print("저장된 컬럼 목록:", df_selected.columns.tolist()[:10], "... 외", len(df_selected.columns) - 10, "개 더 있음")
print("저장 완료:", output_path)

# cog 변수 추출해서 따로 저장