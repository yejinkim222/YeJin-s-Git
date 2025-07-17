import pandas as pd

# 파일 불러오기
path_cog = "G:/내 드라이브/코딩/Project01/data/hrs/cogfinalimp_9520wide/cogfinalimp_9520wide.dta"

df_cog = pd.read_stata(path_cog)

# 인지 관련 열 찾기: 'mmse', 'cog', 'diagnosis', 'mci', 'function', 'imp' 포함하는 열 필터링
cog_cols = df_cog.columns[df_cog.columns.str.contains('mmse|cog|diagnosis|mci|function|imp', case=False)]
print("COG 인지 관련 열 개수:", len(cog_cols))
print("COG 인지 관련 열 목록:", cog_cols.tolist())

#2-1-1. cog 데이터에서 MMSE 또는 인지 점수 후보 열 확인