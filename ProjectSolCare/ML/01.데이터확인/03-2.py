import pandas as pd

# cog 데이터 열 목록 확인
cog_path = r'G:\내 드라이브\코딩\Project01\data\hrs\randhrs1992_2022v1/cogfinalimp_9520wide.dta'
df_cog = pd.read_stata(cog_path)
print("Cognition 데이터 열 이름:")
print(df_cog.columns.tolist())

# tadpole 데이터 열 목록 확인
tad_path = r'G:/내 드라이브/코딩/Project01/data/kaggle_tadpole/TADPOLE_D1_D2.csv'
df_tad = pd.read_csv(tad_path)
print("\nTADPOLE 데이터 열 이름:")
print(df_tad.columns.tolist())

# 코드북 보고 컬럼명 리스트 확인용