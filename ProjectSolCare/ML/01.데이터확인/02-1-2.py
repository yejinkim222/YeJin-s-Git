import pandas as pd

# 파일 불러오기
path_rand = "G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"
path_cog = "G:/내 드라이브/코딩/Project01/data/hrs/cogfinalimp_9520wide/cogfinalimp_9520wide.dta"
path_tad = "G:/내 드라이브/코딩/Project01/data/kaggle_tadpole/TADPOLE_D1_D2.csv"

df_rand = pd.read_stata(path_rand)  # RAND HRS (오래 걸릴 수 있음)
df_cog = pd.read_stata(path_cog)    # Cognitive Final
df_tad = pd.read_csv(path_tad)      # TADPOLE (Kaggle)

# rand 데이터 열 이름 중 필요한 키워드 포함된 것 필터링
rand_cols = df_rand.columns[df_rand.columns.str.contains("r8|raeduc|ragender|hhidpn|age|diab|hibp|psych", case=False)]
print("RAND 열 개수:", len(rand_cols))
print("RAND 관련 열 목록:\n", rand_cols.tolist())

#2-1-2. RAND 데이터에서 입력 변수 및 부가 변수 열 찾기