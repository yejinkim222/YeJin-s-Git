import pandas as pd

# 파일 경로
path_rand = "G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"
path_cog = "G:/내 드라이브/코딩/Project01/data/hrs/cogfinalimp_9520wide/cogfinalimp_9520wide.dta"
path_tad = "G:/내 드라이브/코딩/Project01/data/kaggle_tadpole/TADPOLE_D1_D2.csv"

# 파일 불러오기
df_rand = pd.read_stata(path_rand)  # RAND HRS (오래 걸릴 수 있음)
df_cog = pd.read_stata(path_cog)    # Cognitive Final
df_tad = pd.read_csv(path_tad)      # TADPOLE (Kaggle)

# 각 데이터셋 기본 구조 확인
print("RAND HRS:", df_rand.shape)
print("COG Final:", df_cog.shape)
print("TADPOLE:", df_tad.shape)

print("COG 열 목록:", df_cog.columns.tolist()[:10])  # 열 이름 일부 확인

#세 파일 모두 열어서 구조 확인