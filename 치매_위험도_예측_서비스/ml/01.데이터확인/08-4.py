import pandas as pd

# 파일 경로 설정
path_rand = "G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"
path_cog = "G:/내 드라이브/코딩/Project01/data/hrs/cogfinalimp_9520wide/cogfinalimp_9520wide.dta"

# 데이터 로드
df_rand = pd.read_stata(path_rand, convert_categoricals=False)
df_cog = pd.read_stata(path_cog, convert_categoricals=False)

# 컬럼명을 모두 소문자로 변환 (병합 용이성 ↑)
df_rand.columns = df_rand.columns.str.lower()
df_cog.columns = df_cog.columns.str.lower()

# hhid별 응답자 수 확인
rand_counts = df_rand['hhid'].value_counts()
cog_counts = df_cog['hhid'].value_counts()

# rand와 cog 모두에서 hhid가 유일한 경우만 추출
unique_hhids = list(set(rand_counts[rand_counts == 1].index) & set(cog_counts[cog_counts == 1].index))

# 유일한 hhid만 필터링
df_rand_unique = df_rand[df_rand['hhid'].isin(unique_hhids)]
df_cog_unique = df_cog[df_cog['hhid'].isin(unique_hhids)]

# 병합
df_merged = pd.merge(df_rand_unique, df_cog_unique, on="hhid", how="inner")

# 병합 결과 확인
print("✅ 병합된 유일 hhid 기반 데이터 shape:", df_merged.shape)
print("✅ 유일한 응답자 수:", len(df_merged))

# 병합된 데이터 일부 확인
print("\n📋 병합 데이터 샘플:")
print(df_merged[['hhid']].head())

# 합성 완료
# 유일 응답자 수: 10769명 나옴