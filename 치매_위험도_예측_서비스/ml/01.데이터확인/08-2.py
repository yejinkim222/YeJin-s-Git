import pandas as pd

# 1. 데이터 불러오기
df_rand = pd.read_stata("G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta")
df_cog = pd.read_stata("G:/내 드라이브/코딩/Project01/data/hrs/cogfinalimp_9520wide/cogfinalimp_9520wide.dta")

# 2. rand 컬럼명 정리 (소문자+공백제거)
df_rand.columns = df_rand.columns.str.lower().str.strip()

# 3. cog 컬럼명 정리 (동일하게 적용)
df_cog.columns = df_cog.columns.str.lower().str.strip()

# 4. 각 데이터에서 hhid 값 추출
rand_hhids = df_rand['hhid'].astype(str).str.strip().str.lower().unique()
cog_hhids = df_cog['hhid'].astype(str).str.strip().str.lower().unique()

# 5. 교집합 개수 확인
overlap = set(rand_hhids) & set(cog_hhids)
print(f"공통 HHID 수: {len(overlap)}")

# 6. cog 데이터 내 hhid당 사람 수 분포도 확인
hhid_counts = df_cog['hhid'].value_counts()
print("▶ cog 내 가구당 응답자 수 요약:\n", hhid_counts.describe())
print(f"▶ 2명 이상인 가구 수: {(hhid_counts > 1).sum()}")


# rand와 cog의 hhid 병합 가능한지 확인하기