import pandas as pd

# 파일 경로
path_rand = r"G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"
path_cog = r"G:/내 드라이브/코딩/Project01/data/hrs/cogfinalimp_9520wide/cogfinalimp_9520wide.dta"

# 1. 데이터 불러오기
df_rand = pd.read_stata(path_rand)
df_cog = pd.read_stata(path_cog)

# 2. rand의 모든 컬럼명을 소문자로 변환해서 처리
df_rand.columns = df_rand.columns.str.lower()
df_cog.columns = df_cog.columns.str.lower()

# 3. hhid 기준으로 병합 (공통 컬럼: hhid)
if 'hhid' not in df_rand.columns or 'hhid' not in df_cog.columns:
    raise KeyError("두 데이터 중 하나에 'hhid' 컬럼이 존재하지 않습니다.")

# 4. 병합 (inner: 두 데이터 모두 존재하는 사람만)
merged = pd.merge(df_cog, df_rand, on='hhid', how='inner')

# 5. 병합 결과 확인
print(f"병합된 데이터 크기: {merged.shape}")
print(f"병합된 데이터 예시:\n{merged[['hhid']].head()}")


# 병합하는 가능한지
# 데이터 얼마나 나오는지 확인하는 코드