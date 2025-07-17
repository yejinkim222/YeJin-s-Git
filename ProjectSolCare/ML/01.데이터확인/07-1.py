import pandas as pd

# 1. 경로 지정
path_cog = 'G:/내 드라이브/코딩/Project01/data/hrs/cogfinalimp_9520wide/cogfinalimp_9520wide.dta'

# 2. 데이터프레임으로 읽기
df_cog = pd.read_stata(path_cog)

# 3. ID 관련 변수 추출
id_cols = [col for col in df_cog.columns if 'id' in col.lower()]
print("ID 관련 변수 목록:", id_cols)

# id 관련 변수는 hhid