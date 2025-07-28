import pandas as pd

# 파일 경로 지정
path = "G:/내 드라이브/코딩/Project01/data/hrs/cogfinalimp_9520wide/cogfinalimp_9520wide.dta"

# 데이터 로딩
df_cog = pd.read_stata(path)

# 사용할 컬럼 목록 지정
cog_cols = [col for col in df_cog.columns if col.startswith("cogfunction") or col.startswith("cogtot27")]

# 각 컬럼의 결측치 수와 고유값 분포 확인
for col in cog_cols:
    print(f"📊 {col} 분포:")
    print(df_cog[col].value_counts(dropna=False).sort_index())
    print("-" * 50)

# 치매, mci 결측치랑 분포 확인용 코드