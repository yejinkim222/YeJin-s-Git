import pandas as pd

# Cog 데이터 로드
cog_path = "C:/workspace/ProjectData/hrs/cogfinalimp_9520wide/cogfinalimp_9520wide.dta"
df_cog = pd.read_stata(cog_path)

# 연도별 컬럼 패턴
years = list(range(1995, 2026, 2))  # 1995부터 2025까지 홀수년도만 포함

# cog 데이터에서 연도별 컬럼명만 추출하기 (리스트 컴프리헨션)
cog_columns = [col for col in df_cog.columns if any(str(year) in col for year in years)]

# 19~25 범위의 연도만 포함된 컬럼을 출력
for year in range(1995, 2026, 2):
    columns_for_year = [col for col in cog_columns if str(year) in col]
    if columns_for_year:
        print(f"{year} : {columns_for_year}")

print("포함된 연도별 컬럼 수:")
for year in years:
    count = sum([str(year) in col for col in df_cog.columns])
    print(f"{year}: {count}개")