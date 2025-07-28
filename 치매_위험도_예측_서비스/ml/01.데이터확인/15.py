import pandas as pd

# 원본 경로
cog_path = "C:/workspace/ProjectData/hrs/cogfinalimp_9520wide/cogfinalimp_9520wide.dta"
df_cog = pd.read_stata(cog_path)

# 컬럼명을 소문자로 통일
df_cog.columns = df_cog.columns.str.lower()

# ID 컬럼 유지
id_cols = ["hhid"]

# 사용할 연도 리스트
years = list(range(1995, 2021, 1))  # 1995~2020 연도별

# 추출할 변수 prefix (앞에 붙는 부분)
prefixes = ["cogfunction", "cogtot27_imp", "imrc_imp", "dlrc_imp", "ser7_imp", "bwc20_imp", "memoryp_imp", "numiadl_imp"]

# 연도별로 변수명 조합해서 컬럼 필터링
selected_cols = id_cols + [
    col for col in df_cog.columns
    if any(col.startswith(prefix) and col.endswith(str(year)) for prefix in prefixes for year in years)
]

# 필터링된 데이터 확인
df_cog_selected = df_cog[selected_cols].copy()

# 저장
output_path = cog_path.replace(".dta", "_selected.dta")
df_cog_selected.to_stata(output_path, write_index=False)
print(f"✅ 저장 완료: {output_path}")
print(f"✅ 저장된 컬럼 수: {len(df_cog_selected.columns)}")

# cog 1995년 말고는 다 실종되어서....
# 다시 찾아서 여행 떠나기 ㅜ
# 이거 하고나서 13이랑 14 다시해보기
# 의미없어졌네 다시해