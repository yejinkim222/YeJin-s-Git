import pandas as pd
import os

# 파일 경로 설정 (수정해줘!)
cog_path = "C:/workspace/ProjectData/hrs/selected_data/cogfinalimp_9520wide_selected.dta"
rand_path = "C:/workspace/ProjectData/hrs/selected_data/rand_selected.dta"

# 데이터 불러오기
df_rand = pd.read_stata(rand_path)
df_cog = pd.read_stata(cog_path)

# 유일 응답자 필터링: rand/cog 모두에서 hhid가 1번만 등장하는 경우만 남김
rand_counts = df_rand['hhid'].value_counts()
cog_counts = df_cog['hhid'].value_counts()
unique_hhids = list(set(rand_counts[rand_counts == 1].index) & set(cog_counts[cog_counts == 1].index))

df_rand_unique = df_rand[df_rand['hhid'].isin(unique_hhids)].copy()
df_cog_unique = df_cog[df_cog['hhid'].isin(unique_hhids)].copy()

# 유일 응답자용 파일로 저장 (원래 경로에서 `_unique.dta` 붙여 저장)
df_rand_unique.to_stata(rand_path.replace(".dta", "_unique.dta"), write_index=False)
df_cog_unique.to_stata(cog_path.replace(".dta", "_unique.dta"), write_index=False)

# 한 가정의 유일 응답자만 필터링해서 저장