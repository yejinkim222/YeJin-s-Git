import pandas as pd
import os

# ✅ 유일 응답자 기반 파일 경로 설정 (수정해줘!)
rand_path = "C:/workspace/ProjectData/hrs/unique_data/rand_selected_unique.dta"
cog_path = "C:/workspace/ProjectData/hrs/unique_data/cogfinalimp_9520wide_selected_unique.dta"

# ✅ 저장할 폴더 설정
output_dir = "C:/workspace/ProjectData/hrs/last_data"
os.makedirs(output_dir, exist_ok=True)

# ✅ 데이터 불러오기
df_rand = pd.read_stata(rand_path)
df_cog = pd.read_stata(cog_path)

# ✅ 연도 및 wave 매핑
years = list(range(1995, 2021, 2))  # 1995 ~ 2020
waves = list(range(1, len(years) + 1))  # r1 ~ r14

# ✅ 각 연도별로 rand, cog 분할 + hhid_year 추가 + 저장
for wave, year in zip(waves, years):
    # ✅ RAND: r[wave]로 시작하는 컬럼 + 식별자
    rand_cols = ['hhid', 'pn', 'hhidpn'] + [col for col in df_rand.columns if col.startswith(f"r{wave}")]
    if len(rand_cols) > 3:  # 유의미한 컬럼 있을 때만
        rand_df = df_rand[rand_cols].copy()
        rand_df["hhid_year"] = rand_df["hhid"].astype(str) + f"_{year}"  # 고유 식별자
        rand_df.to_stata(os.path.join(output_dir, f"rand_{year}.dta"), write_index=False)

    # ✅ COG: 해당 연도로 끝나는 컬럼 + hhid
    cog_cols = ['hhid'] + [col for col in df_cog.columns if col.endswith(str(year))]
    if len(cog_cols) > 1:
        cog_df = df_cog[cog_cols].copy()
        cog_df["hhid_year"] = cog_df["hhid"].astype(str) + f"_{year}"  # 고유 식별자
        cog_df.to_stata(os.path.join(output_dir, f"cog_{year}.dta"), write_index=False)

# cog, rand 각각 연도별로 다른사람 취급해서 저장
# 식별용 hhid_year 추가: hhid와 year를 남겨서 저장한 것
# 병합 키로 사용 가능하고, 전처리 확인이나 개별 샘플 추적, 분석 가능
# rand의 wave 15, 16은 cog와 병합 불가하므로 제외하고 저장