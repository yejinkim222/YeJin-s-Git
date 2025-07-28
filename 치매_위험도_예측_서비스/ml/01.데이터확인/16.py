import pandas as pd

# 파일 경로 (예진님 경로로 바꿔줘야 함)
rand_path = "C:/workspace/ProjectData/hrs/randhrs1992_2022v1/rand_selected_unique.dta"
cog_path = "C:/workspace/ProjectData/hrs/cogfinalimp_9520wide/cog_selected.dta"

# 데이터 불러오기
df_rand = pd.read_stata(rand_path)
df_cog = pd.read_stata(cog_path)

# ✅ 확인 1: cog에 'pn' 컬럼 존재 여부
print("✅ cog 컬럼 목록 (일부):", df_cog.columns[:10])
print("👉 cog에 'pn' 컬럼 있음?:", 'pn' in df_cog.columns)

# ✅ 확인 2: 병합을 위해 cog에 'pn'이 없다면 rand에서 hhid:pn mapping 가져오기
if 'pn' not in df_cog.columns:
    hhid_pn_map = df_rand[['hhid', 'pn']].drop_duplicates()
    df_cog = df_cog.merge(hhid_pn_map, on='hhid', how='left')  # pn 추가
    print("✅ pn 추가됨:", 'pn' in df_cog.columns)
else:
    print("ℹ️ cog에 이미 pn 있음")

# ✅ 확인 3: 병합 가능 여부 (rand 기준 inner 병합 시 손실 확인)
merged_check = df_rand[['hhid', 'pn']].merge(df_cog[['hhid', 'pn']], on=['hhid', 'pn'], how='inner')
print(f"✅ 병합 가능한 샘플 수: {len(merged_check)} / rand 총 샘플: {len(df_rand)}")

# 👉 병합 실패 비율 확인
loss_ratio = 100 * (1 - len(merged_check) / len(df_rand))
print(f"❗ 병합 불일치 비율: {loss_ratio:.2f}%")

# 필요 시 병합 미대상 샘플 출력
if loss_ratio > 0:
    unmatched = df_rand[['hhid', 'pn']].merge(df_cog[['hhid', 'pn']], on=['hhid', 'pn'], how='outer', indicator=True)
    print("❌ 병합되지 않은 샘플 예시:")
    print(unmatched[unmatched['_merge'] != 'both'].head())

# rand는 유일응답자 정제로, cog는 전체데이터 사용해서