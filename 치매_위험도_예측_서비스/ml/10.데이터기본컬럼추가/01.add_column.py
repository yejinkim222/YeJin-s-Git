# 인지 문제 여부 컬럼 추가하기
import pandas as pd
import glob

# ✅ 대상 파일 경로 패턴
file_pattern = "C:/workspace/Project01/data/hrs/selected_data/07.r4-r16_del_dep/지워도되나/rand_filtered_r*_processed.csv"

# ✅ memrye2 추출 결과 리스트
memrye2_list = []

# 🔁 파일별로 읽고 필요한 컬럼 추출
for file in glob.glob(file_pattern):
    df_header = pd.read_csv(file, nrows=0)
    if "memrye2" in df_header.columns:
        df = pd.read_csv(file, usecols=["hhid_year", "memrye2"])
        memrye2_list.append(df)

# ✅ 병합
memrye2_combined = pd.concat(memrye2_list, ignore_index=True)

# ✅ 인지저하 여부 플래그 생성 (1: 있음, 0: 없음, -1: 결측)
def map_memrye2(val):
    if pd.isna(val):
        return -1
    elif val in [1, 5]:
        return 1
    else:
        return 0

memrye2_combined["cognitive_decline_flag"] = memrye2_combined["memrye2"].apply(map_memrye2)
memrye2_combined = memrye2_combined[["hhid_year", "cognitive_decline_flag"]]

# ✅ 예진님 분석용 데이터 불러오기
train_df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/10.result/final_AD_train_long_최초시점_filled_derived.csv")

# ✅ 병합 (left join)
train_df = train_df.merge(memrye2_combined, on="hhid_year", how="left")

# ✅ 결측은 -1로 마킹 (병합 시 누락된 경우)
train_df["cognitive_decline_flag"] = train_df["cognitive_decline_flag"].fillna(-1).astype(int)

# ✅ 저장
train_df.to_csv("C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv", index=False)
