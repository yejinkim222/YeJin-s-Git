import pandas as pd

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(file_path)

# ✅ 파생변수 생성
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)
df["edu_level_bucket"] = df["edu_level"] // 2

# ✅ 삭제할 컬럼 정의
cols_to_remove = [
    "ad_year", "year",            # 타겟 유출 위험
    "ad_year_missing", "year_missing"  # 중요도 낮음
]

# ✅ 삭제 수행
df_cleaned = df.drop(columns=[col for col in cols_to_remove if col in df.columns])

# ✅ 저장
save_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_xgb_final.csv"
df_cleaned.to_csv(save_path, index=False)

print("✅ 최종 저장 완료:", save_path)
