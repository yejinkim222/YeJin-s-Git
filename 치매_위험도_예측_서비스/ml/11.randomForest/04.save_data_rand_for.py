# 데이터 저장하기
# 기본 컬럼 없이, 모델 학습용만 남긴 데이터
import pandas as pd

# ✅ 원본 데이터 경로 (결측 마킹 포함된 버전)
input_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"

# ✅ 저장 경로 (RandomForest용 최종 전처리 버전)
output_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_rf_final.csv"

# ✅ 삭제 대상 컬럼
columns_to_drop = [
    "cognitive_decline_flag", "risk_factor_sum", "db_onset_after_missing",
    "edu_is_low", "edu_yrs_missing", "ad_year_missing", "year_missing"
]

# ✅ 학습 제외 대상
exclude_cols = ["AD_MCI_status", "gender", "has_db", "year"]

# ✅ 로딩 및 삭제
df = pd.read_csv(input_path)
df = df.drop(columns=columns_to_drop)

# ✅ 피처와 타겟 분리
target_col = "years_until_ad"
X_cols = [col for col in df.columns if col not in exclude_cols + [target_col]]
df_final = df[X_cols + [target_col]]  # 타겟은 맨 뒤에 포함

# ✅ 저장
df_final.to_csv(output_path, index=False)
print(f"✅ 최종 전처리 완료 → {output_path}")
