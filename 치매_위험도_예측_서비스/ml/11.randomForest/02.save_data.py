# 변수 제거하고 저장하는 코드
# 기본 컬럼 유지 버전
import pandas as pd

# ✅ 파일 경로 지정
input_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
output_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_for_randomForest.csv"

# ✅ 삭제 대상 컬럼
columns_to_drop = [
    "cognitive_decline_flag", "risk_factor_sum", "db_onset_after_missing", 
    "edu_is_low", "edu_yrs_missing", "ad_year_missing", "year_missing"
]

# ✅ 데이터 불러오기
df = pd.read_csv(input_path)

# ✅ 컬럼 삭제
df = df.drop(columns=columns_to_drop)

# ✅ 저장
df.to_csv(output_path, index=False)
print("✅ 불필요한 변수 제거 후 저장 완료:", output_path)
