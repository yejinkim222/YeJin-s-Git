# 결측 있는 컬럼은 결측이었다고 모델에 알려주기 위한 컬럼 추가
# 나이를 5세 단위로 그룹화한 컬럼 추가
import pandas as pd

# 🔧 파일 경로 (예진님이 직접 수정)
file_path = "C:/workspace/ProjectData/hrs/selected_data/11.ml_start/01.AD_train_long_filled.csv"
df = pd.read_csv(file_path)

# 대상 컬럼 리스트
cols_with_minus1_missing = [
    'years_until_hibpe',
    'has_hibpe',
    'years_until_mci',
    'years_until_db',
    'edu_yrs',
    'edu_level'
]

# 각 컬럼에 대해 "_missing" 컬럼 추가
for col in cols_with_minus1_missing:
    missing_col_name = f"{col}_missing"
    df[missing_col_name] = (df[col] == -1).astype(int)

# 5세 단위 그룹화 (예: 60~64세 → 12, 65~69세 → 13)
df['age_group5'] = (df['age'] // 5).astype(int)

print("✅ -1 결측 마킹 완료! 추가된 컬럼:")
print([f"{col}_missing" for col in cols_with_minus1_missing])

# 저장 경로는 예진님이 직접 지정해줘!
save_path = "C:/workspace/ProjectData/hrs/selected_data/11.ml_start/02.AD_train_check_missing.csv"

# 저장 실행
df.to_csv(save_path, index=False, encoding="utf-8-sig")

print("✅ 저장 완료! →", save_path)
