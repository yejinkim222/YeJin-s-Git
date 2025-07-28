# 결측치 비율 확인하고 -1로 채워넣기
import pandas as pd

# 파일 경로
input_path = "C:/workspace/ProjectData/hrs/selected_data/10.result/final_AD_train_long_최초시점_onset.csv"
output_path = "C:/workspace/ProjectData/hrs/selected_data/11.ml_start/01.AD_train_long_filled.csv"

# 1. 데이터 불러오기
df = pd.read_csv(input_path)

# 2. 결측치 비율 확인
missing_ratio = df.isnull().mean().sort_values(ascending=False) * 100
print("📌 컬럼별 결측치 비율 (%):")
print(missing_ratio)

# 3. 결측치 -1로 채우기
df_filled = df.fillna(-1)

# 4. 저장
df_filled.to_csv(output_path, index=False)
print(f"\n✅ 모든 결측치를 -1로 대체하고 저장 완료: {output_path}")
