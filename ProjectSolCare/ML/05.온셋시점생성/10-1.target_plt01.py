# 너무 안나와서
# 데이터 생긴 게 문젠가 싶어가지고
# 일단 years_until_ad 생겨먹은 거 확인하기
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 경로
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"

# 데이터 로딩
df = pd.read_csv(file_path)

# 결측치 제거 및 샘플링 (최대 1000개)
available = df["years_until_ad"].dropna()
sample_size = min(1000, len(available))
sampled = available.sample(n=sample_size, random_state=42)

# 히스토그램 시각화
plt.figure(figsize=(10, 6))
plt.hist(sampled, bins=30, edgecolor="black")
plt.title("Distribution of years_until_ad")
plt.xlabel("Years Until AD")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
