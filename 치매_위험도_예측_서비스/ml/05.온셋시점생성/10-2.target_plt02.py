# 타겟 로그 변환해보기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv"
df = pd.read_csv(file_path)

# 원본과 로그 변환된 타겟 생성
original_target = df['years_until_ad']
log_target = np.log1p(df['years_until_ad'])

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(original_target, bins=25, edgecolor='black')
axes[0].set_title("Original Distribution of years_until_ad")
axes[0].set_xlabel("Years Until AD")
axes[0].set_ylabel("Frequency")

axes[1].hist(log_target, bins=25, edgecolor='black')
axes[1].set_title("Log-Transformed Distribution (log1p)")
axes[1].set_xlabel("log1p(Years Until AD)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
