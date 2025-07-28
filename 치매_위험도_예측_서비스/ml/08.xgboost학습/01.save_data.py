# lgbm에서 만든 데이터 이용해서
# 결측에 -1 다시 넣고
# 저장하는 코드
import pandas as pd
import numpy as np

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/09.AD_train_lgbm_final_pruned.csv"
df = pd.read_csv(file_path)

# 📌 모든 NaN, inf, -inf 처리 → -1로 대체
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(-1, inplace=True)

# 📌 저장
save_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/10.AD_train_xgboost_filled.csv"
df.to_csv(save_path, index=False)

print(f"✅ 저장 완료: {save_path}")
