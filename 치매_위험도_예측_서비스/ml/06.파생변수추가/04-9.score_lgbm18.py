# 중요도 낮은 파생변수 제거하고
# 새로 만든 파생변수 효과 좋길래 저장하기
import pandas as pd
import numpy as np

# ✅ 데이터 로딩
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/07.AD_train_with_features.csv")

# ✅ 새 파생변수 생성
df["age_edu_ratio"] = df["age"] / (df["edu_yrs"] + 1)
df["log_age_ratio"] = np.log1p(df["age_edu_ratio"])
df["edu_male_diff"] = df["edu_yrs"] - df["male_age"]
df["age_male_diff"] = df["male_age"] - df["age"]

# ✅ age_group5 ≥ 10 필터링
df = df[df["age_group5"] >= 10].copy()

# ✅ 제거할 파생변수
remove_cols = [
    "is_old_and_low_edu",
    "has_db_and_hibpe",
    "has_any_chronic",
    "years_until_hibpe",
    "age_minus_risk"
]

# ✅ 제거
df.drop(columns=remove_cols, inplace=True, errors="ignore")

# ✅ 저장
save_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/08.AD_train_selected_features.csv"
df.to_csv(save_path, index=False)
print(f"📁 저장 완료: {save_path}")
