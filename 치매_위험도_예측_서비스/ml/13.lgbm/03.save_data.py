# 데이터 저장
import pandas as pd

# 📌 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
df = pd.read_csv(file_path)

# ✅ ▶▶ 성능 향상된 파생 변수 생성

# 1. years_until_ad_ratio
df["years_until_ad_ratio"] = df["years_until_ad"] / (df["age"] + 1e-5)

# 2. onset_after_total (결측은 0으로 처리)
onset_vars = ["db_onset_after", "hibpe_onset_after", "mci_onset_after"]
df["onset_after_total"] = df[onset_vars].apply(
    lambda row: sum([v if v >= 0 else 0 for v in row]), axis=1
)

# 3. has_multiple_risk (결측 -1은 무시, 2개 이상 위험요인 플래그)
df["has_multiple_risk"] = (
    ((df["has_db"] == 1).astype(int)) +
    ((df["has_hibpe"] == 1).astype(int)) +
    ((df["AD_MCI_status"] >= 1).astype(int))
) >= 2
df["has_multiple_risk"] = df["has_multiple_risk"].astype(int)

# 📌 ✅ 기존 성능 향상 파생변수 (예진님 작성)
df["age_x_edu"] = df["age"] * df["edu_yrs"]
df["hibpe_onset_delay_ratio"] = df["hibpe_onset_after"] / (df["age"] + 1e-3)

# 📌 ⛔ 삭제할 변수 목록 (성능 기여 낮음)
to_delete = [
    "has_hibpe_missing", "hibpe_onset_after_missing", "age_group5",
    "edu_level_bucket", "edu_is_low", "mci_onset_after_missing"
]

# 📌 🔄 모델 학습에서 제외하지만 데이터셋에는 남겨야 할 변수
model_exclude_only = ["has_db", "AD_MCI_status", "edu_level"]

# ⚠️ 예외 처리: 삭제 리스트에서 학습 제외 전용 컬럼은 유지
to_delete_filtered = [col for col in to_delete if col not in model_exclude_only]

# 📌 ✅ 컬럼 제거
df_cleaned = df.drop(columns=[col for col in to_delete_filtered if col in df.columns])

# 📌 💾 저장
save_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_for_lgbm.csv"
df_cleaned.to_csv(save_path, index=False)

print("✅ 저장 완료:", save_path)
