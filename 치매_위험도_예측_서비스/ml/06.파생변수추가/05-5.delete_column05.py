# ✅ 결과 저장하기 (기본 컬럼 보존)
import pandas as pd

# ✅ 데이터 로드
df = pd.read_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/08.AD_train_selected_features.csv")

# ✅ 삭제하면 안 되는 기본 컬럼
essential_cols = [
    "edu_yrs", "edu_level", "years_until_mci", "has_db", "gender",
    "hhid", "year", "hhid_year"
]

# ✅ 제거할 변수 리스트 (중요도 낮은 파생 변수만)
low_importance_cols = [
    "years_until_db_missing", "years_until_mci_missing", "log_age_ratio", 
    "log_risk_weighted_age", "edu_is_low", "age_group5", "has_hibpe_missing",
    "high_risk_group", "risk_factor_sum", "is_old", "years_until_hibpe_missing", 
    "risk_to_age_ratio", "edu_yrs_missing", "is_low_edu"
]

# ✅ 필터링: 삭제 금지 컬럼은 제거 리스트에서 제외
filtered_cols = [col for col in low_importance_cols if col not in essential_cols]

# ✅ 제거 수행
df = df.drop(columns=filtered_cols, errors="ignore")

# ✅ 파일 저장
df.to_csv("C:/workspace/Project01/data/hrs/selected_data/11.ml_start/09.AD_train_lgbm_final_pruned.csv", index=False)
print("🎉 저장 완료: 09.AD_train_final_pruned.csv")
