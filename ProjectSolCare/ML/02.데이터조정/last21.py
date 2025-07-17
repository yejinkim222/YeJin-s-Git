# 컬럼명에 관계없이 결측이랑 분포 확인하기
import pandas as pd

# 파일 경로 설정 (예진님이 직접 수정)
file_path = "C:/workspace/ProjectData/hrs/selected_data/10.result/final_AD_train_long_최초시점_onset.csv"

# 데이터 불러오기
df = pd.read_csv(file_path)

# 컬럼별 결측치, 유니크 수, 상위 값 등 요약
summary = []

for col in df.columns:
    total = len(df)
    missing = df[col].isnull().sum()
    missing_pct = (missing / total) * 100
    nunique = df[col].nunique(dropna=True)
    top_vals = df[col].value_counts(dropna=False).head(5).to_dict()
    top_vals_str = "; ".join([f"{k}: {v}" for k, v in top_vals.items()])

    summary.append({
        "column": col,
        "missing_count": missing,
        "missing_pct": round(missing_pct, 2),
        "unique_values": nunique,
        "top_5_values": top_vals_str
    })

# 결과를 DataFrame으로
summary_df = pd.DataFrame(summary)

# 📁 CSV로 저장
summary_df.to_csv("C:/workspace/ProjectData/hrs/selected_data/10.result/missing_summary.csv", index=False, encoding="utf-8-sig")
print("저장 완료: missing_summary.csv")
