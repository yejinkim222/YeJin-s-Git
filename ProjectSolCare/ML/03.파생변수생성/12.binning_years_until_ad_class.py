# 예측 너무 못 해서
# 
import pandas as pd
import numpy as np

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/03.AD_train_derived.csv"
df = pd.read_csv(file_path)

# 🎯 0~30년까지 3년 단위 구간 생성
bins = list(range(0, 31, 3))  # [0, 3, 6, ..., 30]
labels = list(range(len(bins) - 1))  # [0, 1, ..., 9]

# 📌 Binning: 0~30년까지만
df["ad_class_temp"] = pd.cut(
    df["years_until_ad"],
    bins=bins,
    labels=labels,
    right=False
)

# 📌 실제 값이 있는 클래스만 필터링해서 재매핑
used_classes = sorted(df["ad_class_temp"].dropna().unique())
mapping = {old_label: new_label for new_label, old_label in enumerate(used_classes)}

# 📌 정수형 class로 다시 매핑
df["years_until_ad_class"] = df["ad_class_temp"].map(mapping).astype("Int64")

# ✅ 중간 컬럼 제거 (원하면 유지해도 됨)
df.drop(columns=["ad_class_temp"], inplace=True)

# ✅ 저장
save_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/04.AD_train_years_until_ad_class.csv"
df.to_csv(save_path, index=False)
print(f"✅ 저장 완료: {save_path}")

# 📌 확인용 출력
print(f"✅ 유효 클래스 수: {len(mapping)}개")
print("📌 클래스 매핑:", mapping)
print("\n✅ Sample data:")
print(df[["years_until_ad", "years_until_ad_class"]].head())
