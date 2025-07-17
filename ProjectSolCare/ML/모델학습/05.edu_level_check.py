import pandas as pd

# ✅ 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_for_lgbm.csv"
df = pd.read_csv(file_path)

# ✅ edu_level 컬럼의 고유값 확인
unique_values = df["edu_level"].dropna().unique()
print("🎓 edu_level 고유값 목록:", sorted(unique_values))
