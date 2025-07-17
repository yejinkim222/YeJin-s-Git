# 가장 최신인 2016년 데이터만 test로 쓸 때
# 비율이 적당한지 확인하기
# 연도별 샘플 수 확인
import pandas as pd

# 🔧 파일 경로 (예진님 로컬 경로로 수정)
# input_path = "C:/workspace/ProjectData/h`rs/selected_data/10.result/final_AD_train_long_최초시점_filled.csv"
input_path = "C:/workspace/ProjectData/hrs/selected_data/10.result/final_AD_train_long_치매발생자_filled.csv"
df = pd.read_csv(input_path)

year_counts = df['ad_year'].value_counts().sort_index()
print(year_counts)

# 전체 샘플 수
total = len(df)
latest_year = 2016
prev_year = 2014

# 단일 연도 test 구성
test_count_2016 = year_counts.get(latest_year, 0)
test_ratio_2016 = test_count_2016 / total

# 두 연도 test 구성
test_count_2yr = test_count_2016 + year_counts.get(prev_year, 0)
test_ratio_2yr = test_count_2yr / total

print(f"\n전체 샘플 수: {total}")

print(f"\n📌 전체 샘플 수: {total}")
print(f"🔹 2016년 샘플 수: {test_count_2016} ({test_ratio_2016:.2%})")
print(f"🔸 2014+2016년 샘플 수: {test_count_2yr} ({test_ratio_2yr:.2%})")

# 전체 샘플 수: 653
# 📌 전체 샘플 수: 653
# 🔹 2016년 샘플 수: 65 (9.95%)
# 🔸 2014+2016년 샘플 수: 123 (18.84%)
# 일단 둘 다 해 봐야지