import pandas as pd

# 파일 경로 설정
file_path = "G:/내 드라이브/코딩/Project01/ML/rand_diabetes_flags.csv"

# CSV 파일 읽기
df = pd.read_csv(file_path)

# 총 샘플 수
print(f"🔢 총 샘플 수: {len(df):,}")

# 변수별 결측 비율 확인
missing_percent = df.isnull().mean() * 100
print("\n📌 결측치 비율 (%):")
print(missing_percent.sort_values(ascending=False).round(1))

# 각 변수별 고유값 분포 확인 (5개 이하만)
print("\n📊 변수별 값 분포 (5개 이하):")
for col in df.columns:
    vc = df[col].value_counts(dropna=False)
    if len(vc) <= 5:
        print(f"\n📊 {col} 분포:")
        print(vc)

# 결측치 비율 90% 넘는 열만 따로 보기
high_missing = missing_percent[missing_percent > 90]
print("\n결측치가 90% 넘는 열:")
print(high_missing)

# 모든 열 결측치 제외한 행 수
df_cleaned = df.dropna()
print(f"\n🧼 결측치 없는 행 수: {len(df_cleaned):,}")
