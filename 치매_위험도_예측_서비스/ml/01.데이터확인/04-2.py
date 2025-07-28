import pandas as pd

# 원본 randhrs 데이터 불러오기
file_path = "G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"
df = pd.read_stata(file_path)

# 사용할 변수
selected_vars = ['rahibpef']

# 해당 변수만 필터링 (실제 존재하는지 확인)
existing_vars = [var for var in selected_vars if var in df.columns]
df_bp = df[existing_vars].copy()

# 고혈압 여부 파생 변수 생성
df_bp['has_hypertension'] = df_bp['rahibpef'].apply(
    lambda x: 1 if x == 1 else (0 if x == 0 else pd.NA)
)

# 결측치 제거
df_clean = df_bp[['has_hypertension']].dropna().copy()

# 결과 출력
print(f"총 샘플 수: {len(df)}")
print(f"유효한 고혈압 데이터 수: {len(df_clean)}")
print(df_clean['has_hypertension'].value_counts())

# 저장
df_clean.to_csv("G:/내 드라이브/코딩/Project01/ML/rand_hypertension_final.csv", index=False)
print("✅ 저장 완료: rand_hypertension_final.csv")

print(df['rahibpef'].value_counts(dropna=False))

# 고혈압 변수 사용가능한지 확인용