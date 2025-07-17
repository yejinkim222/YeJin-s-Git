import pandas as pd

# RAND HRS 데이터 로드
df = pd.read_stata("G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta", convert_categoricals=False)

# 최신 wave 순서대로 고혈압 변수 지정
bp_vars = ['r14hibpe', 'r15hibpe', 'r16hibpe']  # 최신부터 순서대로

# 가장 먼저 유효한 응답 사용
df['hibpe_latest'] = df[bp_vars].bfill(axis=1).iloc[:, 0]

# 고혈압 여부 변수 생성 (1=고혈압 있음, 0=없음, 나머지는 결측 처리)
df['has_hypertension'] = df['hibpe_latest'].apply(lambda x: 1 if x == 1 else (0 if x == 0 else pd.NA))

# 요약 출력
print(f"총 샘플 수: {len(df)}")
print("has_hypertension 분포:")
print(df['has_hypertension'].value_counts(dropna=False))

# 저장
df[['has_hypertension']].to_csv("rand_hypertension_final.csv", index=False)

# 고혈압데이터 최종