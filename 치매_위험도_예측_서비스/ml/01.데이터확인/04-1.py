import pandas as pd

# rand 데이터 로드
file_path = "G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"
df = pd.read_stata(file_path, convert_categoricals=True)

# 사용할 변수 목록
hypertension_vars = ['rahibpe', 'rahibpef']  # 소문자 주의!

# 존재하는 변수만 추출
available_vars = [var for var in hypertension_vars if var in df.columns]
df_hyper = df[available_vars].copy()

# 값이 모두 결측인 열은 삭제
df_hyper.dropna(axis=1, how='all', inplace=True)

# 결측치가 아닌 값이 1개 이상 있는 행만 남김
df_hyper = df_hyper.dropna(how='all')

# 저장
save_path = "G:/내 드라이브/코딩/Project01/ML/rand_hypertension_option.csv"
df_hyper.to_csv(save_path, index=False)
print(f"✅ 저장 완료: {save_path}")

# 고혈압 변수 내부 데이터 확인용