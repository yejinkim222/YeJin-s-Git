import pandas as pd

# 파일 경로
file_path = "G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"

# 데이터 불러오기 (열 이름만)
df = pd.read_stata(file_path, convert_categoricals=False)

# 예진님이 확인하고 싶은 변수 리스트 (변수명 기준)
wanted_vars = [
    'HHIDPN', 'HHID', 'PN',                # 병합 키
    'rahage', 'rahgender', 'raeduc',       # 주요 변수 (실제 존재하는지 확인)
    'r8agey', 'r8gender', 'r8educ',        # 다른 wave 기반 변수 후보
    'ragender', 'raagey', 'raedyrs',       # 다른 레이블 가능성
    'radyear',                             # 출생년도 관련
    'rahchdif', 'rahchbp',                 # 건강 상태 (고혈압, 당뇨 여부 추정용 예시)
    'cesd', 'cesdscore', 'cesdsum',        # 우울 관련 점수들
]

# 실제로 존재하는 변수만 추출
existing_vars = [var for var in wanted_vars if var in df.columns]

# 필터링된 데이터프레임 만들기
df_filtered = df[existing_vars].copy()

# 모든 행이 결측치인 변수 제거
df_nonempty = df_filtered.dropna(axis=1, how='all')

# 행 기준으로 완전히 결측치인 경우 제거 (선택사항)
df_nonempty = df_nonempty.dropna(how='all')

# 결과 저장
output_path = "G:/내 드라이브/코딩/Project01/data/hrs/filtered_rand_data.csv"
df_nonempty.to_csv(output_path, index=False)

print(f"저장 완료: {output_path}")
print(f"사용된 변수: {list(df_nonempty.columns)}")
