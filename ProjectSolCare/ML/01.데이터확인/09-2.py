import pandas as pd

# 데이터 불러오기
df = pd.read_stata("G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta")

# 기준이 되는 우울 관련 문항 이름 패턴
base_suffixes = [
    "dbstage", "dborlmed", "dborlmed", "radbdiagyf"
]

# 가능한 웨이브 번호 (보통 r1 ~ r16까지 있음)
waves = list(range(1, 21))

# 각 웨이브별 변수명 구성
expected_vars = []
for wave in waves:
    prefix = f"r{wave}"
    for suffix in base_suffixes:
        expected_vars.append(f"{prefix}{suffix}")

# 실제 존재하는 컬럼만 추리기
existing_vars = [col for col in expected_vars if col in df.columns]

# 결과 확인
print(f"총 추출된 당뇨 관련 변수 수: {len(existing_vars)}")
print(existing_vars)

# 당뇨 관련 증상 변수