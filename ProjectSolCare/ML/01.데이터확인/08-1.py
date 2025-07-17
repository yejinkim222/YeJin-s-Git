import pandas as pd

# rand 데이터 불러오기
df_rand = pd.read_stata("G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta")

# 예진님이 알려준 기준 컬럼명
expected_cols = ["HHID", "PN", "HHIDPN", "RAHHIDPN"] + \
                [f"H{i}HHID" for i in range(1, 17)] + \
                [f"H{i}HHIDC" for i in range(1, 17)]

# 실제 컬럼 이름을 모두 소문자로 변환
actual_cols_lower = [col.lower() for col in df_rand.columns]

# 비교 결과 출력
print("🔍 rand 데이터 내 컬럼 존재 여부 (대소문자 무시):\n")
for col in expected_cols:
    if col.lower() in actual_cols_lower:
        print(f"{col} → 존재함")
    else:
        print(f"{col} → 없음")

# rand 데이터 id컬럼들 확인