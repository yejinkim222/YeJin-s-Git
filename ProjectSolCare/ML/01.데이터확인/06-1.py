import pandas as pd

# 데이터 불러오기 (예진님이 사용하는 로컬 경로 그대로 사용)
df = pd.read_stata("G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta", convert_categoricals=False)

# 확인할 컬럼 그룹 정의
column_groups = {
    "memrye2": [f"r{w}memrye2" for w in range(3, 10)],
    "alzhee2": [f"r{w}alzhee2" for w in range(10, 17)],
    "demene2": [f"r{w}demene2" for w in range(10, 17)],
    "demens": [f"r{w}demens" for w in range(11, 17)],
}

# 컬럼 존재 여부 확인
for group_name, cols in column_groups.items():
    print(f"🔍 {group_name.upper()} 그룹 컬럼 존재 여부")
    for col in cols:
        if col in df.columns:
            print(f"✅ {col} ✔️")
        else:
            print(f"❌ {col} (없음)")
    print()

# 치매, mci 여부랑 연도 확인 가능한지, 
# 컬럼명 먼저 확인하는 코드
# r3memrye2(Wave 3의 2회 연속 기억 관련 질환 진단 여부) 누락
# 그래서 wave 4 이후의 데이터만 가지고 예측 모델 설계 예정