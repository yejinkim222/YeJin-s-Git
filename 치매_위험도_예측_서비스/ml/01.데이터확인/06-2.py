import pandas as pd

df = pd.read_stata("G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta", convert_categoricals=False)

# 확인할 컬럼 그룹 정의
column_groups = {
    "memrye2": [f"r{w}memrye2" for w in range(4, 10)],
    "alzhee2": [f"r{w}alzhee2" for w in range(10, 17)],
    "demene2": [f"r{w}demene2" for w in range(10, 17)],
    "demens": [f"r{w}demens" for w in range(11, 17)],
}

# 값 분포 확인
for group_name, cols in column_groups.items():
    print(f"📊 {group_name.upper()} 값 분포 확인")
    for col in cols:
        if col in df.columns:
            print(f"🔹 {col}:")
            print(df[col].value_counts(dropna=False))
        else:
            print(f"❌ {col} (컬럼 없음)")
    print()

# 값이 실제로 있는지 확인