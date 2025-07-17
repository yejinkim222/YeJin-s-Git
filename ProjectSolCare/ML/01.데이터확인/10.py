import pandas as pd

# 경로 설정
path_rand = r"G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"
df_rand = pd.read_stata(path_rand)

# 사용할 컬럼 리스트 (예진님 입력 그대로 복사)
selected_columns = [
    # ID
    "hhid", "pn", "hhidpn",
    # 나이
    "ragey_e", "raage", "rabyear", "radyear",
    # 성별
    "ragender", "gender",
    # 교육 수준
    "raedyrs", "raedegrm",
    # 당뇨
    "r1dborlmed", "r2dborlmed", "r3dborlmed", "r4dborlmed", "r5dborlmed", "r6dborlmed",
    "r7dborlmed", "r8dborlmed", "r9dborlmed", "r10dborlmed", "r11dborlmed", "r12dborlmed",
    "r13dbstage", "r13dborlmed", "r14dbstage", "r14dborlmed", "r15dbstage", "r15dborlmed",
    "r16dbstage", "r16dborlmed",
    # 고혈압
    "r14hibpe", "r15hibpe", "r16hibpe",
    # 우울증
    *[f"r{w}{suf}" for w in range(3, 17) for suf in ["depyr", "deplos", "deptir", "depnoap", "dephun", "depsle", "depnit", "depcon", "depdown", "deptho"]],
    # ad/mci 관련
    *[f"r{w}demene2" for w in range(10, 17)],
    *[f"r{w}alzhee2" for w in range(4, 17)],
    *[f"r{w}memrye2" for w in range(4, 17)],
    *[f"r{w}demens" for w in range(11, 17)]
]

# 컬럼 존재 여부 확인
df_columns_lower = set(col.lower() for col in df_rand.columns)
existing_columns = [col for col in selected_columns if col.lower() in df_columns_lower]
missing_columns = [col for col in selected_columns if col.lower() not in df_columns_lower]

# 존재하는 컬럼의 결측률 계산
missing_info = df_rand[existing_columns].isnull().mean().sort_values(ascending=False) * 100
missing_info = missing_info.round(2)

# 결과 출력
print("존재하는 컬럼 개수:", len(existing_columns))
print("존재하지 않는 컬럼 개수:", len(missing_columns))
print("\n존재하지 않는 컬럼 목록:\n", missing_columns)
print("\n존재하는 컬럼들의 결측 비율 (%):\n", missing_info)

# rand 변수 다시 확인