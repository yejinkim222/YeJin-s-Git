import pandas as pd

# 1. 파일 경로
path_rand = "G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"
output_path = "G:/내 드라이브/코딩/Project01/data/hrs/rand_preprocessed.dta"

# 2. 전처리할 컬럼 목록 (존재하는 컬럼만, 누락되었던 컬럼 제외)
selected_cols = [
    # ID
    'hhid', 'pn', 'hhidpn',

    # 나이, 생년
    'rabyear', 'radyear',

    # 성별
    'ragender',

    # 교육 수준
    'raedyrs', 'raedegrm',

    # 당뇨 관련
    'r1dborlmed', 'r2dborlmed', 'r3dborlmed', 'r4dborlmed', 'r5dborlmed', 'r6dborlmed',
    'r7dborlmed', 'r8dborlmed', 'r9dborlmed', 'r10dborlmed', 'r11dborlmed', 'r12dborlmed',
    'r13dbstage', 'r13dborlmed', 'r14dbstage', 'r14dborlmed', 'r15dbstage', 'r15dborlmed',
    'r16dbstage', 'r16dborlmed',

    # 고혈압
    'r14hibpe', 'r15hibpe', 'r16hibpe',

    # 우울증
    *[f"r{w}{q}" for w in range(3, 17) for q in ['depyr', 'deplos', 'deptir', 'depnoap', 'dephun', 'depsle', 'depnit', 'depcon', 'depdown', 'deptho']],

    # AD 및 MCI 관련
    'r10demene2', 'r11demene2', 'r12demene2', 'r13demene2', 'r14demene2', 'r15demene2', 'r16demene2',
    'r11demens', 'r12demens', 'r13demens', 'r14demens', 'r15demens', 'r16demens'
]

# 3. 데이터 로드
df = pd.read_stata(path_rand)

# 4. 실제 존재하는 컬럼만 필터링
existing_cols = [col for col in selected_cols if col in df.columns]
df_selected = df[existing_cols]

# 5. 저장
df_selected.to_stata(output_path, write_index=False)

# 6. 저장된 파일 확인
df_check = pd.read_stata(output_path)
print(f"✅ 저장된 컬럼 수: {len(df_check.columns)}")
print("📋 저장된 컬럼 목록:")
print(df_check.columns.tolist())

# rand에서 사용할 컬럼만 추출해서 따로 저장(모든 변수 저장 완료)