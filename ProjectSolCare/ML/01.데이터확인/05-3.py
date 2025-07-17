import pandas as pd

# 1. 데이터 경로 (사용자가 직접 입력)
path = "G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"

# 2. 사용할 wave 목록과 변수 정의
waves = range(9, 15)
base_vars = ["depyr", "deplos", "deptir", "depnoap", "dephun", "depsle", "depnit", "depcon", "depdown", "deptho"]
depression_vars_all = [f"r{wave}{var}" for wave in waves for var in base_vars]

# 3. 데이터 불러오기 (필요 변수만)
df = pd.read_stata(path, columns=depression_vars_all)

# 4. 각 wave별 우울증 판단 로직
for wave in waves:
    prefix = f"r{wave}"
    depyr = f"{prefix}depyr"
    symptoms = [f"{prefix}{v}" for v in base_vars[1:]]
    symptom_count_col = f"{prefix}_symptom_count"
    is_depressed_col = f"{prefix}_is_depressed"

    df[symptom_count_col] = df[symptoms].apply(lambda row: sum(row == 1), axis=1)

    # 우울증 판단: depyr == 1 or 주요 증상 2개 이상
    df[is_depressed_col] = df.apply(
        lambda row: 1 if (row.get(depyr) == 1 or row[symptom_count_col] >= 2) else 0,
        axis=1
    )

# 5. 어떤 wave든 우울증 해당되면 최종 라벨
depression_flags = [f"r{wave}_is_depressed" for wave in waves]
df["is_depressed_anywave"] = df[depression_flags].max(axis=1)

# 6. 통계 확인
print("총 샘플 수:", len(df))
print("우울증 있음:", df["is_depressed_anywave"].sum())
print("우울증 없음:", (df["is_depressed_anywave"] == 0).sum())

# 7. CSV 저장
df.to_csv("rand_depression_combined.csv", index=False)

# wave 9~14 모두 포함해서 신뢰도 높을듯
# 근데 우울증인 사람이 없대...
# 그냥 현재기준으로 가고 수 늘려야겠다