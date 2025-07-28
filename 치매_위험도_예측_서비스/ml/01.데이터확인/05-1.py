import pandas as pd

# 경로 설정 (사용자 환경에 맞게 조정)
path = "G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"

# 불러올 우울 관련 변수 목록 (wave 14 기준)
depression_vars = [
    "r14depyr", "r14deplos", "r14deptir", "r14depnoap", "r14dephun",
    "r14depsle", "r14depnit", "r14depcon", "r14depdown", "r14deptho"
]

# 데이터 불러오기
df = pd.read_stata(path, columns=depression_vars)

# 증상 개수 계산 (결측치 제외하고 합산)
symptom_vars = depression_vars[1:]  # r14depyr 제외
df["depression_symptom_count"] = df[symptom_vars].apply(lambda row: sum(row == 1), axis=1)

# 완화된 기준: r14depyr == 1 또는 증상 >= 2개
df["is_depressed_relaxed"] = df.apply(
    lambda row: 1 if (row["r14depyr"] == 1 or row["depression_symptom_count"] >= 2) else 0,
    axis=1
)

# 결측치 비율 확인
missing_ratio = df.isna().mean() * 100

# 완화 기준 분포 확인
label_dist = df["is_depressed_relaxed"].value_counts(dropna=False)

print(df["r14depyr"].value_counts(dropna=False))

# 우울증 여부 데이터 확인
# 그냥 우울증인 사람은 없어서
# 1년내내 우울감 느낀 사람 수를 찾기 위한 코드