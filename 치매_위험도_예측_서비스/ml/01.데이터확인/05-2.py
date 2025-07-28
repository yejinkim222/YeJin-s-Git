import pandas as pd

# 경로 설정
path = "G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"

# 우울 관련 변수 (Wave 14 기준)
depression_vars = [
    "r14depyr", "r14deplos", "r14deptir", "r14depnoap", "r14dephun",
    "r14depsle", "r14depnit", "r14depcon", "r14depdown", "r14deptho"
]

# 데이터 불러오기
df = pd.read_stata(path, columns=depression_vars)

# 문자열을 숫자로 매핑 (예: '1.yes' → 1, '0.no' → 0)
map_dict = {"1.yes": 1, "0.no": 0}
df = df.applymap(lambda x: map_dict.get(x, x))  # 다른 값은 그대로 두기

# 주요 증상 변수만 따로 리스트로 저장
symptom_vars = depression_vars[1:]  # r14depyr 제외

# 증상 개수 계산: 주요 증상 중 1인 값의 개수 (결측치 무시)
df["depression_symptom_count"] = df[symptom_vars].apply(lambda row: (row == 1).sum(), axis=1)

# 완화된 우울증 기준: r14depyr == 1 또는 증상 2개 이상
df["is_depressed_relaxed"] = df.apply(
    lambda row: 1 if (row["r14depyr"] == 1 or row["depression_symptom_count"] >= 2) else 0,
    axis=1
)

# 기본 통계 출력
print(f"🔢 총 샘플 수: {len(df)}\n")

# 결측치 비율 확인
missing = df.isna().mean().round(4) * 100
print("📌 결측치 비율 (%):")
print(missing)
print()

# 우울증 기준 통과 인원 확인
depyr_count = (df["r14depyr"] == 1).sum()
symptom2_count = (df["depression_symptom_count"] >= 2).sum()
either_count = ((df["r14depyr"] == 1) | (df["depression_symptom_count"] >= 2)).sum()

print("📊 개별 조건 충족 인원:")
print(f"r14depyr == 1: {depyr_count}")
print(f"증상 2개 이상: {symptom2_count}")
print(f"둘 중 하나 이상 해당 (우울증으로 판단): {either_count}\n")

# 우울증 여부 분포 확인
print("📊 완화 기준 우울증 여부 분포:")
print(df["is_depressed_relaxed"].value_counts())

# r14depyr == 1 (우울감 호소) 또는
# 주요 증상 중 2개 이상 해당하면 우울증으로 판단
# 완화된 우울증 여부로 학습 가능할지 확인하기