import pandas as pd

# 원본 데이터 불러오기
df = pd.read_csv("G:/내 드라이브/코딩/Project01/ML/rand_diabetes_b_option.csv")

# 당뇨병 진단 여부 변수 만들기 (1개라도 '1.has diabetes'이면 1)
has_diabetes = df[["r13dbstage", "r14dbstage", "r15dbstage", "r16dbstage"]].apply(
    lambda row: any(str(v).startswith("1.") for v in row), axis=1
)

# 경구약 복용 여부
on_oral_med = df[["r14dborlmed", "r15dborlmed", "r16dborlmed"]].apply(
    lambda row: any(v == "1.yes" for v in row), axis=1
)

# 인슐린 사용 여부
on_insulin = df[["r14dbinsuln", "r15dbinsuln", "r16dbinsuln"]].apply(
    lambda row: any(v == "1.yes" for v in row), axis=1
)

# 진단 연도 충돌 여부 (0이면 정상, 그 외는 충돌)
has_diag_conflict = df["radbdiagyf"].apply(lambda x: 1 if x != "0.no reported year conflict" else 0)

# 새로운 데이터프레임 생성
df_diabetes_flags = pd.DataFrame({
    "has_diabetes": has_diabetes.astype(int),
    "on_oral_med": on_oral_med.astype(int),
    "on_insulin": on_insulin.astype(int),
    "has_diag_conflict": has_diag_conflict
})

# 저장
output_path = "G:/내 드라이브/코딩/Project01/ML/rand_diabetes_flags.csv"
df_diabetes_flags.to_csv(output_path, index=False)
print(f"저장 완료: {output_path}")

# 당뇨인지 아닌지만 확인할수있게 데이터 저장하기