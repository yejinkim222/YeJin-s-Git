# 우울증, ad, mci가 학습 가능한 정도인지 확인
# 확인용
import pandas as pd

# 데이터 로드
file_path = "C:/workspace/ProjectData/hrs/selected_data/rand_filtered.csv"
rand = pd.read_csv(file_path)

# 1. 우울증 판단 (depyr, deplos, deptir, depnoap, depsle, depnit, depdown은 "1.yes"이면 1점, 나머지 변수들은 "0.no"이면 1점)
depression_columns_yes = (
    [f"r{i}depyr" for i in range(3, 17)]
    + [f"r{i}deplos" for i in range(3, 17)]
    + [f"r{i}deptir" for i in range(3, 17)]
    + [f"r{i}depnoap" for i in range(3, 17)]
    + [f"r{i}dephun" for i in range(3, 17)]
    + [f"r{i}depsle" for i in range(3, 17)]
    + [f"r{i}depnit" for i in range(3, 17)]
    + [f"r{i}depcon" for i in range(3, 17)]
    + [f"r{i}depdown" for i in range(3, 17)]
    + [f"r{i}deptho" for i in range(3, 17)]
)

depression_columns_no = []


# 우울증 변수 계산
def calculate_depression(row):
    score = 0
    # Dep variables that should be 1 for each answer (yes = "1.yes")
    for col in depression_columns_yes:  # First 7 are yes = 1
        if row[col] == "1.yes":
            score += 1
    # The rest should be 0 = no (no = "0.no" becomes 1 point)
    return 1 if score >= 4 else 0


rand["depression"] = rand.apply(calculate_depression, axis=1)


# 2. AD / MCI 판단
def calculate_ad_mci(row):
    # AD condition: "1.yes-has/had cond"
    if (
        row["r10demene2"] == "1.yes-has/had cond"
        or row["r10alzhee2"] == "1.yes-has/had cond"
    ):
        return "AD"

    # MCI condition: "1.yes-has/had cond" (for r4memrye2 to r9memrye2)
    for i in range(4, 10):  # Check from r4memrye2 to r9memrye2
        if row[f"r{i}memrye2"] == "1.yes-has/had cond":
            return "MCI"

    # Default to 'Normal' if no AD or MCI
    return "Normal"


rand["AD_MCI_status"] = rand.apply(calculate_ad_mci, axis=1)

# 3. 결측치 처리 (미응답은 missed로 처리)
rand.fillna("missed", inplace=True)

# 4. 한 사람의 모든 wave에서 값이 missed인 경우 해당 row 삭제
wave_columns = (
    [f"r{i}dborlmed" for i in range(1, 17)]
    + [f"r{i}hibpe" for i in range(14, 17)]
    + [f"r{i}depyr" for i in range(3, 17)]
    + [f"r{i}deplos" for i in range(3, 17)]
    + [f"r{i}deptir" for i in range(3, 17)]
    + [f"r{i}depnoap" for i in range(3, 17)]
    + [f"r{i}dephun" for i in range(3, 17)]
    + [f"r{i}depsle" for i in range(3, 17)]
    + [f"r{i}depnit" for i in range(3, 17)]
    + [f"r{i}depcon" for i in range(3, 17)]
    + [f"r{i}depdown" for i in range(3, 17)]
    + [f"r{i}deptho" for i in range(3, 17)]
    + [f"r{w}demene2" for w in range(10, 17)]
    + [f"r{w}alzhee2" for w in range(10, 17)]
    + [f"r{w}memrye2" for w in range(4, 10)]
    + [f"r{w}demens" for w in range(11, 17)]
)

rand = rand.dropna(subset=wave_columns, how="all")

# 5. 비율 및 인원 계산
depression_count = rand["depression"].value_counts()
ad_mci_count = rand["AD_MCI_status"].value_counts()

# 우울증 유무 비율 출력
print("우울증 유무 비율 및 인원:")
print(depression_count)
print(depression_count / len(rand))  # 비율

# AD, MCI, 정상 비율 출력
print("\nAD, MCI, 정상 비율 및 인원:")
print(ad_mci_count)
print(ad_mci_count / len(rand))  # 비율
