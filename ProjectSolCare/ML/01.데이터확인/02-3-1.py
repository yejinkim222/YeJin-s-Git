import pandas as pd

# 파일 경로
path = "G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"

# 데이터프레임 불러오기
df = pd.read_stata(path)

# 열 이름 저장
with open("rand_columns.txt", "w", encoding="utf-8") as f:
    for col in df.columns:
        f.write(col + "\n")

print("열 이름이 rand_columns.txt로 저장되었습니다.")

#randhrs1992_2022v1.dta 컬럼명 파일로 저장하기