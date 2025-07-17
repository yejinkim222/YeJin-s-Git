import pandas as pd

# 파일 경로
path = "G:/내 드라이브/코딩/Project01/data/kaggle_tadpole/TADPOLE_D1_D2.csv"

# 앞부분 몇 줄만 로드 (열 이름만)
df = pd.read_csv(path, nrows=5)

# 열 이름을 파일로 저장
with open("tadpole_columns.txt", "w", encoding="utf-8") as f:
    for col in df.columns:
        f.write(col + "\n")

print("열 이름이 'tadpole_columns.txt' 파일로 저장되었습니다.")

#TADPOLE 열 이름 확인