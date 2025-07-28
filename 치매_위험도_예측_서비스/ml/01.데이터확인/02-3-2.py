import os

# 설정
input_path = "G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/rand_columns.txt"
output_folder = "G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/split_columns"
split_size = 100  # 몇 개씩 자를지

# 출력 폴더 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 열 이름 불러오기
with open(input_path, "r", encoding="utf-8") as f:
    columns = [line.strip() for line in f if line.strip()]

# 자르고 저장
total_parts = (len(columns) + split_size - 1) // split_size

for i in range(total_parts):
    part_cols = columns[i * split_size:(i + 1) * split_size]
    output_path = os.path.join(output_folder, f"rand_columns_part{i+1}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for col in part_cols:
            f.write(col + "\n")

print(f"✅ 총 {total_parts}개 파일이 {output_folder}에 저장되었습니다.")
