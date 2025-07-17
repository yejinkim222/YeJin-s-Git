# -1로 채워넣은 뒤 결측 확인
import pandas as pd

# 파일 경로 (예진님이 직접 수정)
file_path = "C:/workspace/ProjectData/hrs/selected_data/11.ml_start/01.AD_train_long_filled.csv"
df = pd.read_csv(file_path)

# 결과 저장용 리스트
missing_summary = []

# 전체 row 수
total_rows = len(df)

# 컬럼별로 -1 개수 및 비율 계산
for col in df.columns:
    try:
        count_minus1 = (df[col] == -1).sum()
        ratio = count_minus1 / total_rows * 100
        if count_minus1 > 0:
            missing_summary.append({
                'column': col,
                'minus1_count': count_minus1,
                'minus1_pct': round(ratio, 2)
            })
    except:
        # 숫자형이 아닌 경우 비교 불가능할 수 있음
        continue

# 결과 정리
result_df = pd.DataFrame(missing_summary)
result_df.sort_values(by='minus1_pct', ascending=False, inplace=True)

# 보기 좋게 출력
print("-1을 결측치로 간주한 컬럼별 비율:")
print(result_df)

# 결과
#               column  minus1_count  minus1_pct
# 5  years_until_hibpe           643       98.47
# 1          has_hibpe           542       83.00
# 3    years_until_mci           530       81.16
# 4     years_until_db            64        9.80
# 0            edu_yrs             2        0.31
# 2          edu_level             2        0.31
