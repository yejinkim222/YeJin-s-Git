# 하이퍼파라미터 조정 잘 안 돼서
# 일단 파생변수 추가한 버전 저장부터 하기
import pandas as pd

# ✅ 파일 경로
file_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_cog_flag.csv"
output_path = "C:/workspace/Project01/data/hrs/selected_data/12.new_data/AD_train_xgboost_best_features.csv"

# ✅ 데이터 로딩
df = pd.read_csv(file_path)

# ✅ 파생변수 생성 (컬럼 존재 여부 검증은 선택 사항)
if 'age' in df.columns and 'edu_level' in df.columns:
    df["age_x_edu"] = df["age"] * df["edu_level"]
    df["edu_level_squared"] = df["edu_level"] ** 2
    df["edu_level_bucket"] = pd.cut(
        df["edu_level"],
        bins=[-1, 1, 2, 3],
        labels=[0, 1, 2],
        include_lowest=True
    ).astype("Int64")  # NaN 허용

# ✅ 저장
df.to_csv(output_path, index=False)
print(f"✅ 저장 완료: {output_path}")
