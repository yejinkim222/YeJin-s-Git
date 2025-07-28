import pandas as pd

# 파일 경로
cog_path = "G:/내 드라이브/코딩/Project01/data/hrs/cogfinalimp_9520wide/cogfinalimp_9520wide.dta"
rand_path = "G:/내 드라이브/코딩/Project01/data/hrs/randhrs1992_2022v1/randhrs1992_2022v1.dta"

# 데이터 불러오기
df_cog = pd.read_stata(cog_path)
df_rand = pd.read_stata(rand_path)

# RAND에서 hhidpn 생성
df_rand["hhid"] = df_rand["hhid"].astype(str)
df_rand["pn"] = df_rand["pn"].astype(str)
df_rand["hhidpn"] = df_rand["hhid"] + df_rand["pn"]

# 필요한 변수만 필터링 (입력변수 및 ID만)
main_vars = [
    "hhidpn",        # 병합 키
    "r8agey",        # 나이
    "r8gender",      # 성별
    "r8educ",        # 교육 수준
    "r8diab",        # 당뇨 여부
    "r8hearte",      # 고혈압/심장질환 여부
    "r8cesd"         # 우울증 점수
]
df_rand_filtered = df_rand[main_vars].copy()

# COG와 병합
df_merged = pd.merge(df_cog, df_rand_filtered, on="hhidpn", how="inner")

# 결과 확인
print("병합된 데이터 크기:", df_merged.shape)
print("앞부분 미리보기:")
print(df_merged.head())

#데이터 병합하기
# 나이, 성별, 교육수준, 당뇨 여부, 고혈압 여부, 우울증 여부, 병합 키

###
#나이에 관련된 변수명 먼저 찾아줘.
# 실제로 들어있는지도 같이 확인하자.
#  내가 보내준 질문에서 확인해보고, 
# 실제로 존재하는 변수명을 찾아주면 
# 내가 거기 어떤 데이터 있는지 
# 그걸로만 이루어진 데이터 만들 수 있게 니가 나한테 코드를 줘. 
# 앞의 10개만 추리는 식으로 확인하지 말고, 
# 전체를 다 확인해서 안에 데이터가 실제로 없으면 삭제하고, 
# 남은 부분만 확인 가능하게끔 코드 짜줘.
###