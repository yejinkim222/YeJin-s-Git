# 모델성능 최고로 나왔던거...
# 일단 여기 적어놓고 파생변수 생성하는거부터 다시 정리해놔야지
# 일단 이거는 발표용...
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# 데이터 로딩
# 예진님이 이미 df = pd.read_csv(...) 형태로 불러온 상태라고 가정
input_path = "C:/workspace/Project01/data/hrs/selected_data/10.result/final_AD_train_long_최초시점_filled_derived.csv"
df = pd.read_csv(input_path)

# ✅ 나이 40세 이상만 필터링
df = df[df['age'] >= 60]

# ✅ 타겟 변수 정의
y = df["years_until_ad"]

# ✅ 피처 선택 (ad_year, hhid_year는 제외)
X = df.drop(columns=["years_until_ad", "ad_year", "hhid_year"])

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 결과 저장용
results = []

# ✅ max_depth 1~20 실험
for depth in range(1, 21):
    model = RandomForestRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    results.append({'max_depth': depth, 'r2_score': r2})

# 결과 출력
results_df = pd.DataFrame(results)
print(results_df)

# 결과
#     max_depth  r2_score
# 0           1  0.384985
# 1           2  0.554007
# 2           3  0.598320
# 3           4  0.614861
# 4           5  0.622430
# 5           6  0.628698
# 6           7  0.625968
# 7           8  0.632505
# 8           9  0.635320
# 9          10  0.633772
# 10         11  0.633014
# 11         12  0.630296
# 12         13  0.633313
# 13         14  0.634390
# 14         15  0.631869
# 15         16  0.632340
# 16         17  0.631545
# 17         18  0.632730
# 18         19  0.632163
# 19         20  0.632163
