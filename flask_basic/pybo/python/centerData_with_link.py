# 9슬라이드 8번 슬라이드 이어서 시작 



# import pandas as pd
# import requests
# import json
# import os
# import time



# # 카카오 앱키에 내 레스트 앱키 넣고
# KAKAO_API_KEY = '17496727a0f42174aa5d45573137efcb'
# headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}


# # 데이터정리한 주소 위도 경도를 검색해주고
# def get_coords(address):
#     url = "https://dapi.kakao.com/v2/local/search/address.json"
#     params = {"query": address}
#     res = requests.get(url, headers=headers, params=params)
#     try:
#         result = res.json()["documents"][0]
#         return result["y"], result["x"]
#     except:
#         return None, None


# # 장소 이름을 장소 url로 검색
# def get_place_url(query, x=None, y=None):
#     url = "https://dapi.kakao.com/v2/local/search/keyword.json"
#     params = {"query": query}
#     if x and y:
#         params.update({"x": x, "y": y, "radius": 2000})
#     res = requests.get(url, headers=headers, params=params)
#     try:
#         return res.json()["documents"][0]["place_url"]
#     except:
#         return "#"



# # 파일 경로 출력하기
# input_path = "C:/Users/user/Desktop/프로젝트 데이터/치매센터현황_링크사용01.xlsx"
# df = pd.read_excel(input_path)
# print(df.head())

# output_path = "../data/centerData_with_link.js"


# os.makedirs("../data", exist_ok=True)

# output = []

# for i, row in df.iterrows():
#     name = str(row["센터 이름"]).strip()
#     address = str(row["주소"]).strip()
#     phone = str(row["전화번호"]).strip()
#     region = str(row["지역"]).strip()
#     center_type = row["센터 구분"]

#     if not address.startswith("서울"):
#         address = "서울특별시 " + address

#     lat, lng = get_coords(address)
#     if lat and lng:
#         place_url = get_place_url(name, x=lng, y=lat)
#         output.append({
#             "region": region,
#             "type": "치매" if "치매" in center_type else "복지",
#             "name": name,
#             "address": address,
#             "phone": phone,
#             "lat": float(lat),
#             "lng": float(lng),
#             "link": place_url
#         })

#         print(f"✅ [{i}] {name} → 링크 완료")
#         time.sleep(0.3)  # 카카오 API 과다 호출 방지하기위해 
#     else:
#         print(f"❌ [{i}] {name} → 주소 실패")

# # 저장과 함께 센터링크 생성되면 깨지지않게, 완료가 터미널에 뜨게 
# with open(output_path, "w", encoding="utf-8") as f:
#     f.write("const centerData = ")
#     json.dump(output, f, ensure_ascii=False, indent=2)
#     f.write(";")

# print("\n🎉 centerData_with_link.js 생성 완료!")


import pandas as pd
import requests
import json
import os
import time

KAKAO_API_KEY = '17496727a0f42174aa5d45573137efcb'
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}

# 저장 경로 설정
os.makedirs("../data", exist_ok=True)

# 엑셀 파일 경로
excel_path = "치매센터현황_link01.xlsx"
df = pd.read_excel(excel_path)

output = []

# 주소를 위도/경도로 변환
def get_coords(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    params = {"query": address}
    res = requests.get(url, headers=headers, params=params)
    try:
        result = res.json()["documents"][0]
        return result["y"], result["x"]
    except:
        return None, None

# 센터명으로 장소 검색 후 place_url 가져오기
def get_place_url(name):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    params = {"query": name}
    res = requests.get(url, headers=headers, params=params)
    try:
        result = res.json()["documents"][0]
        return result["place_url"]
    except:
        return "#"

# 데이터 정제 및 생성
for _, row in df.iterrows():
    name = str(row["센터 이름"]).strip()
    address = str(row["주소"]).strip()
    phone = str(row["전화번호"]).strip()
    region = str(row["지역"]).strip()
    center_type = str(row["센터 구분"]).strip()

    if not address.startswith("서울"):
        address = "서울특별시 " + address

    lat, lng = get_coords(address)
    if not lat or not lng:
        print(f"[X] 주소 변환 실패: {address}")
        continue

    link = get_place_url(name)
    time.sleep(0.2)  # 카카오 API 호출 제한 회피 (초당 10건)

    output.append({
        "region": region,
        "type": "치매" if "치매" in center_type else "복지",
        "name": name,
        "address": address,
        "phone": phone,
        "lat": float(lat),
        "lng": float(lng),
        "link": link
    })

# JS 파일로 저장하기
with open("../data/centerData_with_link.js", "w", encoding="utf-8") as f:
    f.write("const centerData = ")
    json.dump(output, f, ensure_ascii=False, indent=2)
    f.write(";")

print(" centerData_with_link.js 생성 완료!")

