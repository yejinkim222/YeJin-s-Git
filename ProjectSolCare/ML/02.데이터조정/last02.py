# 필요한 변수만 저장
import pandas as pd

# 데이터 로드 (경로 설정 후 사용)
rand = pd.read_stata("C:/workspace/ProjectData/hrs/selected_data/rand_filtered.dta")  # 실제 경로로 변경

# 주어진 컬럼 목록
rand_columns = {
    "id": ["hhid", "pn", "hhidpn"],
    "나이": ["rabyear", "radyear"],
    "성별": ["ragender"],  # gender는 제거된 것으로 처리
    "교육 수준": ["raedyrs", "raedegrm"],
    "당뇨": [f"r{i}dborlmed" for i in range(1, 17)],
    "고혈압": [f"r{i}hibpe" for i in range(14, 17)],
    # "우울증": [f"r{i}depyr" for i in range(3, 17)] + 
    #           [f"r{i}deplos" for i in range(3, 17)] + 
    #           [f"r{i}deptir" for i in range(3, 17)] + 
    #           [f"r{i}depnoap" for i in range(3, 17)] + 
    #           [f"r{i}dephun" for i in range(3, 17)] + 
    #           [f"r{i}depsle" for i in range(3, 17)] + 
    #           [f"r{i}depnit" for i in range(3, 17)] + 
    #           [f"r{i}depcon" for i in range(3, 17)] + 
    #           [f"r{i}depdown" for i in range(3, 17)] + 
    #           [f"r{i}deptho" for i in range(3, 17)],
    "AD, MCI": [f"r{w}demene2" for w in range(10, 17)] + 
               [f"r{w}alzhee2" for w in range(10, 17)] + 
               [f"r{w}memrye2" for w in range(4, 10)],
    "치매 onset 시점": [f"r{w}demens" for w in range(11, 17)]
}

# 원하는 컬럼만 추출
selected_columns = [col for sublist in rand_columns.values() for col in sublist]
rand_filtered = rand[selected_columns]

# CSV로 저장 (결측치 처리 전에 저장)
output_file = 'C:/workspace/ProjectData/hrs/selected_data/rand_filtered.csv'  # 원하는 파일 경로로 변경
rand_filtered.to_csv(output_file, index=False)

print(f"파일이 저장되었습니다: {output_file}")
