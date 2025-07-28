import pandas as pd
import joblib
from python.predict_module import preprocess_input

# ✅ 예진님이 테스트하고 싶은 입력
raw_input = {
    "age": 75,
    "education": 12,
    "gender": 0,
    "db": 1,
    "hibpe": 0,
    "cog_input_mode": "mmse",
    "mmse_score": 22,
    "has_mci": None,
    "period": 10
}

# ✅ 전처리
input_dict = preprocess_input(raw_input)
df = pd.DataFrame([input_dict])

# ✅ 모델 불러오기 & 예측
model = joblib.load("C:/workspace/Project01/model_storage/xgb_best_model_final2.pkl")
y_pred = float(model.predict(df)[0])  # 예측된 발병 시점 (년 단위)
print(f"예측된 치매 발생 시점: {y_pred:.2f}년")
