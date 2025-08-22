import json
from db_utils import save_to_db
from input_handler import preprocess_input
from model_predict import load_model, predict, get_feature_importance
from result_summary import summarize_input
from result_save import save_results_to_file  
from sklearn.metrics import accuracy_score


# 1. 입력 불러오기
with open("python/data/input.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 전처리 + 예측
df, processed = preprocess_input(data)
model = load_model()
pred, proba = predict(df, model)
importance = get_feature_importance(model)

print("✅ [DEBUG] 전처리된 입력값:")
print(df)

print("\n✅ [DEBUG] 예측 확률:")
print(model.predict_proba(df))

print("\n✅ [DEBUG] 모델 Feature 순서:")
print(model.get_booster().feature_names)

print("\n✅ [DEBUG] 입력 Feature 순서:")
print(df.columns.tolist())


# 3. 결과 생성
summary = summarize_input(processed)
result = {
    "name": str(data.get("name", "")),
    "risk_score": round(proba * 100, 2),
    "risk_level": "높음" if proba >= 0.75 else "보통" if proba >= 0.4 else "낮음",
    "prediction": int(pred),
    "input_summary": summary
}

# # 4. 저장 
# save_results_to_file(result, importance)       
# save_to_db(result, data)                       
