import pandas as pd
import joblib

label_encoders = joblib.load("python/LABEL_ENCODERS.pkl")

def safe_float(x):
    try:
        return float(x)
    except:
        return -1

def preprocess_input(data):
    data = {k.lower(): v for k, v in data.items()}

    # ✅ LabelEncoder를 사용하는 컬럼 처리
    def encode_with_le(column):
        val = str(data.get(column, "")).lower()
        le = label_encoders.get(column)
        if le is None or val not in le.classes_:
            return -1
        return int(le.transform([val])[0])

    base_features = {
        "diabetic": safe_float(data.get("diabetic", -1)),
        "heartrate": safe_float(data.get("heartrate", -1)),
        "bloodoxygenlevel": safe_float(data.get("bloodoxygenlevel", -1)),
        "weight": safe_float(data.get("weight", -1)),
        "age": safe_float(data.get("age", -1)),
        "education_level": encode_with_le("education_level"),
        "gender": encode_with_le("gender"),
        "family_history": encode_with_le("family_history"),
        "apoe_e4": encode_with_le("apoe_e4"),
        "medication_history": encode_with_le("medication_history"),
        "sleep_quality": encode_with_le("sleep_quality"),
        "chronic_health_conditions": encode_with_le("chronic_health_conditions"),
    }

    # ✅ 원핫 인코딩 대상 항목
    def encode_onehot(value, mapping):
        value = value.lower()
        return {v: int(value == k) for k, v in mapping.items()}

    smoking_map = {
        "never": "smoking_status_never smoked",
        "former": "smoking_status_former smoker",
        "current": "smoking_status_current smoker"
    }
    diet_map = {
        "normal": "nutrition_diet_balanced diet",
        "low_carb": "nutrition_diet_low-carb diet",
        "low_salt": "nutrition_diet_mediterranean diet"
    }
    activity_map = {
        "mild": "physical_activity_mild activity",
        "moderate": "physical_activity_moderate activity",
        "sedentary": "physical_activity_sedentary"
    }

    smoking_encoded = encode_onehot(data.get("smoking_status", ""), smoking_map)
    diet_encoded = encode_onehot(data.get("nutrition_diet", ""), diet_map)
    activity_encoded = encode_onehot(data.get("physical_activity", ""), activity_map)

    # ✅ 최종 feature 조립 및 정렬
    features = {**base_features, **smoking_encoded, **diet_encoded, **activity_encoded}
    ordered_cols = [
        'diabetic', 'heartrate', 'bloodoxygenlevel', 'weight', 
        'age', 'education_level', 'gender', 'family_history', 'apoe_e4',
        'medication_history', 'sleep_quality', 'chronic_health_conditions',
        'smoking_status_current smoker', 'smoking_status_former smoker', 'smoking_status_never smoked',
        'nutrition_diet_balanced diet', 'nutrition_diet_low-carb diet', 'nutrition_diet_mediterranean diet',
        'physical_activity_mild activity', 'physical_activity_moderate activity', 'physical_activity_sedentary'
    ]

    df = pd.DataFrame([features])[ordered_cols].astype("float64")
    return df, data  # 모델 입력, 사용자 원본
