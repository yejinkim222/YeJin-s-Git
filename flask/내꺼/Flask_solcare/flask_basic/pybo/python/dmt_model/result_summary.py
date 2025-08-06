# result_summary.py

def summarize_input(data):
    input_summary_map = {
        "Age": "{value}세",
        "Gender": {"Male": "남성", "Female": "여성"},
        "Diabetic": {1: "당뇨 있음", 0: "당뇨 없음"},
        "Family_History": {1: "가족력 있음", 0: "가족력 없음"},
        "Depression_Status": {1: "우울증 있음", 0: "우울증 없음"},
        "Sleep_Quality": {"good": "수면 질 좋음", "fair": "수면 질 보통", "poor": "수면 질 나쁨"},
        "Physical_Activity": {"low": "운동 부족", "medium": "운동 보통", "high": "운동 충분"},
        "Smoking_Status": {"never": "비흡연", "former": "과거 흡연", "current": "현재 흡연 중"},
        "Chronic_Health_Conditions": {
            "diabetes": "만성질환(당뇨)",
            "hypertension": "만성질환(고혈압)",
            "stroke": "만성질환(뇌졸중)",
            "cardio": "만성질환(심혈관)"
        },
        "APOE_e4": {1: "APOE ε4 있음", 0: "APOE ε4 없음", -1: "APOE ε4 모름"},
        "Nutrition_Diet": {"normal": "일반식", "low_salt": "저염식", "low_carb": "저탄수화물 식단"}
    }

    summary = []
    for key, rule in input_summary_map.items():
        val = data.get(key.lower(), None)
        if isinstance(val, list):
            val = val[0] if val else None
        if val is None:
            continue
        if isinstance(rule, str):
            summary.append(rule.format(value=val))
        elif isinstance(rule, dict):
            label = rule.get(val, None)
            if label:
                summary.append(label)
    return ", ".join(summary)
