# ✅ dementia/db_utils.py

import cx_Oracle

def save_to_db(result, input_data):
    try:
        # Oracle 연결 설정
        dsn = cx_Oracle.makedsn("localhost", 1521, service_name="xe")
        conn = cx_Oracle.connect(user="elderly_user", password="1234", dsn=dsn)
        cursor = conn.cursor()

        insert_sql = """
        INSERT INTO PREDICTION_RESULT (
            NAME, CREATED_AT,
            DIABETIC, HEARTRATE, BLOODOXYGENLEVEL, WEIGHT, 
            AGE, EDUCATION_LEVEL, GENDER, FAMILY_HISTORY, SMOKING_STATUS,
            APOE_E4, PHYSICAL_ACTIVITY, MEDICATION_HISTORY,
            NUTRITION_DIET, SLEEP_QUALITY, CHRONIC_HEALTH_CONDITIONS, 
            RISK_SCORE, RISK_LEVEL, PREDICTION
        ) VALUES (
            :1, CURRENT_TIMESTAMP,
            :2, :3, :4, :5, :6,
            :7, :8, :9, :10, :11,
            :12, :13, :14, :15,
            :16, :17, :18,
            :19
        )
        """

       # dementia/db_utils.py 내부
        cursor.execute(insert_sql, (
            input_data.get("name"),
            input_data.get("diabetic"),
            input_data.get("heartrate"),
            input_data.get("bloodoxygenlevel"),
            input_data.get("weight"),
            input_data.get("age"),
            input_data.get("education_level"),
            input_data.get("gender"),
            input_data.get("family_history"),
            input_data.get("smoking_status"),
            input_data.get("apoe_e4"),
            input_data.get("physical_activity"),
            input_data.get("medication_history"),
            input_data.get("nutrition_diet"),
            input_data.get("sleep_quality"),
            None if input_data.get("chronic_health_conditions") == "none" else input_data.get("chronic_health_conditions"),
            result.get("risk_score"),
            result.get("risk_level"),
            result.get("prediction")
        ))


        conn.commit()
        cursor.close()
        conn.close()
        print("예측 결과가 DB에 저장되었습니다.")
    except Exception as e:
        print("DB 저장 실패:", str(e))
