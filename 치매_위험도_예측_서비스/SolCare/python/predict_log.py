import cx_Oracle  # ✅ Oracle 11g 지원 확실한 모듈
import datetime
from prepredict import preprocess_input  # 예진님 모듈

# ✅ 1. Oracle 접속 정보 (환경 맞게 수정)
ORACLE_USER = "asdf"
ORACLE_PW = "asdf"
ORACLE_DSN = "localhost:1521/xe"

# ✅ 2. 입력값 수동 정의 (JS 없이 직접 테스트용)
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
form_data = raw_input

# ✅ 3. 전처리 함수 호출
processed = preprocess_input(form_data)

# ✅ 4. 예측값 (지금은 고정값, 나중에 모델 붙이기)
predicted_year = 4.27

# ✅ 5. SQL 준비 (시퀀스+트리거 방식, log_id는 INSERT 생략)
sql = """
    INSERT INTO user_input_log (
        log_time, age, edu_yrs, gender, has_db, has_hibpe,
        cog_input_mode, has_mci, mmse_score, period, predicted_year
    ) VALUES (
        :1, :2, :3, :4, :5, :6,
        :7, :8, :9, :10, :11
    )
"""

# ✅ 6. 파라미터 구성 함수 (빈 문자열이나 None → -1로 변환)
def get_val(key, cast=int):
    val = form_data.get(key)
    if val in [None, ""]:
        return -1
    return cast(val)

params = (
    datetime.datetime.now(),
    get_val("age"),
    get_val("education"),
    get_val("gender"),
    get_val("db"),
    get_val("hibpe"),
    form_data.get("cog_input_mode", ""),
    get_val("has_mci"),
    get_val("mmse_score"),
    get_val("period"),
    float(predicted_year)
)

# ✅ 7. DB 연결 및 실행
try:
    conn = cx_Oracle.connect(user=ORACLE_USER, password=ORACLE_PW, dsn=ORACLE_DSN)
    cursor = conn.cursor()
    cursor.execute(sql, params)
    conn.commit()
    cursor.close()
    conn.close()
except Exception as e:
    raise RuntimeError(f"❌ DB INSERT 실패: {e}")
