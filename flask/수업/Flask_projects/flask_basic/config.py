import os
from flask_sqlalchemy import SQLAlchemy

BASE_DIR = os.path.dirname(__file__) # 현재 파일의 경로(참고용)
print("BASE_DIR:",BASE_DIR)

# Oracle 11g 설정(cx_Oracle 사용)
# DB 접속 정보: id=scott, password=tiger, 호스트=localhost, 포트=1521, SID=xe
SQLALCHEMY_DATABASE_URI = "oracle+cx_oracle://scott:tiger@localhost:1521/xe"
print("SQLALCHEMY_DATABASE_URI:",SQLALCHEMY_DATABASE_URI)

SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY="dev"