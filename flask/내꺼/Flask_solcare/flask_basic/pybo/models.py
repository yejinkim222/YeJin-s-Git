from datetime import datetime

from sqlalchemy.orm import backref

from pybo import db

class InputData(db.Model):
    __tablename__ = 'input_data' # 사용자 입력값 저장 테이블

    id = db.Column(db.Integer,  db.Sequence('input_seq', start=1, increment=1), primary_key=True) # 시퀀스 생성
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'),nullable=True)  # 사용자 id 외부 키 사용, 유저 아직 없어서 null 가능하게 둠

    # 유저 테이블에서 해당 유저의 결과 전체 참조, 이 테이블에서 해당 유저 정보 참조 가능하게 연결
    user = db.relationship('Users', backref=backref('input_datas', lazy=True))
    input_date = db.Column(db.DateTime(), nullable=False)  # 날짜

    # 사용자 입력값 변수들
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Integer, nullable=False)
    edu_level = db.Column(db.Integer, nullable=False)
    has_db = db.Column(db.Integer, nullable=False)
    has_hibpe = db.Column(db.Integer, nullable=False)
    # mci는 js에서 변환한 뒤 받아 오기
    has_mci = db.Column(db.Integer, nullable=False)
    base_yrs = db.Column(db.Integer, nullable=False)



class OutputData(db.Model):
    __tablename__ = 'output_data'  # 모델 예측용 저장 테이블

    id = db.Column(db.Integer, db.Sequence('data_seq', start=1, increment=1),primary_key=True)  # 시퀀스 생성
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # 아직 유저 없어서 null 가능
    user = db.relationship('Users', backref=backref('output_datas', lazy=True))
    input_date = db.Column(db.DateTime(), nullable=False)  # 날짜

    # 모델이 학습할 변수로 저장
    # 사용자 입력 그대로 사용
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Integer, nullable=False)
    edu_level = db.Column(db.Integer, nullable=False)
    has_db = db.Column(db.Integer, nullable=False)
    has_hibpe = db.Column(db.Integer, nullable=False)
    # mci는 js에서 변환한 뒤 받아 오기
    has_mci = db.Column(db.Integer, nullable=False)
    base_yrs = db.Column(db.Integer, nullable=False)

    # 파생 변수
    edu_yrs = db.Column(db.Integer, nullable=False)
    # onset 변수
    db_onset_after = db.Column(db.Integer, nullable=False)
    mci_onset_after = db.Column(db.Integer, nullable=False)
    hibpe_onset_after = db.Column(db.Integer, nullable=False)
    # 파생 변수
    age_group5 = db.Column(db.Integer, nullable=False)
    risk_factor_sum = db.Column(db.Integer, nullable=False)
    edu_is_low = db.Column(db.Integer, nullable=False)
    risk_weighted_age = db.Column(db.Float, nullable=False)
    age_gender_interact = db.Column(db.Integer, nullable=False)
    # 결측 마킹
    hibpe_onset_after_missing = db.Column(db.Integer, nullable=False)
    has_hibpe_missing = db.Column(db.Integer, nullable=False)
    mci_onset_after_missing = db.Column(db.Integer, nullable=False)
    edu_yrs_missing = db.Column(db.Integer, nullable=False)
    db_onset_after_missing = db.Column(db.Integer, nullable=False)
    # 파생 변수
    cognitive_decline_flag = db.Column(db.Integer,nullable=False)
    age_x_edu = db.Column(db.Integer, nullable=False)
    hibpe_onset_delay_ratio = db.Column(db.Float, nullable=False)
    age_edu_ratio = db.Column(db.Float, nullable=False)

class Users(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, db.Sequence('users_seq', start=1, increment=1),primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
