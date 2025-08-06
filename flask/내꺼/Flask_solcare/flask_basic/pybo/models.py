from pybo import db

class InputData(db.Model):
    # 사용자 입력값 저장 테이블
    __tablename__ = 'inputData'
    id = db.Column(db.Integer,  db.Sequence('input_seq', start=1, increment=1), primary_key=True)

    # 사용자 입력값 변수들
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Integer, nullable=False)
    edu_yrs = db.Column(db.Integer, nullable=False)
    has_db = db.Column(db.Integer, nullable=False)
    has_hibpe = db.Column(db.Integer, nullable=False)
    # mci는 js에서 변환한 뒤 받아 오기
    has_mci = db.Column(db.Integer, nullable=False)
    base_yrs = db.Column(db.Integer, nullable=False)
    input_date = db.Column(db.DateTime(), nullable=False)

class ModelData(db.Model):
    # 모델 예측용 저장 테이블
    __tablename__ = 'modelData'
    id = db.Column(db.Integer, db.Sequence('data_seq', start=1, increment=1),primary_key=True)

    # 입력값 아이디(시퀀스) 외부 키
    input_id = db.Column(db.Integer, db.ForeignKey('inputData.id', ondelete='CASCADE'))

    # 모델이 학습할 변수로 저장
    # 사용자 입력 그대로 사용
    age = db.Column(db.Integer, db.ForeignKey('inputData.age'))
    gender = db.Column(db.Integer, db.ForeignKey('inputData.gender'))
    edu_yrs = db.Column(db.Integer, db.ForeignKey('inputData.edu_yrs'))
    has_db = db.Column(db.Integer, db.ForeignKey('inputData.has_db'))
    AD_MCI_status = db.Column(db.Integer, db.ForeignKey('inputData.has_mci')) # == has_mci
    has_hibpe = db.Column(db.Integer, db.ForeignKey('inputData.has_hibpe'))
    edu_level = db.Column(
        db.Integer,
        db.ForeignKey('inputData.edu_yrs').apply(
            lambda x: 0 if x <= 5 else 1 if x <= 8 else 2 if x <= 11 else 3 if x <= 13 else 4))
    # onset 변수
    db_onset_after = db.Column(
        db.Integer,
        db.ForeignKey('inputData.has_db').apply(lambda x: -1 if x == 0 else 0))
    mci_onset_after = db.Column(db.Integer,
        db.ForeignKey('inputData.has_mci').apply(lambda x: -1 if x == 0 else 0))
    hibpe_onset_after = db.Column(db.Integer,
        db.ForeignKey('inputData.has_hibpe').apply(lambda x: -1 if x == 0 else 0))
    # 파생 변수
    age_group5 = db.Column(db.Integer, (db.ForeignKey('inputData.age')//5).astype(int))
    risk_factor_sum = db.Column(
        db.Integer, (
                db.ForeignKey('inputData.has_db') +
                db.ForeignKey('inputData.has_mci') +
                db.ForeignKey('inputData.has_hibpe')))
    edu_is_low = db.Column(
        db.Integer, db.ForeignKey('inputData.edu_level').apply(lambda a:1 if a==1 else 0).astype(int))
    risk_weighted_age = db.Column(db.Integer, (db.ForeignKey('inputData.age') / (1 + risk_factor_sum)))
    age_gender_interact = db.Column(db.Integer, (age * gender).astype(int))
    # 결측 마킹
    hibpe_onset_after_missing = db.Column(db.Integer, 0)
    has_hibpe_missing = db.Column(db.Integer, 0)
    mci_onset_after_missing = db.Column(db.Integer, 0)
    edu_yrs_missing = db.Column(db.Integer, 0)
    db_onset_after_missing = db.Column(db.Integer, 0)
    # 파생 변수
    cognitive_decline_flag = AD_MCI_status  # 인지 문제 여부
    age_x_edu = (age * edu_yrs).astype(int)
    hibpe_onset_delay_ratio = (hibpe_onset_after / (age + 1e-3 )).astype(int)
    age_edu_ratio = (age / edu_yrs+1).astype(int)

class Users(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, db.Sequence('users_seq', start=1, increment=1),primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
