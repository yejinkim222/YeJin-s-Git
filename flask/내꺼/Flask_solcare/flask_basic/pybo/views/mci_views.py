from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request, url_for, g, flash
from werkzeug.utils import redirect

from .. import db
from pybo.models import InputData
from .auth_views import login_required
from ..forms import InputDataForm

bp = Blueprint('mci', __name__, url_prefix='/mci')



# 모델 경로 불러오기
MODEL_PATH = Path("C:/workspace/Project01/model_storage/xgb_best_model_final3.pkl")
# 모델 학습 시 사용한 컬럼명 순서대로 고정
MODEL_COLUMNS = [
    'age','gender','edu_yrs','has_db','ad_mci_status','has_hibpe','edu_level',
    'db_onset_after','hibpe_onset_after','mci_onset_after','age_group5','risk_factor_sum',
    'edu_is_low','risk_weighted_age','age_gender_interact',
    'hibpe_onset_after_missing','has_hibpe_missing','mci_onset_after_missing',
    'edu_yrs_missing','db_onset_after_missing','cognitive_decline_flag',
    'age_x_edu','hibpe_onset_delay_ratio','age_edu_ratio'
]
INT_COLS = [
    'age','gender','edu_yrs','has_db','ad_mci_status','has_hibpe','edu_level',
    'db_onset_after','mci_onset_after','hibpe_onset_after','age_group5','risk_factor_sum',
    'edu_is_low','age_gender_interact','hibpe_onset_after_missing','has_hibpe_missing',
    'mci_onset_after_missing','edu_yrs_missing','db_onset_after_missing',
    'cognitive_decline_flag','age_x_edu'
]
FLOAT_COLS = ['risk_weighted_age','hibpe_onset_delay_ratio','age_edu_ratio']



# 모델 1회 가져오기
@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        # 운영 시엔 config에서 경로 주입 권장: current_app.config['MODEL_PATH']
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)



@bp.route('/input/', methods=['GET', 'POST'])
# @login_required
def create_input():
    form = InputDataForm()

    if request.method == 'POST' and form.validate_on_submit():
        input_data = InputData(
            age = form.age.data,
            gender = form.gender.data,
            edu_level = form.edu_level.data,
            has_db=form.has_db.data,
            has_hibpe = form.has_hibpe.data,
            has_mci = int(form.has_mci.data) if form.has_mci.data not in (None, '') else 0,
            base_yrs = form.base_yrs.data,
            input_date = datetime.now(),
            user_id = g.user.id if g.user else None
        )

        db.session.add(input_data)
        db.session.commit()

        return redirect(url_for('mci.output', input_id=input_data.id))

    return render_template('mci/mci_form.html', form=form)



# 파생 변수 생성
def generate_features(row: InputData):

    level_to_years = {0: 5, 1: 8, 2: 11, 3: 13, 4: 14}
    edu_yrs = level_to_years.get(getattr(row, 'edu_level', 3), 12)

    # 온셋 시점 생성용
    def onset(data: int)->int:
        return 0 if data == 1 else -1

    risk_factor_sum = row.has_db + row.has_mci + row.has_hibpe

    feats = {
        'age': row.age,
        'gender': row.gender,
        'edu_yrs': edu_yrs,
        'has_db': row.has_db,
        'ad_mci_status': row.has_mci,
        'has_hibpe': row.has_hibpe,
        'edu_level': getattr(row, 'edu_level', None),
        'db_onset_after': onset(row.has_db),
        'mci_onset_after': onset(row.has_mci),
        'hibpe_onset_after': onset(row.has_hibpe),
        'age_group5': row.age // 5,
        'risk_factor_sum': risk_factor_sum,
        'edu_is_low': 1 if row.edu_level in (0, 1) else 0,
        'risk_weighted_age': row.age / (1 + risk_factor_sum),
        'age_gender_interact': row.age * row.gender,
        # 결측 마킹(없으면 0으로 두는 안전값)
        'hibpe_onset_after_missing': 0,
        'has_hibpe_missing': 0,
        'mci_onset_after_missing': 0,
        'edu_yrs_missing': 0,
        'db_onset_after_missing': 0,
        # 추가 파생
        'cognitive_decline_flag': row.has_mci,
        'age_x_edu': row.age * edu_yrs,
        'hibpe_onset_delay_ratio': (onset(row.has_hibpe) / (row.age + 1e-3)),
        'age_edu_ratio': (row.age / (edu_yrs + 1))
    }
    return feats






# 예측 함수(더미)
def predict_one(feats: dict) -> float:

    model = get_model()

    # df 생성, 컬럼 순서 명시
    X = pd.DataFrame([feats], columns=MODEL_COLUMNS)

    # dtype 맞추기
    X[INT_COLS] = X[INT_COLS].astype(np.int64)
    X[FLOAT_COLS] = X[FLOAT_COLS].astype(np.float32)

    # 확률 예측 가능하면 사용하기
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:,1]
        return float(proba[0])

    yhat = model.predict(X)
    return float(yhat[0])



@bp.route('/output/<int:input_id>/', methods=['GET'])
# @login_required
def output(input_id):

    row = InputData.query.get_or_404(input_id)


    feats = generate_features(row)


    yhat = predict_one(feats)  # 모델 연결 전이면 0.0

    # 템플릿에서 input_data, features, yhat 사용
    return render_template('mci/mci_output.html',
                           input_data=row,
                           features=feats,
                           yhat=yhat)