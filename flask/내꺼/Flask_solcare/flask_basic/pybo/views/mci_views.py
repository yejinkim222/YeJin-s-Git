from datetime import datetime
from flask import Blueprint, render_template, request, url_for, g, flash
from werkzeug.utils import redirect

from .. import db
from pybo.models import InputData
from .auth_views import login_required
from ..forms import InputDataForm

bp = Blueprint('mci', __name__, url_prefix='/mci')

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

        return redirect(url_for('mci.mci_output', input_id=input_data.id))

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
        'age_edu_ratio': (row.age / (edu_yrs + 1)),
        # 필요시 base_yrs도 전달
        'base_yrs': row.base_yrs,
    }
    return feats


# 예측 함수(더미)
def predict_one(feats: dict) -> float:
    """
    실제 사용 시:
    import joblib, pandas as pd
    model = joblib.load("C:/workspace/Project01/model_storage/xgb_best_model_final3.pkl")
    columns = [
        'age','gender','edu_yrs','has_db','ad_mci_status','has_hibpe','edu_level',
        'db_onset_after','hibpe_onset_after','mci_onset_after','age_group5','risk_factor_sum',
        'edu_is_low','risk_weighted_age','age_gender_interact','hibpe_onset_after_missing',
        'has_hibpe_missing','mci_onset_after_missing','edu_yrs_missing','db_onset_after_missing',
        'cognitive_decline_flag','age_x_edu','hibpe_onset_delay_ratio','age_edu_ratio'
    ]
    X = pd.DataFrame([feats], columns=columns)  # 학습시 컬럼 순서와 동일해야 함
    yhat = float(model.predict(X)[0])
    return yhat
    """
    return 0.0  # 우선 0.0 반환(모델 연결 전)

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