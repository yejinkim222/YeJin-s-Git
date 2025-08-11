from datetime import datetime
from flask import Blueprint, render_template, request, url_for, g, flash
from pybo.models import InputData
from .auth_views import login_required
from ..forms import InputDataForm, ModelDataForm
from werkzeug.utils import redirect
from .. import db

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
            has_mci = form.has_mci.data,
            base_yrs = form.base_yrs.data,
            input_date = datetime.now(),
            user_id = g.user.id if g.user else None
        )

        db.session.add(input_data)
        db.session.commit()

        return redirect(url_for('mci.create_input', input_id=input_data.id))

    return render_template('mci/mci_form.html', form=form)





# 파생 변수 생성
def generate_features(input_data: InputData):
    # 온셋시점 생성용
    def onset(data):
        return 0 if data else -1

    # 람다함수만 따로
    x = input_data.edu_yrs
    edu_level = 0 if x <= 5 else 1 if x <= 8 else 2 if x <= 11 else 3 if x <= 13 else 4
    risk_factor_sum = input_data.has_db + input_data.has_mci + input_data.has_hibpe


    form = ModelDataForm()
    model_data = ModelDataForm.query.get_or_404(input_data)


    if form.validate_on_submit():
        # 원본 입력값 그대로 사용하는 변수
        input_data = InputData(
            age = input.age.data,
            gender = input.gender.data,
            edu_level = input.edu_level.data,
            has_db = input.has_db.data,
            has_hibpe = input.has_hibpe.data,
            has_mci = input.AD_MCI_status.data,
            base_yrs= input.base_yrs.data
        )
        model_data = ModelDataForm(
            input_data,
            # 파생 변수
            edu_yrs = input_data.edu_yrs.apply(lambda x: 0 if x == 0 else 6 if x == 1 else 9 if x == 2 else 12 if x == 3 else 14),
            db_onset_after = onset(input_data.has_db),
            mci_onset_after = onset(input_data.has_mci),
            hibpe_onset_after = onset(input_data.has_hibpe),
            age_group5 = (input_data.age//5).astype(int),
            risk_factor_sum = input_data.has_db + input_data.has_mci + input_data.has_hibpe,
            edu_is_low = input_data.edu_level.apply(lambda x: 1 if x == (0 or 1) else 0),
            risk_weighted_age = input_data.age / (1 + input_data.risk_factor_sum),
            age_gender_interact = (input_data.age * input_data.gender).astype(int),
            cognitive_decline_flag = input_data.has_mci,
            age_x_edu = (input_data.age * input_data.edu_yrs).astype(int),
            hibpe_onset_delay_ratio = (input_data.hibpe_onset_after / (input_data.age + 1e-3)),
            age_edu_ratio = (input_data.age / (input_data.edu_yrs + 1))
        )
        return redirect(url_for('mci_output', input_id=input_data.id))

    return render_template('mci/mci_form.html',form=form)

def prediction(model_data):
    import joblib

    model_path = "C:/workspace/Project01/model_storage/xgb_best_model_final3.pkl"
    model = joblib.load(model_path)

    feature = {
        'age', 'gender', 'edu_yrs', 'has_db', 'ad_mci_status', 'has_hibpe', 'edu_level', 'db_onset_after',
        'hibpe_onset_after', 'mci_onset_after', 'age_group5', 'risk_factor_sum', 'edu_is_low', 'risk_weighted_age',
        'age_gender_interact', 'hibpe_onset_after_missing', 'has_hibpe_missing', 'mci_onset_after_missing',
        'edu_yrs_missing', 'db_onset_after_missing', 'cognitive_decline_flag', 'age_x_edu', 'hibpe_onset_delay_ratio',
        'age_edu_ratio'
    }

    model_list = (model_data for model_data.length in feature)

    result = model.predict(model_list)

    return result

@bp.route('/output/<int:inputData_id>', methods=['POST'])
@login_required
def create_output(input_id):
    input_data = InputData.query.get_or_404(input_id)
    features = features(input_data.id)
    result = prediction(features)

    model_result = ModelDataForm(
        id = ModelDataForm.id,
        input_id = input_id,
        date = datetime.now(),
        result = prediction(model_data)
    )

    db.session.add(model_data)
    db.session.commit()

    return model_result.id