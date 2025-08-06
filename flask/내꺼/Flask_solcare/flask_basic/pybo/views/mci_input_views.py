from datetime import datetime
from flask import Blueprint, render_template, request, url_for, g, flash
from pybo.models import InputData
from .auth_views import login_required
from ..forms import InputDataForm
from werkzeug.utils import redirect
from .. import db

bp = Blueprint('mci_input', __name__, url_prefix='/mci_input')

@bp.route('/input/', methods=['GET', 'POST'])
@login_required
def create():
    form = InputDataForm()

    if request.method == 'POST' and form.validate_on_submit():
        input_data = InputData(
            age = form.age.data,
            gender = form.gender.data,
            edu_yrs = form.edu_yrs.data,
            has_db=form.has_db.data,
            has_hibpe = form.has_hibpe.data,
            has_mci = form.has_mci.data,
            base_yrs = form.base_yrs.data
        )

        db.session.add(input_data)
        db.session.commit()

        return redirect(url_for('mci_input.index'))

    return render_template('mci/mci_input.html', form=form)
