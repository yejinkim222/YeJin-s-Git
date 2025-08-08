from flask import Blueprint, url_for, render_template
from werkzeug.utils import redirect

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return redirect(url_for('auth.login'))

@bp.route('/disease')
def disease():
    return render_template('disease.html')