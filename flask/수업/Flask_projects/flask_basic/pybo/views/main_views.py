from flask import Blueprint, url_for
from werkzeug.utils import redirect

# 여기 맨앞에 메인은 파이참에서 적어준 것. 내가 작성 x
bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/hello')
def hello_pybo():
    return 'Hello, Pybo!'

@bp.route('/')
def index():
    return redirect(url_for('question._list'))
