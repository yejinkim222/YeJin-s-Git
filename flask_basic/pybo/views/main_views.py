# pybo/views/main_views.py
from flask import Blueprint, render_template

bp = Blueprint("main", __name__, url_prefix="/")

@bp.route("/")
def index():
    """메인 랜딩 페이지."""
    return render_template("index.html")

@bp.route("/disease")
def disease_home():
    """질병 정보 허브 페이지."""
    return render_template("disease/disease.html")

@bp.route("/disease/audio")
def disease_voice():
    """질병 정보 > 자가검사(음성) 페이지."""
    return render_template("disease/disease_voice.html")


