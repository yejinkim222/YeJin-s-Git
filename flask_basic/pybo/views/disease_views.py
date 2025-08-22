from __future__ import annotations

from flask import Blueprint, render_template

bp = Blueprint("disease", __name__, url_prefix="/disease")


@bp.route("/")
def home():
    """
    질병 정보 허브 페이지를 렌더링한다.
    템플릿: templates/disease/disease.html
    """
    return render_template("disease/disease.html")


@bp.route("/voice")
def voice():
    """
    질병 정보 > AI 자가검사(음성) 페이지를 렌더링한다.
    템플릿: templates/disease/disease_voice.html

    이 페이지의 JS는 다음 API를 호출한다고 가정한다(별도 블루프린트 `/screening`):
      - 전사:  POST /screening/asr      (FormData: audio)
      - 점수:  POST /screening/score    (JSON: {"answers": {...}})
    """
    return render_template("disease/disease_voice.html")
