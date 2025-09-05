# pybo/views/disease_views.py
from __future__ import annotations

import os
import uuid
import re

from flask import Blueprint, render_template, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename

from pybo.python.dmt_model.dementia_screen import screen_audio_for_dementia

# ===== AD8-유사 문항 기반 자가진단(음성) 플로우 =====
from flask import session, g
from pybo import db
from pybo.models import ScreeningResult

bp = Blueprint("disease", __name__, url_prefix="/disease")

ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".opus"}


@bp.route("/")
def home():
    """질병 정보 허브 페이지를 렌더링한다."""
    return render_template("disease/disease.html")


@bp.route("/voice")
def voice():
    """질병 정보 > AI 자가검사(음성) 페이지를 렌더링한다."""
    return render_template("disease/disease_voice.html")


@bp.route("/dementia", methods=["POST"])
def dementia_screen():
    """자가진단 오디오 업로드를 받아 전사/특징/리스크를 반환한다."""
    f = request.files.get("file")
    if not f:
        return jsonify(error="파일이 없습니다. (form field name: file)"), 400

    updir = current_app.config.get("DEMENTIA_UPLOAD_DIR") or \
            os.path.join(current_app.root_path, "data", "uploads", "dementia")
    os.makedirs(updir, exist_ok=True)

    fname = secure_filename(f.filename or f"audio_{uuid.uuid4().hex}.wav")
    ext = os.path.splitext(fname)[1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify(error=f"지원하지 않는 오디오 형식입니다: {ext}"), 400

    path = os.path.join(updir, fname)
    f.save(path)

    try:
        result = screen_audio_for_dementia(
            path,
            asr_model=current_app.config.get("DEMENTIA_ASR_MODEL_SIZE", "small"),
            clf_pkl=current_app.config.get("DEMENTIA_CLF_PKL", "")
        )
        return jsonify(
            transcript=result.transcript,
            features=result.features,
            risk_score=result.risk_score,
            risk_label=result.risk_label,
            notice="본 결과는 정보제공용 스크리닝이며, 진단이 아닙니다."
        )
    except FileNotFoundError as e:
        current_app.logger.exception("dementia_screen: 모델/리소스 누락")
        return jsonify(error=f"모델 또는 리소스가 없습니다: {e}"), 500
    except Exception as e:
        current_app.logger.exception("dementia_screen: 처리 실패")
        return jsonify(error=f"처리 중 오류가 발생했습니다: {e}"), 500

@bp.route("/data/output.json")
def serve_output_json():
    data_dir = os.path.join(current_app.root_path, "data")   # <프로젝트>/pybo/data
    return send_from_directory(
        directory=data_dir,
        path="output.json",
        mimetype="application/json",
        max_age=0,   # 캐시 방지
    )


# AD8의 8개 핵심 영역을 생활 문장으로 재구성(비공식/참고용)
_AD8_LIKE_QUESTIONS = [
    "최근 몇 년 사이, 판단이나 결정이 예전보다 서툴러졌다고 느끼시나요? (예: 금전/계산 실수가 잦다 등)",
    "예전에 즐기던 취미나 활동에 대한 흥미가 눈에 띄게 줄었나요?",
    "같은 질문이나 이야기를 반복해서 하는 일이 늘었나요?",
    "새 기기·도구(리모컨, 전자레인지, 휴대폰 등)를 배우거나 다루는 게 유난히 어려워졌나요?",
    "정확한 달(월)이나 해(년도)를 헷갈리는 일이 생기나요?",
    "복잡한 금전 처리(계좌·세금·고지서 등)가 어려워졌나요?",
    "약속이나 일정(병원 예약 등)을 잊는 일이 잦아졌나요?",
    "일상에서 생각이나 기억 때문에 불편함이 거의 매일 생기나요?",
]

def _ad8_reset():
    session["ad8_idx"] = 0
    session["ad8_score"] = 0
    session["ad8_answers"] = []

_YES_PAT = re.compile(r"(?:예|네|맞|그렇|있어|있다|늘었|줄었|어려워|자주|헷갈|잊|문제|불편)", re.I)
_NO_PAT  = re.compile(r"(?:아니|아닙|없|줄지 않|변화 없|괜찮|문제 없)", re.I)
_DK_PAT  = re.compile(r"(?:모르겠|잘 모르|기억 안|글쎄|모름)", re.I)

def _score_from_answer(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0  # 무응답은 0으로 처리
    if _YES_PAT.search(t) and not _NO_PAT.search(t):
        return 1
    if _NO_PAT.search(t):
        return 0
    if _DK_PAT.search(t):
        return 0  # AD8의 DK는 점수 미부여
    # 모호하면 0
    return 0

def _comm_flag(answer: str) -> bool:
    """답변 품질이 낮아 보이면 True (짧음/모름/군말투 과다 등 단순 휴리스틱)"""
    a = (answer or "").strip()
    if len(a) < 4:
        return True
    if _DK_PAT.search(a):
        return True
    fillers = re.findall(r"(음+|어+|그[음]+|저+|그냥|뭔가)", a)
    return len(fillers) >= 3

def _save_screening_result(label: str, score: int, total: int, advice: str, summary_text: str) -> None:
    if not getattr(g, "user", None):
        return
    row = ScreeningResult(
        user_id=g.user.id,
        total_score=int(score),
        max_score=int(total),
        result_summary=label,
        need_referral=(label != "정상"),
        advice=advice,
        result_text=summary_text,
    )
    db.session.add(row)
    db.session.commit()

@bp.route("/ad8/step", methods=["POST"])
def ad8_step():
    """reset 또는 다음 답변 처리 → 다음 질문/최종 결과 반환"""
    data = request.get_json(silent=True) or {}

    if data.get("reset"):
        _ad8_reset()
        q0 = _AD8_LIKE_QUESTIONS[0]
        return jsonify(ok=True, done=False, index=0, total=len(_AD8_LIKE_QUESTIONS), reply=q0), 200

    # 진행 상태
    idx   = int(session.get("ad8_idx", 0))
    score = int(session.get("ad8_score", 0))
    hist  = list(session.get("ad8_answers", []))

    # 현재 문항에 대한 사용자 답변 처리
    user_text = (data.get("text") or "").strip()
    if idx < len(_AD8_LIKE_QUESTIONS):
        score += _score_from_answer(user_text)
        hist.append({"q": _AD8_LIKE_QUESTIONS[idx], "a": user_text, "pt": _score_from_answer(user_text)})
        idx += 1

    # 다음 단계 결정
    if idx >= len(_AD8_LIKE_QUESTIONS):
        # 최종 스코어 산출
        label = "정상" if score <= 1 else "평가 권고"
        base_advice = "현재 점수는 참고용입니다. 변화가 지속되거나 걱정이 되면 보건소·치매안심센터 또는 병원 상담을 권합니다."
        cutoff_note = "(참고: AD8 기준 0–1 정상, 2점 이상이면 추가 평가 권고로 해석됩니다.)"

        # 의사소통 품질 경고(답변들 중 품질 낮음이 2회 이상이면 알림)
        comm_flags = sum(1 for it in hist if _comm_flag(it.get("a", "")))
        extra = ""
        if comm_flags >= 2:
            extra = "\n또한 몇몇 답변이 매우 짧거나 '모르겠다'가 잦아 실제 상태 파악이 제한적이었어요. 대면 검사를 고려해 주세요."

        summary = (
            f"진단이 아닌 참고용 결과입니다.\n"
            f"- 총점: {score} / {len(_AD8_LIKE_QUESTIONS)} → {label}\n"
            f"- 권고: {base_advice}\n"
            f"{cutoff_note}{extra}"
        )

        # 로그인 사용자라면 저장
        try:
            _save_screening_result(label, score, len(_AD8_LIKE_QUESTIONS), base_advice, summary)
        except Exception as e:
            current_app.logger.warning("AD8 save failed: %s", e)

        # 세션 정리
        session.pop("ad8_idx", None)
        session.pop("ad8_score", None)
        session.pop("ad8_answers", None)

        return jsonify(ok=True, done=True, score=score, total=len(_AD8_LIKE_QUESTIONS), reply=summary), 200

    # 아직 진행 중 → 다음 질문
    session["ad8_idx"] = idx
    session["ad8_score"] = score
    session["ad8_answers"] = hist
    next_q = _AD8_LIKE_QUESTIONS[idx]
    return jsonify(ok=True, done=False, index=idx, total=len(_AD8_LIKE_QUESTIONS), reply=next_q), 200
