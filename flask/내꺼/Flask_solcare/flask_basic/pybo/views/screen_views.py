from __future__ import annotations
import os, uuid
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from pybo.python.dmt_model.dementia_screen import screen_audio_for_dementia

bp = Blueprint("screen", __name__, url_prefix="/screen")

@bp.route("/dementia", methods=["POST"])
def dementia_screen():
    """
    multipart/form-data 업로드:
      - file: 오디오 파일(.wav/.mp3 등)
    응답(JSON): transcript, features{...}, risk_score, risk_label
    """
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "파일이 없습니다.(form field name: file)"}), 400

    updir = current_app.config.get("DEMENTIA_UPLOAD_DIR")
    os.makedirs(updir, exist_ok=True)
    fname = secure_filename(f.filename or f"audio_{uuid.uuid4().hex}.wav")
    path = os.path.join(updir, fname)
    f.save(path)

    result = screen_audio_for_dementia(
        path,
        asr_model=current_app.config.get("DEMENTIA_ASR_MODEL_SIZE", "medium"),
        clf_pkl=current_app.config.get("DEMENTIA_CLF_PKL", "")
    )

    return jsonify({
        "transcript": result.transcript,
        "features": result.features,
        "risk_score": result.risk_score,
        "risk_label": result.risk_label,
        "notice": "본 결과는 정보제공용 스크리닝이며, 진단이 아닙니다."
    })
