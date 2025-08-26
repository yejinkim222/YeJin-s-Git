# pybo/views/genai_views.py
from __future__ import annotations

import json
import re
import html
from types import SimpleNamespace
from typing import Dict, List, Optional, Any

from flask import (
    Blueprint, render_template, request, session, current_app,
    jsonify, g, redirect, url_for
)

from pybo import db
from pybo.models import Conversation, Message, ScreeningResult


bp = Blueprint("genai", __name__, url_prefix="/genai")


# =========================================================
# 공통 유틸 ― 로그인/세션
# =========================================================
def _is_logged_in() -> bool:
    return getattr(g, "user", None) is not None


def _to_int(x: Any, default: int = 0) -> int:
    """mypy/pylance 경고 없이 안전하게 정수 변환."""
    try:
        return int(x)
    except Exception:
        return default


# =========================================================
# 상담 챗봇 (/genai/chat)
# =========================================================
def _get_or_create_user_chat() -> Conversation:
    conv = (
        Conversation.query
        .filter_by(user_id=g.user.id)
        .order_by(Conversation.created_at.desc())
        .first()
    )
    if not conv:
        conv = Conversation(user_id=g.user.id, title="상담 챗봇")
        db.session.add(conv)
        db.session.commit()
    return conv


def _guest_chat_msgs() -> List[Dict[str, str]]:
    return session.get("chat_guest_messages", [])


def _save_guest_chat_msgs(msgs: List[Dict[str, str]]) -> None:
    session["chat_guest_messages"] = msgs


@bp.route("/chat", methods=["GET", "POST"])
def chat():
    # --- GET: reset 처리 ---
    if request.method == "GET":
        if request.args.get("reset") == "1":
            session.pop("chat_guest_messages", None)
            return redirect(url_for("genai.chat"))

    # --- POST: 폼/JSON 모두 허용 ---
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form or {}
        text = (data.get("text") or data.get("message") or "").strip()
        if text:
            system = (
                "너는 한국어로 편안하게 대화하는 상담사다. 검사용 문항(기억 단어 제시, 계산, 점수/진단 언급)은 절대 하지 말라. "
                "모르는 것은 잘 모르겠다, 죄송하다 하고 화제전환한다. "
                "항상 1–2문장으로 공감→짧은 질문 순으로 말한다. 같은 질문을 두 번 실패하면 다른 화제로 전환한다. "
                "대화 전환 규칙:\n"
                " - [회피/거부]('몰라','모르겠어','그냥 그래' 등): 허용·공감 후 '선택지 제시' 또는 '감각 단서(보이는 것/먹은 것/날씨)'로 좁혀 묻는다.\n"
                " - [반복/고집] 같은 내용이 2회 반복되면 요약 한 문장 후 화제를 옮긴다(예: 오늘 식사/창밖/TV/가까운 사람).\n"
                " - [혼란/막힘] 말이 막히면 질문을 더 구체화하고 예시 2개만 제시한다(예: 산책/TV처럼). '왜' 질문은 피하고 '무엇/어떻게' 위주로 묻는다.\n"
                " - [부정 감정] 짧게 정서 반영 후 선택지 2개로 묻는다(지금은 쉬고 싶은지, 가벼운 얘기를 이어갈지).\n"
                "스타일: 존댓말, 따뜻하고 단정한 어조, 의료·진단 표현 금지. 매 턴 끝에 부드러운 후속 질문 1개만."
            )

            if _is_logged_in():
                conv = _get_or_create_user_chat()
                last_k = int(current_app.config.get("GENAI_MAX_CTX_MESSAGES", 16))
                msgs = (
                    Message.query
                    .filter_by(conversation_id=conv.id)
                    .order_by(Message.created_at.desc())
                    .limit(last_k)
                    .all()
                )
                msgs = list(reversed(msgs))
                history = [{"role": m.role, "content": m.content} for m in msgs]
                history.append({"role": "user", "content": text})

                llm = get_llm()
                answer = _generate(
                    llm,
                    [{"role": "system", "content": system}] + history,
                    max_new_tokens=180,
                ) or "도움이 될 만한 이야기를 조금 더 들려주실래요?"

                db.session.add(Message(conversation_id=conv.id, role="user", content=text))
                db.session.add(Message(conversation_id=conv.id, role="assistant", content=answer))
                db.session.commit()

            else:
                history = _guest_chat_msgs()
                history.append({"role": "user", "content": text})

                llm = get_llm()
                answer = _generate(
                    llm,
                    [{"role": "system", "content": system}] + history,
                    max_new_tokens=180,
                ) or "혹시 더 나눠보고 싶은 주제가 있을까요?"

                history.append({"role": "assistant", "content": answer})
                _save_guest_chat_msgs(history)


        # 폼 전송이면 리다이렉트, JSON이면 JSON 응답도 가능하지만 현 템플릿은 폼 기준
        if request.is_json:
            return jsonify(ok=True), 200
        return redirect(url_for("genai.chat"))


    # --- GET 렌더 ---
    if _is_logged_in():
        conv = _get_or_create_user_chat()
        msgs = (
            Message.query
            .filter_by(conversation_id=conv.id)
            .order_by(Message.created_at.asc())
            .all()
        )
        view_msgs = [{"role": m.role, "content": m.content} for m in msgs]
        return render_template(
            "genai/genai_chat.html",
            messages=view_msgs,
            conversation=conv,
            is_guest=False,
            # 이 페이지에서의 전송/리셋 주소
            post_url=url_for("genai.chat"),
            reset_url=url_for("genai.chat") + "?reset=1",
            bot_mode_rag=False,
            bot_mode_agent=False,
        )
    else:
        guest_msgs = _guest_chat_msgs()
        fake_conv = type("Conv", (), {"title": "게스트 상담 챗봇"})
        return render_template(
            "genai/genai_chat.html",
            messages=guest_msgs,
            conversation=fake_conv,
            is_guest=True,
            # 이 페이지에서의 전송/리셋 주소
            post_url=url_for("genai.chat"),
            reset_url=url_for("genai.chat") + "?reset=1",
            bot_mode_rag=False,
            bot_mode_agent=False,
        )


# =========================================================
# 인지 스크리닝 화면 (/genai/audio)
# =========================================================
@bp.route("/audio", methods=["GET"])
def audio():
    return render_template("genai/genai_audio.html")


# =========================================================
# LLM 어댑터 (지연 import / 로컬·Gemini 겸용)
# =========================================================
def _make_openai_compat_chat(base_url: str, model: str):
    """
    OpenAI 호환 Chat Completions. 타임아웃/토큰/중단토큰을 설정값에서 읽고,
    타임아웃 시 더 짧은 max_tokens로 1회 재시도.
    """
    base = (base_url or "").rstrip("/") or "http://127.0.0.1:11434"

    def _chat(messages: list[dict], **gen_kwargs) -> str:
        import requests

        timeout_sec = int(current_app.config.get("GENAI_HTTP_TIMEOUT_SEC", 180))
        max_new = int(gen_kwargs.get(
            "max_new_tokens",
            current_app.config.get("GENAI_MAX_NEW_TOKENS", 120),
        ))
        stop = current_app.config.get("GENAI_STOP", ["\n\n"])

        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(current_app.config.get("GENAI_TEMPERATURE", 0.2)),
            "top_p": float(current_app.config.get("GENAI_TOP_P", 0.85)),
            "max_tokens": max_new,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        try:
            r = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=timeout_sec)
            r.raise_for_status()
            data = r.json()
            return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

        except requests.Timeout:
            try:
                payload["max_tokens"] = min(60, max_new)
                r2 = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=timeout_sec)
                r2.raise_for_status()
                data2 = r2.json()
                return (data2.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            except Exception as e2:
                current_app.logger.exception("OpenAI-compat retry failed: %s", e2)
                return "연결이 불안정해요. 잠시 후 다시 시도해 주세요."

        except Exception as e:
            current_app.logger.exception("OpenAI-compat call failed: %s", e)
            return "지금은 답변 생성이 원활하지 않습니다. 잠시 뒤 다시 시도해 주세요."

    return _chat


def _get_ollama_loader():
    base = (current_app.config.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
    model = current_app.config.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q5_K_M")
    return SimpleNamespace(chat=_make_openai_compat_chat(base, model))


def _get_llamacpp_loader():
    base = (current_app.config.get("LLAMACPP_BASE_URL") or "http://127.0.0.1:8080").rstrip("/")
    model = current_app.config.get("LLAMACPP_MODEL", "local-llama")
    return SimpleNamespace(chat=_make_openai_compat_chat(base, model))


def _get_gemini_loader():
    """Google Generative AI 어댑터 (타입 경고 없이 GenerationConfig 사용)"""
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted

    api_key = (
        current_app.config.get("GEMINI_API_KEY")
        or current_app.config.get("GOOGLE_API_KEY")
        or None
    )
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY가 필요합니다.")
    genai.configure(api_key=api_key)

    model_name = current_app.config.get("GEMINI_MODEL", "gemini-1.5-flash")
    system_ko = (
        "너는 한국어로 편안하게 대화하는 상담사다. 검사용 문항(기억 단어 제시, 계산, 점수/진단 언급)은 절대 하지 말라. "
        "항상 1–2문장으로 공감→짧은 질문 순으로 말한다. 같은 질문을 두 번 실패하면 다른 화제로 전환한다. "
        "대화 전환 규칙:\n"
        " - [회피/거부]('몰라','모르겠어','그냥 그래' 등): 허용·공감 후 '선택지 제시' 또는 '감각 단서(보이는 것/먹은 것/날씨)'로 좁혀 묻는다.\n"
        " - [반복/고집] 같은 내용이 2회 반복되면 요약 한 문장 후 화제를 옮긴다(예: 오늘 식사/창밖/TV/가까운 사람).\n"
        " - [혼란/막힘] 말이 막히면 질문을 더 구체화하고 예시 2개만 제시한다(예: 산책/TV처럼). '왜' 질문은 피하고 '무엇/어떻게' 위주로 묻는다.\n"
        " - [부정 감정] 짧게 정서 반영 후 선택지 2개로 묻는다(지금은 쉬고 싶은지, 가벼운 얘기를 이어갈지).\n"
        "스타일: 존댓말, 따뜻하고 단정한 어조, 의료·진단 표현 금지. 매 턴 끝에 부드러운 후속 질문 1개만. 마지막 턴에서는 질문 없이 요약만 한다."
    )

    class _Gemini:
        def __init__(self):
            self.model = genai.GenerativeModel(model_name, system_instruction=system_ko)

        def chat(self, messages: List[Dict], **kw) -> str:
            turns = []
            for m in messages:
                r, c = (m.get("role") or "user"), (m.get("content") or "").strip()
                if not c:
                    continue
                turns.append({"role": "model" if r == "assistant" else "user", "parts": [c]})

            try:
                try:
                    GenConfig = genai.types.GenerationConfig  # type: ignore[attr-defined]
                except Exception:
                    GenConfig = genai.GenerationConfig  # type: ignore[attr-defined]

                gc = GenConfig(
                    temperature=float(current_app.config.get("GENAI_TEMPERATURE", 0.2)),
                    top_p=float(current_app.config.get("GENAI_TOP_P", 0.85)),
                    max_output_tokens=int(kw.get("max_new_tokens", current_app.config.get("GENAI_MAX_NEW_TOKENS", 160))),
                )
                resp = self.model.generate_content(turns, generation_config=gc)
                return (getattr(resp, "text", None) or "").strip()
            except ResourceExhausted:
                return "잠시 후 다시 시도해 주세요."
            except Exception:
                cfg = {
                    "temperature": float(current_app.config.get("GENAI_TEMPERATURE", 0.2)),
                    "top_p": float(current_app.config.get("GENAI_TOP_P", 0.85)),
                    "max_output_tokens": int(kw.get("max_new_tokens", current_app.config.get("GENAI_MAX_NEW_TOKENS", 160))),
                }
                try:
                    resp = self.model.generate_content(turns, generation_config=cfg)  # type: ignore[arg-type]
                    return (getattr(resp, "text", None) or "").strip()
                except Exception:
                    return "잠시 후 다시 시도해 주세요."

    return _Gemini()


def get_llm():
    """앱 단위 LLM 인스턴스를 캐시해 반환한다."""
    store = current_app.extensions.setdefault("llm_store", {})
    provider = (current_app.config.get("GENAI_PROVIDER", "ollama") or "ollama").lower()

    if provider == "ollama":
        if "ollama" not in store:
            store["ollama"] = _get_ollama_loader()
        return store["ollama"]

    if provider == "llamacpp":
        if "llamacpp" not in store:
            store["llamacpp"] = _get_llamacpp_loader()
        return store["llamacpp"]

    key = f"gemini:{current_app.config.get('GEMINI_MODEL','gemini-1.5-flash')}"
    if key not in store:
        store[key] = _get_gemini_loader()
    return store[key]


# =========================================================
# 생성/후처리 유틸
# =========================================================
_HEAD_LABEL_RE = re.compile(r"(?mi)^(?:Assistant|User|System|SolCare)\s*:\s*")
_LABEL_PREFIX_RE = re.compile(r"(?mi)^(?:Korean|한국어|English|영어)\s*:\s*")
_TRAIL_WS_NL_RE = re.compile(r"\s+\n")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def _light_clean(text: str) -> str:
    if not text:
        return ""
    t = _HEAD_LABEL_RE.sub("", text)
    t = _LABEL_PREFIX_RE.sub("", t)
    t = _TRAIL_WS_NL_RE.sub("\n", t)
    t = _MULTI_NL_RE.sub("\n\n", t)
    return t.strip()


def _generate(llm, messages: List[Dict], **kw) -> str:
    return _light_clean(llm.chat(messages, **kw))


# =========================================================
# 휴리스틱 채점 유틸 (기존 그대로)
# =========================================================
_TOPIC_KW = {
    "meal": ["식사", "밥", "먹", "아침", "점심", "저녁", "커피", "빵"],
    "weather": ["날씨", "비", "해", "햇빛", "바람", "덥", "추"],
    "outside": ["산책", "공원", "나갔", "외출", "마트", "시장"],
    "media": ["TV", "티비", "드라마", "뉴스", "영화", "라디오"],
    "people": ["가족", "남편", "아내", "아들", "딸", "친구", "이웃"],
    "sleep": ["잠", "수면", "밤새", "깼", "피곤"],
    "mood": ["기분", "감정", "우울", "즐겁", "걱정", "불안"],
    "pain": ["아프", "통증", "허리", "무릎", "두통"],
    "house": ["집", "청소", "설거지", "빨래"],
}
_KO_FILLERS_RE = re.compile(r"(?:\b|^)(음+|엄+|어+|그[음]+|저[기]+|그니까|그러니까)(?:\b|$)")
_STOPWORDS = {"은", "는", "이", "가", "을", "를", "에", "도", "과", "와", "으로", "해서", "하다", "하고", "근데", "그냥", "뭔가", "좀", "조금"}
_DEICTIC_RE = re.compile(r"\b(그거|이거|저거|거기|저기|그곳|이곳|저곳)\b")


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w가-힣\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _content_words(s: str):
    return [w for w in _norm(s).split() if w not in _STOPWORDS and len(w) > 1]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(1, union)


def _detect_topics(text: str):
    t = set()
    for k, kws in _TOPIC_KW.items():
        if any(kw in text for kw in kws):
            t.add(k)
    return t


def _get_asked_topics() -> set:
    return set(session.get("asked_topics", []))


def _remember_topics_from_text(text: str) -> None:
    topics = _detect_topics(text or "")
    if not topics:
        return
    cur = _get_asked_topics()
    cur.update(topics)
    session["asked_topics"] = list(cur)


def _heuristic_scores(history: List[Dict]) -> Dict[str, int]:
    user_utts = [m["content"] for m in history if m.get("role") == "user"]
    last = user_utts[-1] if user_utts else ""
    last_words = set(_content_words(last))

    short = len(last_words) < 3
    dontknow = bool(re.search(r"(몰라|모르겠)", last))
    coherence = 0 if (short or dontknow) else (1 if len(last_words) < 8 else 2)

    rep_sim = 0.0
    for prev in user_utts[-4:-1]:
        rep_sim = max(rep_sim, _jaccard(last_words, set(_content_words(prev))))
    repetition = 0 if rep_sim >= 0.6 else (1 if rep_sim >= 0.35 else 2)

    fillers = len(_KO_FILLERS_RE.findall(last))
    fluency = 0 if fillers >= 6 else (1 if fillers >= 3 else 2)

    deictic = len(_DEICTIC_RE.findall(last))
    word_choice = 0 if deictic >= 4 else (1 if deictic >= 2 else 2)

    inconsist = 0
    if len(user_utts) >= 2:
        prev = user_utts[-2]
        if re.search(r"없었", prev) and re.search(r"(했|했다|다녀)", last):
            inconsist = 1
        if dontknow and len(last_words) >= 10:
            inconsist = 1
    consistency = 1 if inconsist else 2

    return {
        "coherence": coherence,
        "repetition": repetition,
        "fluency": fluency,
        "word_choice": word_choice,
        "consistency": consistency,
    }


def _merge_scores(heur: Dict[str, int], llm: Dict[str, int]) -> (Dict[str, int], int):
    keys = ["coherence", "repetition", "fluency", "word_choice", "consistency"]
    merged = {}
    for k in keys:
        h = int(heur.get(k, 0))
        l = int((llm or {}).get(k, 0))
        merged[k] = round(0.6 * h + 0.4 * l)
    total = sum(merged.values())
    return merged, total


# =========================================================
# 대화 평가(종료 시 1회)
# =========================================================
EVAL_PROMPT = (
    "다음은 상담사(assistant)와 사용자(user)의 한국어 일상 대화 기록이다.\n"
    "사용자의 언어/인지적 특징을 일상 대화 속에서만 평가하라. 테스트성 질문은 무시하라. 중간값 남발 금지, 엄격 채점.\n"
    "각 항목을 0=나쁨, 1=보통, 2=좋음으로 채점하라: "
    "coherence(맥락 일치), repetition(반복), fluency(유창성), word_choice(어휘 적절), consistency(일관성). "
    "총점은 합계(최대 10점).\n"
    "채점 규칙: 애매하면 1점을 부여하고, 2점은 명확히 우수할 때만 허용한다. 0점은 관찰되면 즉시 부여한다.\n"
    'JSON만 출력: {"scores":{"coherence":int,"repetition":int,"fluency":int,"word_choice":int,"consistency":int},"total":int}\n'
    "대화:\n"
)


def _evaluate_conversation_with_llm(history: List[Dict]) -> Dict[str, object]:
    heur = _heuristic_scores(history)

    llm = get_llm()
    transcript = []
    for m in history:
        r, c = m.get("role"), (m.get("content") or "").strip()
        if r in ("user", "assistant") and c:
            transcript.append(f"{r}: {c}")
    prompt = EVAL_PROMPT + "\n".join(transcript)

    raw = _generate(llm, [{"role": "user", "content": prompt}], max_new_tokens=220)
    try:
        s, e = raw.find("{"), raw.rfind("}")
        js = json.loads(raw[s:e+1])
        llm_scores = js.get("scores") or {}
    except Exception:
        llm_scores = {}

    merged_scores, merged_total = _merge_scores(heur, llm_scores)
    coh = merged_scores.get("coherence", 0)
    cst = merged_scores.get("consistency", 0)

    if merged_total >= 9 and coh >= 2 and cst >= 2:
        label, advice = "정상", "현재는 상담 권고 대상은 아닙니다. 변화가 느껴지면 다시 검사하세요."
    elif merged_total >= 6:
        label, advice = "주의 필요", "반복/맥락 흔들림이 관찰됩니다. 변화가 지속되면 보건소/치매안심센터 상담을 고려하세요."
    else:
        label, advice = "의심됨", "전문가 상담을 권합니다. 가까운 보건소·치매안심센터 또는 병원에서 평가를 받아보세요."

    return {"scores": merged_scores, "total": merged_total, "max": 10, "label": label, "advice": advice}


# =========================================================
# 결과 저장(종료 시 1회, 로그인 사용자만)
# =========================================================
def _save_final_result_for_user(result: Dict[str, object]) -> Optional[int]:
    if not _is_logged_in():
        return None
    conv = Conversation(user_id=g.user.id, title="인지 스크리닝 결과")
    db.session.add(conv)
    db.session.flush()

    payload = {"type": "screening_v1", "result": result}
    text = (
        f"[스크리닝 결과] 총점 {_to_int(result.get('total'), 0)}/{_to_int(result.get('max'), 10)} → {str(result.get('label',''))}\n"
        f"권고: {str(result.get('advice',''))}\n"
        "※ 본 도구는 참고용이며, 의학적 진단이 아닙니다.\n\n"
        f"[[SCREEN_JSON]]{json.dumps(payload, ensure_ascii=False)}[[/SCREEN_JSON]]"
    )
    db.session.add(Message(conversation_id=conv.id, role="assistant", content=text))
    db.session.commit()
    return conv.id


def _save_screening_result_row(result: Dict[str, object], summary_text: str) -> Optional[int]:
    if not _is_logged_in():
        return None
    row = ScreeningResult(
        user_username=g.user.username,
        total_score=_to_int(result.get("total"), 0),
        max_score=_to_int(result.get("max"), 10),
        result_summary=str(result.get("label", "")),
        need_referral=(str(result.get("label", "")) != "정상"),
        advice=str(result.get("advice", "")),
        result_text=str(summary_text or ""),
    )
    db.session.add(row)
    db.session.commit()
    return row.id


# =========================================================
# 대화 API (/genai/audio/converse)
# =========================================================
@bp.route("/audio/converse", methods=["POST"])
def audio_converse():
    try:
        data = request.get_json(silent=True) or {}

        if data.get("reset"):
            for k in ("screen_history", "screen_turns", "asked_topics"):
                session.pop(k, None)
            return jsonify(ok=True), 200

        user_text = (data.get("text") or "").strip()
        if not user_text:
            return jsonify(reply="편하게 아무 이야기나 시작해 주세요."), 200

        history: List[Dict[str, str]] = session.get("screen_history", [])
        history.append({"role": "user", "content": user_text})

        turns_prev = int(session.get("screen_turns", 0))
        turns = turns_prev + 1
        target_turns = int(current_app.config.get("SCREEN_TURNS", 8))
        is_final = turns >= target_turns

        if is_final:
            session["screen_history"] = history
            result = _evaluate_conversation_with_llm(history)
            summary = (
                "여기까지의 대화를 바탕으로 간단히 정리해 드릴게요.\n"
                f"- 총점: {result['total']} / {result['max']} → {result['label']}\n"
                f"- 권고: {result['advice']}\n"
                "이 도구는 교육·연구용 참고 도구이며, 의학적 진단이 아닙니다."
            )
            conv_id = _save_final_result_for_user(result)
            _save_screening_result_row(result, summary)
            session.pop("screen_history", None)
            session.pop("screen_turns", None)
            session.pop("asked_topics", None)
            return jsonify(reply=summary, conversation_id=conv_id), 200

        avoid = ", ".join(sorted(_get_asked_topics())) or "없음"
        system = (
            "너는 한국어로 자연스러운 일상 대화를 이어가는 상담사다. "
            "검사/점수/진단/테스트 같은 단어를 쓰지 말라. "
            "단어 기억 제시나 숫자 계산을 요구하지 말라. "
            "한 번에 1~2문장으로 공감하고 마지막에 부드러운 후속 질문 1개만 해라. "
            "이미 다룬 화제는 반복하지 말고 다른 화제로 전환하라. "
            f"이미 다룬 화제(피해야 함): {avoid}. "
            "회피/거부가 있으면 허용·공감 후 선택지 2개나 감각 단서로 좁혀 묻는다."
        )

        llm = get_llm()
        reply = _generate(llm, [{"role": "system", "content": system}] + history, max_new_tokens=160) \
                or "잘 들었습니다. 조금 더 자세히 말씀해 주실 수 있을까요?"

        history.append({"role": "assistant", "content": reply})
        session["screen_history"] = history
        session["screen_turns"] = turns
        _remember_topics_from_text(reply)

        return jsonify(reply=reply, conversation_id=None), 200

    except Exception as e:
        current_app.logger.exception("audio_converse error")
        return jsonify(reply="서버 설정에 잠시 문제가 있어요. 다시 시도해 주세요.", error=str(e)), 200


# =========================================================
# (통합) 봇 챗 ― /genai/bot, /genai/bot/api/message, /genai/bot/api/reset
# =========================================================
def _bot_history() -> List[Dict[str, str]]:
    return session.get("bot_history", [])


def _save_bot_history(hist: List[Dict[str, str]]):
    max_msgs = int(current_app.config.get("GENAI_MAX_MESSAGES", 80))
    session["bot_history"] = hist[-max_msgs:]


def _reset_bot_session():
    session.pop("bot_history", None)


def _bot_system_prompt() -> str:
    return (
        "너는 한국어로만 간결, 정확, 친절하게 대답하는 도우미다. "
        "다른 언어는 요청이 없으면 절대 출력하지 않는다."
        "이전 발언은 잊었다 말하지 않는다. "
        "확실치 않은 내용은 추측하지 않고, 모른다고 답한다. "
        "사과는 1회 이내로 한다. "
        "답변은 기본 3~6문장 정도로 한다. "
        "사용자가 상세한 설명 요청 시 300자 이내로 답변한다. "
        "참고자료가 주어지면 그 범위 안에서만 답한다."
    )


def _call_llm(messages: List[Dict], max_new_tokens: int = 200) -> str:
    llm = get_llm()
    try:
        return (llm.chat(messages, max_new_tokens=max_new_tokens) or "").strip()
    except Exception as e:
        current_app.logger.exception("LLM call failed: %s", e)
        return "지금은 답변 생성이 원활하지 않습니다. 질문을 조금만 바꿔 다시 시도해 보시겠어요?"


def _postprocess_for_display(full_text: str, max_chars: int = 200) -> Dict[str, str]:
    t = (full_text or "").strip()
    if not t:
        return {"brief": "", "keywords": "", "html": "", "full": ""}

    if max_chars is None:
        max_chars = int(current_app.config.get("BOT_MAX_DISPLAY_CHARS", 500))

    window = t[: max_chars + 120]
    end = max(
        window.rfind("다."),
        window.rfind("요."),
        window.rfind("."),
        window.rfind("!"),
        window.rfind("?"),
    )

    if 120 <= end <= max_chars + 120:
        cut = window[: end + 1]
    else:
        cut = t[: max_chars].rstrip() + "…"

    esc = html.escape
    html_out = esc(cut)
    return {"brief": cut, "keywords": "", "html": html_out, "full": t}


# (선택) RAG: 없으면 조용히 비활성
try:
    from ..rag import rag_search_snippets as _rag_search_impl
except Exception:
    _rag_search_impl = None


@bp.route("/bot", methods=["GET"])
def bot_view():
    # reset=1 쿼리로 히스토리 초기화
    if request.args.get("reset") == "1":
        session.pop("bot_history", None)
        return redirect(url_for("genai.bot_view"))

    return render_template(
        "genai/genai_chat.html",
        messages=_bot_history(),
        bot_mode_rag=str(current_app.config.get("BOT_RAG_MODE", "false")).lower() == "true",
        bot_mode_agent=str(current_app.config.get("BOT_AGENT_MODE", "false")).lower() == "true",
        # NEW: 이 페이지에서의 전송/리셋 주소
        post_url=url_for("genai.bot_api_message"),
        reset_url=url_for("genai.bot_api_reset"),
        is_guest=True,
        conversation=type("Conv", (), {"title": "AI 검색·요약(에이전트)"}),  # 템플릿 호환용
    )


@bp.route("/bot/api/reset", methods=["POST"])
def bot_api_reset():
    _reset_bot_session()
    return jsonify(ok=True)


@bp.route("/bot/api/message", methods=["POST"])
def bot_api_message():
    data = request.get_json(silent=True) or request.form or {}
    user_text = (data.get("text") or data.get("message") or "").strip()
    if not user_text:
        return jsonify(reply="무엇을 도와드릴까요?"), 200

    hist = _bot_history()
    hist.append({"role": "user", "content": user_text})

    use_rag = str(current_app.config.get("BOT_RAG_MODE", "false")).lower() == "true"
    use_agent = str(current_app.config.get("BOT_AGENT_MODE", "false")).lower() == "true"

    # 표시 길이: 기본/에이전트·RAG 구분
    display_limit = int(current_app.config.get("BOT_MAX_DISPLAY_CHARS", 500))
    if use_rag or use_agent:
        display_limit = int(current_app.config.get("BOT_MAX_DISPLAY_CHARS_RAG", 900))

    # ---------- Agent 우선 시도 ----------
    if use_agent:
        try:
            from pybo.agent.runner import agent_run
            full = agent_run(user_text, hist, get_llm)
            pp = _postprocess_for_display(full, max_chars=display_limit)
            hist.append({"role": "assistant", "content": pp["html"]})
            _save_bot_history(hist)
            return jsonify(reply=pp["html"], keywords=pp["keywords"], mode="agent"), 200
        except Exception as e:
            current_app.logger.warning("Agent failed: %s", e)

    messages = [{"role": "system", "content": _bot_system_prompt()}] + hist

    if use_rag and _rag_search_impl is not None:
        try:
            top_k = int(current_app.config.get("RAG_TOP_K", 3))
            ctx = _rag_search_impl(user_text, top_k=top_k)
            if ctx and not str(ctx).startswith("[error]"):
                messages.insert(1, {
                    "role": "system",
                    "content": f"다음 참고자료를 바탕으로만 답하라(스니펫+출처 힌트):\n{ctx[:3500]}"
                })
        except Exception as e:
            current_app.logger.warning("RAG failed: %s", e)

    max_new = int(current_app.config.get("GENAI_MAX_NEW_TOKENS", 320))
    full = _call_llm(messages, max_new_tokens=max_new)
    pp = _postprocess_for_display(full, max_chars=display_limit)

    hist.append({"role": "assistant", "content": pp["html"]})
    _save_bot_history(hist)
    return jsonify(reply=pp["html"], keywords=pp["keywords"], mode="chat_rag" if use_rag else "chat"), 200
