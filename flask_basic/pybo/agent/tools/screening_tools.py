# pybo/agent/tools/screening_tools.py
from __future__ import annotations
from typing import Dict, List, Callable, Optional
import json, re
from flask import current_app, session

# ===== 휴리스틱 유틸(순환 import 피하기 위해 이 파일에 포함) =====
_KO_FILLERS_RE = re.compile(r"(?:\b|^)(음+|엄+|어+|그[음]+|저[기]+|그니까|그러니까)(?:\b|$)")
_STOPWORDS = {"은","는","이","가","을","를","에","도","과","와","으로","해서","하다","하고","근데","그냥","뭔가","좀","조금"}
_DEICTIC_RE = re.compile(r"\b(그거|이거|저거|거기|저기|그곳|이곳|저곳)\b")
_TOPIC_KW = {
    "meal": ["식사","밥","먹","아침","점심","저녁","커피","빵"],
    "weather": ["날씨","비","해","햇빛","바람","덥","추"],
    "outside": ["산책","공원","나갔","외출","마트","시장"],
    "media": ["TV","티비","드라마","뉴스","영화","라디오"],
    "people": ["가족","남편","아내","아들","딸","친구","이웃"],
    "sleep": ["잠","수면","밤새","깼","피곤"],
    "mood": ["기분","감정","우울","즐겁","걱정","불안"],
    "pain": ["아프","통증","허리","무릎","두통"],
    "house": ["집","청소","설거지","빨래"],
}

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

def _heuristic_scores(history: List[Dict[str, str]]) -> Dict[str, int]:
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

# ===== 스크리닝 대화 한 턴(기존 /audio/converse 흐름 그대로) =====
def screening_reset() -> Dict:
    for k in ("screen_history", "screen_turns", "asked_topics"):
        session.pop(k, None)
    return {"ok": True}

def screening_turn(user_text: str, get_llm: Callable[[], object]) -> Dict:
    user_text = (user_text or "").strip()
    if not user_text:
        return {"reply": "편하게 아무 이야기나 시작해 주세요.", "final": False, "conversation_id": None}

    history: List[Dict[str, str]] = session.get("screen_history", [])
    history.append({"role": "user", "content": user_text})

    turns_prev = int(session.get("screen_turns", 0))
    turns = turns_prev + 1
    target_turns = int(current_app.config.get("SCREEN_TURNS", 8))
    is_final = turns >= target_turns

    if is_final:
        # LLM 평가(간략 버전: 휴리스틱 + LLM JSON 채점)
        heur = _heuristic_scores(history)
        llm = get_llm()
        transcript = []
        for m in history:
            r, c = m.get("role"), (m.get("content") or "").strip()
            if r in ("user","assistant") and c:
                transcript.append(f"{r}: {c}")
        prompt = (
            "다음은 상담사(assistant)와 사용자(user)의 한국어 일상 대화 기록이다.\n"
            "검사성 문항은 무시하고, 일상 대화에서만 관찰 가능한 특징으로 평가하라.\n"
            "각 항목을 0/1/2로 채점: coherence, repetition, fluency, word_choice, consistency.\n"
            'JSON만 출력: {"scores":{"coherence":int,"repetition":int,"fluency":int,"word_choice":int,"consistency":int},"total":int}\n'
            "대화:\n" + "\n".join(transcript)
        )
        raw = (llm.chat([{"role":"user","content":prompt}], max_new_tokens=220) or "").strip()
        try:
            s,e = raw.find("{"), raw.rfind("}")
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

        summary = f"정리드릴게요. 총점 {merged_total}/10 → {label}. 권고: {advice}"
        # 세션 정리
        session.pop("screen_history", None)
        session.pop("screen_turns", None)
        session.pop("asked_topics", None)
        return {"reply": summary, "final": True, "conversation_id": None}

    # 진행 턴: 시스템 지시 + 한 턴 생성
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
    reply = (llm.chat([{"role":"system","content":system}] + history, max_new_tokens=160) or "").strip()

    history.append({"role":"assistant","content":reply})
    session["screen_history"] = history
    session["screen_turns"] = turns
    _remember_topics_from_text(reply)

    # 200자 이내로 잘라 반환(에이전트 출력 정책 일관)
    cut = reply[:200].rstrip()
    if len(reply) > 200:
        cut += "…"
    return {"reply": cut, "final": False, "conversation_id": None}

# ===== 간단 자가진단(현재까지 대화 기반 200자 요약) =====
def selfcheck_brief(get_llm: Callable[[], object]) -> Dict:
    history: List[Dict[str,str]] = session.get("screen_history", [])
    if not history:
        return {"reply":"최근 대화가 없어요. 간단한 안부부터 이야기 나눠볼까요?", "final": False}

    heur = _heuristic_scores(history)
    msg = (
        f"간단 자가점검: 일관성 {heur['consistency']}/2, 맥락 {heur['coherence']}/2, 반복 {heur['repetition']}/2, "
        f"유창성 {heur['fluency']}/2, 어휘 {heur['word_choice']}/2."
    )
    return {"reply": msg[:200], "final": False}
