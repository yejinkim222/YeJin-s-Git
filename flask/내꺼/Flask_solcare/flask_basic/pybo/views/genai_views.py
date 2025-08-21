# pybo/views/genai_views.py
from __future__ import annotations
import os, secrets, re
from functools import lru_cache
from typing import List, Dict, Optional
from flask import Blueprint, render_template, request, redirect, url_for, g, session, current_app

from .. import db
from pybo.models import Conversation, Message
from pybo.rag.retriever import get_retriever

from typing import List, Dict, Optional
from flask import current_app



# =========================
# 전역 정규식/상수
# =========================
_HEAD_LABEL_RE   = re.compile(r"(?mi)^(?:Assistant|User|System|SolCare)\s*:\s*")
_TRAIL_WS_NL_RE  = re.compile(r"\s+\n")
_MULTI_NL_RE     = re.compile(r"\n{3,}")

_RX_NOISY_TOKENS = re.compile(
    r"""
    (?:\[\s*(?:SYS|INST|USER|ASSISTANT)\s*\])       # [SYS][INST]...
  | (?:\[\s*/\s*(?:SYS|INST)\s*\])                  # [/SYS][/INST]
  | (?:</?s>)                                       # <s> </s>
  | (?:<\|\s*/?(?:im_start|im_end|user|assistant|system)\s*\|>)  # ChatML
  | (?:<\|\s*endoftext\s*\|>)                       # <|endoftext|>
  | (?:<<\s*/?SYS\s*>>)                             # <<SYS>><</SYS>>
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RX_QWEN_ASSIST_1 = re.compile(r"<\|assistant\|>\s*(.*?)\s*(?:<\|im_end\|>|<\|user\|>|$)", re.I | re.S)
_RX_QWEN_ASSIST_2 = re.compile(r"<\|im_start\|>\s*assistant\s*\n(.*?)\s*(?:<\|im_end\|>|$)", re.I | re.S)

_NOISY_LINE_ECHO_RE = re.compile(
    r"(?i)^(?:you are a helpful assistant|reply in korean|as a helpful assistant|system:|assistant:|user:)\b"
)

_LABEL_PREFIX_RE = re.compile(r"(?mi)^(?:Korean|한국어|English|영어)\s*:\s*")

bp = Blueprint("genai", __name__, url_prefix="/genai")


# =========================
# 1) LLM 로더(캐시 버스트 가능)
# =========================
def _load_llm_cached(model_id: str, do_sample_default: bool, max_new_tokens: int):
    """
    실제 모델 로드를 수행하는 내부 함수입니다.
    lru_cache 키에 모델/옵션을 넣어, 값이 바뀌면 자동으로 재빌드됩니다.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=True,
    ).eval()

    qwen_eos_id = tok.convert_tokens_to_ids("<|im_end|>") or tok.eos_token_id

    gen_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tok,
        do_sample=do_sample_default,                      # 기본은 False로 전달할 예정
        temperature=float(current_app.config.get("GENAI_TEMPERATURE", 0.0)),
        top_p=float(current_app.config.get("GENAI_TOP_P", 1.0)),
        max_new_tokens=max_new_tokens,
        repetition_penalty=float(current_app.config.get("GENAI_REPETITION_PENALTY", 1.05)),
        no_repeat_ngram_size=int(current_app.config.get("GENAI_NO_REPEAT_NGRAM_SIZE", 4)),
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        eos_token_id=qwen_eos_id,
        return_full_text=False,
        num_return_sequences=1,
    )

    class LocalChat:
        """
        HF 파이프라인을 감싸는 간단한 채팅 래퍼입니다.
        - _run()에서 GENAI_DEBUG=True이면 프롬프트/원문 로그를 남깁니다.
        - model_id를 저장해 family 추정에도 사용합니다.
        """

        def __init__(self, pipe, tok, model_id, eos_id):
            self.pipe = pipe
            self.tokenizer = tok
            self.model_id = model_id
            self._eos_id = eos_id

        def chat(self, messages: List[Dict], **gen_kwargs) -> str:
            prompt = _build_chatml_from_messages(messages)
            return self._run(prompt, **gen_kwargs)

        def __call__(self, prompt_or_messages, **gen_kwargs) -> str:
            if isinstance(prompt_or_messages, list):
                prompt = _build_chatml_from_messages(prompt_or_messages)
            else:
                prompt = _build_chatml_from_messages([
                    {"role": "system", "content": "당신은 한국어로만 간결하게 답변하는 비서입니다."},
                    {"role": "user", "content": str(prompt_or_messages)},
                ])
            return self._run(prompt, **gen_kwargs)

        def _run(self, input_text: str, **gen_kwargs) -> str:
            if current_app.config.get("GENAI_DEBUG"):
                current_app.logger.info("GENAI PROMPT >>>\n%s", input_text)

            out = self.pipe(
                input_text,
                eos_token_id=self._eos_id,
                max_new_tokens=gen_kwargs.get("max_new_tokens", None),
                temperature=gen_kwargs.get("temperature", None),
                top_p=gen_kwargs.get("top_p", None),
                do_sample=gen_kwargs.get("do_sample", None),
                repetition_penalty=gen_kwargs.get("repetition_penalty", None),
                no_repeat_ngram_size=gen_kwargs.get("no_repeat_ngram_size", None),
            )
            text = (out[0].get("generated_text") if isinstance(out, list) and out and isinstance(out[0], dict)
                    else str(out)) or ""
            text = text.strip()

            if current_app.config.get("GENAI_DEBUG"):
                current_app.logger.info("GENAI RAW >>>\n%s", text)

            return text

    return LocalChat(gen_pipe, tok, model_id, qwen_eos_id)


@lru_cache(maxsize=8)
def _cached_entry(model_key: str, cache_salt: str, do_sample_default: bool, max_new_tokens: int):
    """
    캐시 엔트리 생성자입니다.
    model_key, cache_salt, do_sample_default, max_new_tokens가 바뀌면 재빌드됩니다.
    """
    return _load_llm_cached(model_key, do_sample_default, max_new_tokens)


def get_llm():
    """
    Qwen 2.5–3B Instruct를 기본으로 사용하며, 캐시 버스트가 가능하도록 구성합니다.
    - GENAI_MODEL / HF_MODEL_ID / GENAI_CACHE_SALT 값이 바뀌면 새로 로드합니다.
    - 기본 do_sample=False로 고정합니다.
    """
    presets = {
        "qwen2.5-3b":          "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-1.5b":        "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5-0.5b":        "Qwen/Qwen2.5-0.5B-Instruct",
    }
    picked = (current_app.config.get("GENAI_MODEL")
              or current_app.config.get("HF_MODEL_ID")
              or "qwen2.5-3b").lower()
    model_key = presets.get(picked, picked)

    # 캐시 버스트용 소금값: 배포/디버그 시 바꿔주면 재빌드됩니다.
    cache_salt = str(current_app.config.get("GENAI_CACHE_SALT", "v1"))

    # 기본은 탐색 끔
    do_sample_default = False
    max_new_tokens = int(current_app.config.get("GENAI_MAX_NEW_TOKENS", 200))

    return _cached_entry(model_key, cache_salt, do_sample_default, max_new_tokens)


# =========================
# 2) ChatML 빌더/정리
# =========================
def _sanitize_history_content(s: str) -> str:
    """
    히스토리(특히 과거 assistant 출력)에 섞인 템플릿/역할 토큰/라벨/에코 라인을 제거합니다.
    """
    if not s:
        return ""
    s = _RX_NOISY_TOKENS.sub("", s)
    s = _LABEL_PREFIX_RE.sub("", s)
    s = _HEAD_LABEL_RE.sub("", s)
    lines = [ln for ln in s.splitlines() if not _NOISY_LINE_ECHO_RE.search(ln.strip())]
    s = "\n".join(lines)
    s = _TRAIL_WS_NL_RE.sub("\n", s)
    s = _MULTI_NL_RE.sub("\n\n", s)
    return s.strip()


def _build_chatml_from_messages(messages: List[Dict]) -> str:
    """
    messages(list[dict])를 Qwen ChatML 문자열로 구성합니다.
    system → 턴들 → assistant 시작 순서로 이어 붙입니다.
    """
    sys = next((m["content"] for m in messages if m.get("role") == "system"), "")
    parts = []
    if sys:
        parts.append(f"<|im_start|>system\n{sys}<|im_end|>")

    for m in messages:
        r, c = m.get("role"), (m.get("content") or "").strip()
        if not c or r == "system":
            continue
        c = _sanitize_history_content(c)
        parts.append(f"<|im_start|>{r}\n{c}<|im_end|>")

    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# =========================
# 3) 히스토리 → messages
# =========================
def _build_messages_from_history(history: Optional[List[Dict]]) -> List[Dict]:
    """
    히스토리를 정규화하여 messages(list[dict])를 만든다.
    - 맨 앞에 system 지침을 둔다(한국어로만, 템플릿/역할 토큰 출력 금지).
    - 마지막 user 발화 끝에는 '한국어만 사용' 지침을 일시적으로 추가한다(저장은 안 함).
    - 또한 마지막 user 발화에 RAG 검색 결과(상위 k개)를
      [참고문서] 블록으로 덧붙인다(DB/세션에는 저장되지 않음).
    입력:
      - history: list[dict] | None (각 dict는 {"role": "...", "content": "..."} 형태)
    출력:
      - messages: list[dict]  (system 포함)
    """
    messages: List[Dict] = [{
        "role": "system",
        "content": (
            "당신은 한국어로만 간결하게 답변하는 비서입니다. "
            "질문에 직접적으로만 답하고, 템플릿/역할 토큰([SYS],[USER],[INST],<|im_*|>,</s>)은 절대 출력하지 마세요. "
            "모호하거나 자료가 없으면 '모르겠습니다'라고 답하세요."
        ),
    }]

    for m in (history or []):
        role = (m.get("role") if isinstance(m, dict) else getattr(m, "role", "user")) or "user"
        content = (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
        if content:
            messages.append({"role": role, "content": content})

    # 마지막 user 발화 보강(한국어 강제 지침 + RAG 컨텍스트)
    if len(messages) >= 2 and messages[-1].get("role") == "user":
        # 1) 한국어 강제 지침(중복 삽입 방지)
        marker = "답변은 반드시 한국어로만 작성하세요"
        if marker not in messages[-1]["content"]:
            messages[-1]["content"] = (
                messages[-1]["content"].rstrip()
                + "\n\n(지침: 답변은 반드시 한국어로만 작성하세요. 영어를 사용하지 마세요.)"
            )

        # 2) RAG 컨텍스트 주입(있을 때만)
        hits = _retrieve_knowledge(messages[-1]["content"])
        if hits:
            lines = [
                "\n\n[참고문서]",
                "(아래 내용만을 근거로 한국어로 간결하게 답하세요. 불확실하면 모른다고 답하세요.)"
            ]
            for h in hits:
                snippet = h["text"].replace("\n", " ").strip()
                lines.append(f"- {snippet} (출처: {h['source']})")
            messages[-1]["content"] += "\n".join(lines)

    return messages



# =========================
# 4) 생성 및 한국어-고정 안전망
# =========================
def _looks_like_english_or_echo(s: str) -> bool:
    """
    생성 결과가 영어 위주이거나, 시스템 문구/에코가 섞인 것처럼 보이면 True를 반환합니다.
    """
    if not s:
        return True
    # 에코 키워드
    if re.search(r"(?i)\b(reply in korean|you are a helpful assistant|as a helpful assistant)\b", s):
        return True
    # 영문 비중
    letters = re.findall(r"[A-Za-z]", s)
    non_space = re.findall(r"\S", s)
    if non_space and (len(letters) / max(1, len(non_space))) > 0.35:
        return True
    return False


def _extract_assistant_body_for_qwen(raw: str) -> str:
    """
    return_full_text=True 로 돌아오는 경우나 모델이 ChatML을 섞어 낸 경우,
    어시스턴트 본문만 추출합니다.
    """
    if not raw:
        return ""
    m = _RX_QWEN_ASSIST_1.search(raw) or _RX_QWEN_ASSIST_2.search(raw)
    return (m.group(1).strip() if m else raw.strip())


def _light_clean(text: str) -> str:
    """
    역할 라벨/라벨 접두사/여분 개행을 가볍게 정리합니다.
    """
    if not text:
        return ""
    t = _HEAD_LABEL_RE.sub("", text)
    t = _LABEL_PREFIX_RE.sub("", t)
    t = _TRAIL_WS_NL_RE.sub("\n", t)
    t = _MULTI_NL_RE.sub("\n\n", t)
    return t.strip()


def _generate_with_messages(llm, messages: List[Dict], **gen_kwargs) -> str:
    """
    1차 생성(도샘플 OFF) → 본문 추출/클린업 → 영어/에코 감지 시 2차 엄격 생성으로 재시도합니다.
    """
    raw = llm.chat(messages, **gen_kwargs) if hasattr(llm, "chat") else llm(messages, **gen_kwargs)
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "generated_text" in raw[0]:
        raw = raw[0]["generated_text"] or ""
    raw = str(raw)

    body = _light_clean(_extract_assistant_body_for_qwen(raw))
    if current_app.config.get("GENAI_DEBUG"):
        current_app.logger.info("GENAI BODY(1st) >>>\n%s", body)

    if not _looks_like_english_or_echo(body):
        return body or "(응답이 비어 있습니다)"

    # ---- 2차 재시도: 최근 user만 남긴 미니 대화 + 매우 강한 한국어 잠금 ----
    last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    strict_msgs = [
        {"role": "system", "content": "지금부터 한국어만 사용합니다. 답변은 1–3문장으로 간결하게 작성하세요."},
        {"role": "user",   "content": last_user.strip() if last_user else "간단히 인사해 주세요."},
    ]
    raw2 = llm.chat(strict_msgs, do_sample=False, temperature=0.0, top_p=1.0,
                    max_new_tokens=gen_kwargs.get("max_new_tokens", None))
    if isinstance(raw2, list) and raw2 and isinstance(raw2[0], dict) and "generated_text" in raw2[0]:
        raw2 = raw2[0]["generated_text"] or ""
    body2 = _light_clean(_extract_assistant_body_for_qwen(str(raw2)))

    if current_app.config.get("GENAI_DEBUG"):
        current_app.logger.info("GENAI BODY(2nd) >>>\n%s", body2)

    return body2 or "(응답이 비어 있습니다)"


# =========================
# 5) 공통/게스트 유틸
# =========================
def _is_logged_in() -> bool:
    """로그인 여부 판단."""
    return getattr(g, "user", None) is not None


def _ensure_anon_key() -> str:
    """게스트 세션 식별 키 생성/반환."""
    if "anon_key" not in session:
        session["anon_key"] = secrets.token_hex(16)
    return session["anon_key"]


def _get_or_create_user_conversation() -> Conversation:
    """로그인 사용자 기본 대화 조회/생성."""
    conv = (Conversation.query
            .filter_by(user_id=g.user.id)
            .order_by(Conversation.created_at.desc())
            .first())
    if not conv:
        conv = Conversation(user_id=g.user.id, title=f"{g.user.username}의 대화")
        db.session.add(conv); db.session.commit()
    return conv


def _trim_cap() -> int:
    """히스토리 전체 보존 상한."""
    return int(current_app.config.get("GENAI_MAX_MESSAGES", 80))


def _get_anon_messages() -> List[Dict]:
    """게스트 대화 히스토리 읽기."""
    return session.get("anon_messages", [])


def _save_anon_messages(msgs: List[Dict]) -> None:
    """게스트 대화 히스토리 저장."""
    session["anon_messages"] = msgs


def _append_anon(role: str, content: str) -> None:
    """게스트 히스토리에 추가하고 상한 초과분 제거."""
    msgs = _get_anon_messages()
    msgs.append({"role": role, "content": content})
    cap = _trim_cap()
    if len(msgs) > cap:
        msgs = msgs[-cap:]
    _save_anon_messages(msgs)


def _clear_anon() -> None:
    """게스트 히스토리 삭제."""
    session.pop("anon_messages", None)


# =========================
# 6) 라우트
# =========================
@bp.route("/chat", methods=["GET", "POST"])
def chat():
    """
    채팅 페이지 렌더 및 메시지 제출 처리.
    do_sample=True를 넘기지 않아 탐색이 켜지지 않도록 합니다.
    캐시 버스트는 GENAI_CACHE_SALT를 바꾸면 됩니다.
    """
    if (not _is_logged_in()) and (request.args.get("reset") == "1"):
        _clear_anon()

    if request.method == "POST":
        user_text = (request.form.get("message") or "").strip()
        if user_text:
            conv = None

            if _is_logged_in():
                conv = _get_or_create_user_conversation()
                db.session.add(Message(conversation_id=conv.id, role="user", content=user_text))
                db.session.commit()

                q = (Message.query
                     .filter_by(conversation_id=conv.id)
                     .order_by(Message.created_at.asc()))
                count = q.count()
                cap = _trim_cap()
                if count > cap:
                    to_delete = count - cap
                    old_ids = [m.id for m in q.limit(to_delete).all()]
                    if old_ids:
                        Message.query.filter(Message.id.in_(old_ids)).delete(synchronize_session=False)
                        db.session.commit()

                last_k = int(current_app.config.get("GENAI_MAX_CTX_MESSAGES", 16))
                msgs = (Message.query
                        .filter_by(conversation_id=conv.id)
                        .order_by(Message.created_at.desc())
                        .limit(last_k)
                        .all())
                msgs.reverse()
                history = [{"role": m.role, "content": m.content} for m in msgs]
            else:
                _append_anon("user", user_text)
                last_k = int(current_app.config.get("GENAI_MAX_CTX_MESSAGES", 16))
                history = _get_anon_messages()[-last_k:]

            messages = _build_messages_from_history(history)
            llm = get_llm()
            try:
                answer = _generate_with_messages(
                    llm, messages,
                    max_new_tokens=int(current_app.config.get("GENAI_MAX_NEW_TOKENS", 200)),
                    # do_sample을 넘기지 마세요(기본 False 유지)
                )
            except Exception as e:
                answer = f"(모델 호출 오류) {type(e).__name__}: {e}"

            if _is_logged_in():
                db.session.add(Message(conversation_id=conv.id, role="assistant", content=answer))
                db.session.commit()
            else:
                _append_anon("assistant", answer)

        return redirect(url_for("genai.chat"))

    if _is_logged_in():
        conv = _get_or_create_user_conversation()
        msgs = (Message.query
                .filter_by(conversation_id=conv.id)
                .order_by(Message.created_at.asc())
                .all())
        view_msgs = [{"role": m.role, "content": m.content} for m in msgs]
        return render_template("genai/chat.html", messages=view_msgs, conversation=conv, is_guest=False)
    else:
        anon_msgs = _get_anon_messages()
        fake_conv = type("Conv", (), {"title": f"게스트-{_ensure_anon_key()[:6]}의 대화"})
        return render_template("genai/chat.html", messages=anon_msgs, conversation=fake_conv, is_guest=True)


# rag 구현
def _retrieve_knowledge(query: str) -> List[Dict]:
    """
    하는 일:
      - RAG가 켜져 있으면 top-k 문단을 검색해 [{text, source, score}]를 반환한다.
      - 꺼져 있거나 인덱스가 없으면 빈 리스트를 반환한다.
    입력:
      - query: str  (마지막 사용자 발화)
    출력:
      - List[Dict]  각 항목은 {"text": str, "source": str, "score": float}
    """
    if not current_app.config.get("RAG_ENABLED", False):
        return []
    try:
        retr = get_retriever()
        top_k = int(current_app.config.get("RAG_TOP_K", 3))
        hits = retr.search(query, top_k=top_k)
        return [{"text": h.text, "source": h.source, "score": h.score} for h in hits]
    except Exception:
        # 인덱스가 아직 없거나 로드 실패 시 조용히 RAG 미사용
        return []
