from __future__ import annotations

import os
import re
import secrets
from functools import lru_cache
from typing import List, Dict, Optional

from flask import (
    Blueprint, render_template, request, redirect, url_for,
    g, session, current_app
)

from pybo import db
from pybo.models import Conversation, Message
from pybo.python.rag.retriever import get_retriever


bp = Blueprint("genai", __name__, url_prefix="/genai")

# =========================
# 0) 전역 정규식/상수
# =========================
_HEAD_LABEL_RE   = re.compile(r"(?mi)^(?:Assistant|User|System|SolCare)\s*:\s*")
_TRAIL_WS_NL_RE  = re.compile(r"\s+\n")
_MULTI_NL_RE     = re.compile(r"\n{3,}")
_RX_QWEN_ASSIST_1 = re.compile(r"<\|assistant\|>\s*(.*?)\s*(?:<\|im_end\|>|<\|user\|>|$)", re.I | re.S)
_RX_QWEN_ASSIST_2 = re.compile(r"<\|im_start\|>\s*assistant\s*\n(.*?)\s*(?:<\|im_end\|>|$)", re.I | re.S)
_RX_NOISY_TOKENS = re.compile(
    r"(?:\[\s*(?:SYS|INST|USER|ASSISTANT)\s*\])|(?:</?s>)|"
    r"(?:<\|/?(?:im_start|im_end|user|assistant|system)\|>)|"
    r"(?:<\|endoftext\|>)",
    re.IGNORECASE
)
_LABEL_PREFIX_RE = re.compile(r"(?mi)^(?:Korean|한국어|English|영어)\s*:\s*")
_NOISY_LINE_ECHO_RE = re.compile(
    r"(?i)^(?:you are a helpful assistant|reply in korean|as a helpful assistant|system:|assistant:|user:)\b"
)


# =========================
# 1) LLM 백엔드 로더
# =========================
def _get_gemini_loader():
    """
    Google Gemini API용 래퍼를 생성해 반환한다.
    - 기본 모델: config['GEMINI_MODEL'] (없으면 'gemini-1.5-flash')
    - system 지침으로 한국어 고정
    - 429(ResourceExhausted) 1회 재시도(권장 대기시간 활용)
    반환: 객체( .chat(messages: list[dict], **gen_kwargs) -> str )
    """
    import time
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted, PermissionDenied, FailedPrecondition

    api_key = os.getenv("GOOGLE_API_KEY") or current_app.config.get("GEMINI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY 환경변수(또는 GEMINI_API_KEY 설정)가 없습니다.")
    genai.configure(api_key=api_key)

    model_name = current_app.config.get("GEMINI_MODEL", "gemini-1.5-flash")
    SYSTEM_KO = (
        "당신은 한국어로만 간결하게 답변하는 비서입니다. "
        "역할/템플릿/토큰을 출력하지 말고, 질문에만 직접적으로 답하세요."
    )

    def _to_gemini_messages(messages: List[Dict]) -> List[Dict]:
        turns = []
        for m in messages:
            r = (m.get("role") or "user").lower()
            c = (m.get("content") or "").strip()
            if not c:
                continue
            if r == "assistant":
                turns.append({"role": "model", "parts": [c]})
            elif r == "system":
                # system은 모델 생성 시 system_instruction으로 전달
                continue
            else:
                turns.append({"role": "user", "parts": [c]})
        return turns

    class GeminiChat:
        def __init__(self, name: str):
            self.model_name = name
            self._model = genai.GenerativeModel(self.model_name, system_instruction=SYSTEM_KO)

        def _generate(self, contents, **gen_kwargs) -> str:
            gconf = {
                "temperature": float(gen_kwargs.get("temperature", current_app.config.get("GENAI_TEMPERATURE", 0.2))),
                "top_p": float(gen_kwargs.get("top_p", current_app.config.get("GENAI_TOP_P", 0.85))),
                "max_output_tokens": int(gen_kwargs.get("max_new_tokens", current_app.config.get("GENAI_MAX_NEW_TOKENS", 200))),
            }
            try:
                resp = self._model.generate_content(contents, generation_config=gconf)
                return (resp.text or "").strip()
            except ResourceExhausted as e:
                delay = getattr(getattr(e, "retry_delay", None), "seconds", None)
                delay = 3 if delay is None else min(int(delay), 5)
                time.sleep(max(1, delay))
                resp = self._model.generate_content(contents, generation_config=gconf)
                return (resp.text or "").strip()

        def chat(self, messages: List[Dict], **gen_kwargs) -> str:
            try:
                contents = _to_gemini_messages(messages)
                out = self._generate(contents, **gen_kwargs)
                return out or "(응답이 비어 있습니다)"
            except PermissionDenied as e:
                return f"(Gemini 호출 오류) PermissionDenied: {e}"
            except FailedPrecondition as e:
                return f"(Gemini 호출 오류) FailedPrecondition: {e}"
            except ResourceExhausted:
                return "(Gemini 호출 오류) ResourceExhausted: 쿼터를 초과했습니다. 잠시 후 다시 시도해 주세요."
            except Exception as e:
                return f"(Gemini 호출 오류) {type(e).__name__}: {e}"

        def __call__(self, prompt_or_messages, **gen_kwargs) -> str:
            if isinstance(prompt_or_messages, list):
                return self.chat(prompt_or_messages, **gen_kwargs)
            return self.chat(
                [{"role": "user", "content": str(prompt_or_messages)}],
                **gen_kwargs
            )

    return GeminiChat(model_name)


def _get_local_qwen_loader():
    """
    로컬 Qwen2.5-3B-Instruct 파이프라인 래퍼(백업/비상용).
    - HuggingFace transformers pipeline 사용(텍스트 생성)
    - ChatML 수동 구성 사용
    반환: LocalChat( .chat(messages) / __call__ )
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    try:
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))
        torch.set_num_interop_threads(int(os.getenv("TORCH_NUM_INTEROP_THREADS", "1")))
    except Exception:
        pass

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
        do_sample=False,
        max_new_tokens=int(current_app.config.get("GENAI_MAX_NEW_TOKENS", 200)),
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        eos_token_id=qwen_eos_id,
        return_full_text=False,
        num_return_sequences=1,
    )

    def _build_chatml_from_messages(messages: List[Dict]) -> str:
        parts = []
        sys = next((m["content"] for m in messages if m.get("role") == "system"), "")
        if sys:
            parts.append(f"<|im_start|>system\n{sys}<|im_end|>")
        for m in messages:
            r, c = m.get("role"), (m.get("content") or "").strip()
            if r == "system" or not c:
                continue
            parts.append(f"<|im_start|>{r}\n{c}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    class LocalChat:
        def __init__(self, pipe, tok, model_id, eos_id):
            self.pipe = pipe
            self.tokenizer = tok
            self.model_id = model_id
            self._eos_id = eos_id
            self.provider = "local"

        def chat(self, messages: List[Dict], **gen_kwargs) -> str:
            prompt = _build_chatml_from_messages(messages)
            out = self.pipe(
                prompt,
                eos_token_id=self._eos_id,
                max_new_tokens=gen_kwargs.get("max_new_tokens", None),
            )
            txt = out[0]["generated_text"] if isinstance(out, list) else str(out)
            return (txt or "").strip()

        def __call__(self, prompt_or_messages, **gen_kwargs) -> str:
            if isinstance(prompt_or_messages, list):
                return self.chat(prompt_or_messages, **gen_kwargs)
            return self.chat(
                [
                    {"role": "system", "content": "당신은 한국어로만 간결하게 답변하는 비서입니다."},
                    {"role": "user", "content": str(prompt_or_messages)},
                ],
                **gen_kwargs,
            )

    return LocalChat(gen_pipe, tok, model_id, qwen_eos_id)


from flask import current_app

def get_llm():
    """앱별로 LLM 인스턴스를 보관/재사용한다."""
    store = current_app.extensions["llm_store"]
    # 캐시 키는 백엔드/모델/주요 파라미터로 구성 (앱마다 current_app이 다름)
    key = "|".join([
        current_app.config.get("GENAI_PROVIDER", "gemini"),
        current_app.config.get("GEMINI_MODEL", "gemini-1.5-flash"),
        current_app.config.get("GENAI_MODEL", ""),
    ])

    if key not in store:
        provider = current_app.config.get("GENAI_PROVIDER", "gemini").lower()
        if provider == "gemini":
            store[key] = _get_gemini_loader()         # Gemini 어댑터
        elif provider == "local":
            store[key] = _get_local_qwen_loader()     # 로컬 Qwen
        else:
            store[key] = _get_gemini_loader()         # 기본값
    return store[key]


# =========================
# 2) 입력 정제/프롬프트 빌더
# =========================
def _sanitize_history_content(s: str) -> str:
    """
    히스토리(특히 과거 assistant 출력)에서 템플릿/역할 토큰/라벨/에코 라인을 제거한다.
    입력: s(str)
    출력: 정리된 문자열(str)
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
    messages(list[dict])를 Qwen ChatML 문자열로 구성한다.
    system → 턴들 → assistant 시작 순서로 이어붙인다.
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


def _build_messages_from_history(history: Optional[List[Dict]]) -> List[Dict]:
    """
    히스토리를 정규화하여 messages(list[dict])를 만든다.
    - 맨 앞에 system 지침(한국어 고정/토큰 출력 금지)을 둔다.
    - 마지막 user 발화 끝에 '한국어만 사용' 지침을 일시적으로 추가한다(저장은 안 함).
    - 마지막 user 발화에 RAG 검색 결과(상위 k개)를 [참고문서] 블록으로 덧붙인다(있을 때만).
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

    if len(messages) >= 2 and messages[-1].get("role") == "user":
        marker = "답변은 반드시 한국어로만 작성하세요"
        if marker not in messages[-1]["content"]:
            messages[-1]["content"] = (
                messages[-1]["content"].rstrip()
                + "\n\n(지침: 답변은 반드시 한국어로만 작성하세요. 영어를 사용하지 마세요.)"
            )
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
# 3) 생성/후처리
# =========================
def _extract_assistant_body_for_qwen(raw: str) -> str:
    """
    return_full_text=True 로 돌아오는 경우나 모델이 ChatML을 섞어 낸 경우,
    어시스턴트 본문만 추출한다.
    """
    if not raw:
        return ""
    m = _RX_QWEN_ASSIST_1.search(raw) or _RX_QWEN_ASSIST_2.search(raw)
    return (m.group(1).strip() if m else raw.strip())


def _light_clean(text: str) -> str:
    """
    역할 라벨/라벨 접두사/여분 개행을 가볍게 정리한다.
    """
    if not text:
        return ""
    t = _HEAD_LABEL_RE.sub("", text)
    t = _LABEL_PREFIX_RE.sub("", t)
    t = _TRAIL_WS_NL_RE.sub("\n", t)
    t = _MULTI_NL_RE.sub("\n\n", t)
    return t.strip()


def _looks_like_english_or_echo(s: str) -> bool:
    """
    생성 결과가 영어 위주이거나 시스템 문구/에코가 섞인 것처럼 보이면 True.
    """
    if not s:
        return True
    if re.search(r"(?i)\b(reply in korean|you are a helpful assistant|as a helpful assistant)\b", s):
        return True
    letters = re.findall(r"[A-Za-z]", s)
    non_space = re.findall(r"\S", s)
    if non_space and (len(letters) / max(1, len(non_space))) > 0.35:
        return True
    return False


def _generate_with_messages(llm, messages: List[Dict], **gen_kwargs) -> str:
    """
    1차 생성(보수적) → 본문 추출/클린업 → 영어/에코 감지 시 2차 엄격 생성으로 재시도한다.
    입력: llm, messages, gen_kwargs
    출력: 최종 한국어 답변 문자열
    """
    raw = llm.chat(messages, **gen_kwargs) if hasattr(llm, "chat") else llm(messages, **gen_kwargs)
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "generated_text" in raw[0]:
        raw = raw[0]["generated_text"] or ""
    body = _light_clean(_extract_assistant_body_for_qwen(str(raw)))

    if not _looks_like_english_or_echo(body):
        return body or "(응답이 비어 있습니다)"

    # 2차 엄격 재시도(최근 user만)
    last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    strict_msgs = [
        {"role": "system", "content": "지금부터 한국어만 사용합니다. 답변은 1–3문장으로 간결하게 작성하세요."},
        {"role": "user",   "content": last_user.strip() or "간단히 인사해 주세요."},
    ]
    raw2 = llm.chat(strict_msgs, do_sample=False, temperature=0.0, top_p=1.0,
                    max_new_tokens=gen_kwargs.get("max_new_tokens", None))
    if isinstance(raw2, list) and raw2 and isinstance(raw2[0], dict) and "generated_text" in raw2[0]:
        raw2 = raw2[0]["generated_text"] or ""
    return _light_clean(_extract_assistant_body_for_qwen(str(raw2))) or "(응답이 비어 있습니다)"


# =========================
# 4) RAG 검색
# =========================
def _retrieve_knowledge(query: str) -> List[Dict]:
    """
    RAG가 켜져 있으면 top-k 문단을 검색해 [{text, source, score}]를 반환한다.
    꺼져 있거나 인덱스가 없으면 빈 리스트를 반환한다.
    """
    if not current_app.config.get("RAG_ENABLED", False):
        return []
    try:
        retr = get_retriever()
        top_k = int(current_app.config.get("RAG_TOP_K", 3))
        hits = retr.search(query, top_k=top_k)
        return [{"text": h.text, "source": h.source, "score": h.score} for h in hits]
    except Exception:
        return []


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
# 6) 라우트 (챗봇 / 인지 스크리닝 대화)
# =========================
@bp.route("/chat", methods=["GET", "POST"])
def chat():
    """
    채팅 페이지 렌더 및 메시지 제출 처리.
    - POST: 사용자 발화 저장 → 히스토리 구성 → LLM 호출 → 응답 저장 → 리다이렉트
    - GET : 전체 히스토리 렌더
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
                    old_ids = [m.id for m in q.limit(count - cap).all()]
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


@bp.route("/screen", methods=["GET"])
def screen():
    """
    인지 스크리닝 대화(일상 대화 기반) 페이지 렌더.
    - 템플릿: templates/genai/screen.html
    - Whisper 전사/점수는 사용하지 않음(대화형으로 판단).
    """
    return render_template("genai/screen.html")
