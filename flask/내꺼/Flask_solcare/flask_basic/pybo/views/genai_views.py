# pybo/views/genai_views.py
from __future__ import annotations
import secrets, re
from functools import lru_cache

from flask import Blueprint, render_template, request, redirect, url_for, g, session

from .. import db
from pybo.models import Conversation, Message
from pybo.rag.retriever import get_retriever


from typing import List, Dict, Optional
import os
from flask import current_app

from pybo.python.screening import score_screening


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


# -------------------------------------------------------------------
# Gemini 로더
# -------------------------------------------------------------------
def _get_gemini_loader():
    """
    Google Gemini API용 래퍼를 생성해 반환한다.
    - 기본 모델: gemini-1.5-flash
    - 429(ResourceExhausted) 발생 시: 권장 대기시간(최대 5초) 1회 재시도
    - 여전히 실패하고 현재 모델이 pro면 flash로 다운시프트 시도
    반환: 객체( .chat(messages: list[dict], **gen_kwargs) -> str )
    """
    import os, time
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted, PermissionDenied, FailedPrecondition

    api_key = os.getenv("GOOGLE_API_KEY") or current_app.config.get("GEMINI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY 환경변수(또는 GEMINI_API_KEY 설정)가 없습니다.")
    genai.configure(api_key=api_key)

    model_name = current_app.config.get("GEMINI_MODEL", "gemini-1.5-flash")

    # 한국어 강제 system 지침
    SYSTEM_KO = (
        "당신은 한국어로만 간결하게 답변하는 비서입니다. "
        "역할/템플릿/토큰을 출력하지 말고, 질문에만 직접적으로 답하세요."
    )

    def _to_gemini_messages(messages):
        """
        우리의 messages(list[{'role','content'}]) -> Gemini contents 포맷으로 변환
        - Gemini는 'user' | 'model' 두 role만 사용
        - system은 모델 생성 시 system_instruction으로 이미 전달했으므로 스킵
        """
        conv = []
        for m in messages:
            role = (m.get("role") or "user").lower()
            text = (m.get("content") or "").strip()
            if not text:
                continue
            if role == "assistant":
                conv.append({"role": "model", "parts": [text]})
            elif role == "system":
                # system은 이미 model 생성 시 전달 → 스킵
                continue
            else:
                conv.append({"role": "user", "parts": [text]})
        return conv

    class GeminiChat:
        def __init__(self, model_name):
            self.model_name = model_name
            self._model = genai.GenerativeModel(self.model_name, system_instruction=SYSTEM_KO)

        def _generate(self, contents, **gen_kwargs):
            # Gemini generation_config 매핑
            gconf = {
                "temperature": float(gen_kwargs.get("temperature", current_app.config.get("GENAI_TEMPERATURE", 0.2))),
                "top_p": float(gen_kwargs.get("top_p", current_app.config.get("GENAI_TOP_P", 0.85))),
                # top_k는 선택적
                "max_output_tokens": int(gen_kwargs.get("max_new_tokens", current_app.config.get("GENAI_MAX_NEW_TOKENS", 200))),
            }

            # 1차 시도
            try:
                resp = self._model.generate_content(contents, generation_config=gconf)
                return (resp.text or "").strip()
            except ResourceExhausted as e:
                # 권장 대기시간 파싱(있다면)
                delay = getattr(getattr(e, "retry_delay", None), "seconds", None)
                if delay is None:
                    delay = 3
                delay = min(int(delay), 5)
                time.sleep(max(1, delay))
                # 2차 시도
                resp = self._model.generate_content(contents, generation_config=gconf)
                return (resp.text or "").strip()

        def chat(self, messages, **gen_kwargs):
            contents = _to_gemini_messages(messages)
            try:
                out = self._generate(contents, **gen_kwargs)
                if out:
                    return out
                return "(응답이 비어 있습니다)"
            except ResourceExhausted:
                # 여전히 터질 때, pro면 flash로 다운시프트(자동 완화)
                if self.model_name.startswith("gemini-1.5-pro"):
                    try:
                        fallback = "gemini-1.5-flash"
                        self.model_name = fallback
                        self._model = genai.GenerativeModel(self.model_name, system_instruction=SYSTEM_KO)
                        out = self._generate(contents, **gen_kwargs)
                        return out or "(응답이 비어 있습니다)"
                    except Exception as e2:
                        return f"(Gemini 호출 오류) {type(e2).__name__}: {e2}"
                return "(Gemini 호출 오류) ResourceExhausted: 쿼터를 초과했습니다. 잠시 후 다시 시도해 주세요."
            except PermissionDenied as e:
                return f"(Gemini 호출 오류) PermissionDenied: {e}"
            except FailedPrecondition as e:
                # API 비활성 등
                return f"(Gemini 호출 오류) FailedPrecondition: {e}"
            except Exception as e:
                return f"(Gemini 호출 오류) {type(e).__name__}: {e}"

        # 우리 프레임워크와의 호환을 위해 __call__도 지원
        def __call__(self, prompt_or_messages, **gen_kwargs):
            if isinstance(prompt_or_messages, list):
                return self.chat(prompt_or_messages, **gen_kwargs)
            # 문자열 프롬프트 → 단일 user turn으로 감싸기
            return self.chat(
                [{"role": "user", "content": str(prompt_or_messages)}],
                **gen_kwargs
            )

    return GeminiChat(model_name)


# -------------------------------------------------------------------
# (선택) 로컬 Qwen 로더 - 백업 용도
# -------------------------------------------------------------------
def _get_local_qwen_loader():
    """
    로컬 Qwen 파이프라인 래퍼(백업/비상용).
    - HuggingFace transformers pipeline 사용
    - 한국어 시스템 지침 반영
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
        # apply_chat_template 미사용: ChatML 수동 구성
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


# -------------------------------------------------------------------
# 선택자: get_llm
# -------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_llm():
    """
    GENAI_PROVIDER 설정에 따라 백엔드를 선택해 래퍼를 반환한다.
    - 'gemini'  -> Google GeminiChat
    - 'local'   -> 로컬 Qwen LocalChat
    """
    provider = (current_app.config.get("GENAI_PROVIDER") or "gemini").lower()
    if provider == "gemini":
        return _get_gemini_loader()
    if provider == "local":
        return _get_local_qwen_loader()
    # 알 수 없는 값이면 Gemini 우선
    return _get_gemini_loader()


# ------------------------------
# EXAONE (OpenAI 호환 REST 가정)
# ------------------------------
def _get_exaone_loader():
    """
    역할:
      - EXAONE 제공처의 OpenAI 호환 REST 엔드포인트를 호출하는 래퍼를 만든다.
      - 실제 엔드포인트/모델 id는 공급자 문서대로 config에서 설정.
    필요 env/config:
      - EXAONE_API_KEY, EXAONE_API_BASE, EXAONE_MODEL
    반환:
      - ExaoneChat 래퍼( .chat(messages), __call__(prompt) )
    """
    import requests

    api_key  = current_app.config.get("EXAONE_API_KEY")  or ""
    base_url = current_app.config.get("EXAONE_API_BASE") or "https://api.exaone.ai/v1"
    model_id = current_app.config.get("EXAONE_MODEL")    or "exaone-3.0-instruct"
    if not api_key:
        raise RuntimeError("EXAONE_API_KEY 가 설정되지 않았습니다.")

    chat_url = base_url.rstrip("/") + "/chat/completions"  # OpenAI 호환 경로 가정

    class ExaoneChat:
        def __init__(self, model_id, url, key):
            self.model_id = model_id
            self.url = url
            self.key = key

        def _headers(self):
            return {
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            }

        def chat(self, messages, **gen_kwargs) -> str:
            # temperature, top_p, max_tokens 매핑
            payload = {
                "model": self.model_id,
                "messages": messages,  # system/user/assistant 구조 그대로 사용
            }
            if gen_kwargs.get("temperature") is not None:
                payload["temperature"] = float(gen_kwargs["temperature"])
            if gen_kwargs.get("top_p") is not None:
                payload["top_p"] = float(gen_kwargs["top_p"])
            if gen_kwargs.get("max_new_tokens") is not None:
                payload["max_tokens"] = int(gen_kwargs["max_new_tokens"])

            r = requests.post(self.url, headers=self._headers(), json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()

            # OpenAI 호환: choices[0].message.content
            try:
                text = data["choices"][0]["message"]["content"]
            except Exception:
                # 일부 구현체는 'output_text' 또는 'choices[0].text' 등 변형일 수 있음
                text = data.get("output_text") or data.get("choices", [{}])[0].get("text", "")
            return (text or "").strip()

        def __call__(self, prompt_or_messages, **gen_kwargs) -> str:
            if isinstance(prompt_or_messages, list):
                return self.chat(prompt_or_messages, **gen_kwargs)
            return self.chat(
                [
                    {"role": "system", "content": "당신은 한국어로만 간결하게 답변하는 비서입니다."},
                    {"role": "user",   "content": str(prompt_or_messages)},
                ],
                **gen_kwargs,
            )

    return ExaoneChat(model_id, chat_url, api_key)


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


# === Whisper 로더 & 전사 ===
from functools import lru_cache

@lru_cache(maxsize=1)
def _get_asr_model():
    """
    faster-whisper 모델 단일 인스턴스 로더.
    """
    from faster_whisper import WhisperModel
    model_size  = current_app.config.get("ASR_MODEL", "small")
    # CPU 기준 권장 설정
    return WhisperModel(model_size, device="cpu", compute_type="int8")

def _asr_transcribe_ko(audio_path: str) -> str:
    """
    한국어 우선 전사. 실패 시 빈 문자열.
    입력: 로컬 오디오 파일 경로
    출력: 공백 정리된 문자열
    """
    model = _get_asr_model()
    beam  = int(current_app.config.get("ASR_BEAM", 5))
    segments, info = model.transcribe(audio_path, language="ko", task="transcribe", beam_size=beam)
    text = "".join(seg.text for seg in segments).strip()
    # 여분 공백/개행 정리
    import re
    text = re.sub(r"\s+", " ", text)
    return text


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


# === (A) 인지 스크리닝(음성) 페이지 ===
@bp.route("/screen", methods=["GET"])
def screen():
    """
    인지 스크리닝(음성 기반) 페이지 렌더.
    """
    return render_template("genai/screen_voice.html", is_guest=(getattr(g, "user", None) is None))


# === (B) ASR: 브라우저에서 업로드한 오디오를 Whisper로 받아 적기 ===
@bp.route("/asr", methods=["POST"])
def asr_transcribe():
    """
    브라우저 MediaRecorder로 녹음한 파일(webm/opus 등)을 받아
    faster-whisper로 한국어 전사 후 {text: "..."} 반환.
    """
    from flask import jsonify
    import tempfile, subprocess, uuid, os

    f = request.files.get("audio")
    if not f:
        return jsonify(error="no file"), 400

    # 업로드 임시 저장
    tmp_dir = os.path.join(current_app.root_path, "data", "asr_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    in_path  = os.path.join(tmp_dir, f"rec_{uuid.uuid4().hex}.webm")
    out_wav  = os.path.join(tmp_dir, f"rec_{uuid.uuid4().hex}.wav")
    f.save(in_path)

    # ffmpeg로 wav(16kHz/mono) 변환
    try:
        cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", out_wav]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio_path = out_wav
    except Exception:
        # ffmpeg 없으면 원본 파일로 시도(환경에 따라 실패할 수 있음)
        audio_path = in_path

    try:
        text = _asr_transcribe_ko(audio_path)
        return jsonify(text=text)
    finally:
        # 정리
        for p in (in_path, out_wav):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# === (C) 점수 계산 ===
@bp.route("/screen/score", methods=["POST"])
def screen_score():
    """
    클라이언트가 누적 답변(텍스트)을 보내면 점수/판정/피드백 반환.
    바디: {"answers": {"date": "...", "weekday": "...", ...}}
    """
    from flask import jsonify
    data = request.get_json(silent=True) or {}
    answers = data.get("answers") or {}
    result = score_screening(answers)
    return jsonify(result)


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
