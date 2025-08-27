# pybo/agent/runner.py
from __future__ import annotations
import time, re
from typing import List, Dict, Tuple, Callable, Optional
from flask import current_app

from .registry import calc, web_search_links, local_upsert_text
from pybo.rag.engine import rag_search_snippets

__all__ = ["agent_run"]

# -----------------------------
# 쿨다운 캐시 (1분 TTL)
# -----------------------------
def _cache() -> dict:
    return current_app.extensions.setdefault("agent_cache", {})

def _norm_arg(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _get_cached(tool: str, arg: str, ttl_sec: int = 60) -> Optional[str]:
    key = (tool, _norm_arg(arg))
    obj = _cache().get(key)
    if not obj:
        return None
    if time.time() - obj["ts"] > ttl_sec:
        _cache().pop(key, None)
        return None
    return obj["val"]

def _set_cached(tool: str, arg: str, val: str) -> None:
    key = (tool, _norm_arg(arg))
    _cache()[key] = {"ts": time.time(), "val": val}

# -----------------------------
# 1) 질의 리라이트
# -----------------------------
_REWRITE_PROMPT = (
    "다음 사용자의 한국어 질문을 의미를 바꾸지 않고 1문장으로 명료하게 바꿔라. "
    "대명사/지시어는 구체 명사로 치환하라. 결과만 출력:\n\n질문: {q}"
)

def _rewrite_query(user_text: str, history: List[Dict[str, str]], get_llm: Callable[[], object]) -> str:
    """LLM로 질문을 명확하게 1문장으로 재작성. 실패 시 원문 반환."""
    try:
        llm = get_llm()
        chat = getattr(llm, "chat", None)
        if not callable(chat):
            raise AttributeError("LLM adapter has no callable 'chat'")

        prompt_text = _REWRITE_PROMPT.format(q=user_text)
        out = chat([{"role": "user", "content": prompt_text}], max_new_tokens=80)
        if out is None:
            out = ""
        if not isinstance(out, str):
            out = str(out)

        out = out.strip()
        if 5 <= len(out) <= 200:
            return out
    except Exception as e:
        try:
            current_app.logger.warning("agent: rewrite failed: %s", e)
        except Exception:
            pass
    return user_text

# -----------------------------
# 2) 툴 선택 휴리스틱
# -----------------------------
_MATH_RE = re.compile(r"^\s*[-+*/().0-9\s%^]+\s*$")

def _choose_tool(q: str) -> Tuple[str, dict]:
    """입력 질의에 따라 사용할 툴과 인자 선택."""
    # 특수 포맷: "업서트: 제목 | 본문"
    if q.startswith("업서트:"):
        _, rest = q.split("업서트:", 1)
        parts = [p.strip() for p in rest.split("|", 1)]
        title = parts[0] if parts else "internal"
        text  = parts[1] if len(parts) > 1 else ""
        return "local_upsert", {"title": title, "text": text}

    # 수식 계산
    if _MATH_RE.match(q):
        return "calc", {"expr": q}

    # 기본: RAG
    return "rag", {"query": q, "k": int(current_app.config.get("RAG_TOP_K", 3))}

# -----------------------------
# 3) 실행기
# -----------------------------
def _run_tool(name: str, args: dict) -> str:
    """선택된 툴 실행 + 1분 쿨다운 캐시."""
    key = args.get("query") or args.get("expr") or args.get("title", "")
    cached = _get_cached(name, str(key), ttl_sec=60)
    if cached:
        return cached + "\n[메모] 직전과 동일한 요청이라 같은 결과일 수 있어요."

    if name == "calc":
        res = calc(args.get("expr", ""))

    elif name == "local_upsert":
        res = local_upsert_text(args.get("title", "internal"), args.get("text", ""), {})

    elif name == "rag":
        res = rag_search_snippets(args.get("query", ""), top_k=int(args.get("k", 3)))
        # RAG 실패/빈 결과 → 외부 웹검색(함수 내에서 비활성화 메시지 처리)
        if "유의미한 결과를 찾지 못했습니다" in res or res.startswith("[error]"):
            ws = web_search_links(args.get("query", ""))
            if not ws.startswith("외부 웹검색은 현재 비활성화"):
                res = res + "\n\n" + ws

    elif name == "web":
        res = web_search_links(args.get("query", ""))

    else:
        res = "알 수 없는 도구 요청입니다."

    _set_cached(name, str(key), res)
    return res

# -----------------------------
# 메인: agent_run
# -----------------------------
def agent_run(user_text: str, history: List[Dict[str, str]], get_llm: Callable[[], object]) -> str:
    """
    단일 턴 에이전트 실행:
      1) 질의 리라이트 → 2) 툴 선택 → 3) 툴 실행(필요 시 웹 보강) → 결과 반환
    """
    q = _rewrite_query(user_text, history, get_llm=get_llm)
    tool, args = _choose_tool(q)

    start = time.time()
    res = _run_tool(tool, args)
    elapsed = int((time.time() - start) * 1000)
    try:
        current_app.logger.info(
            "AGENT %s in %dms args=%s",
            tool, elapsed, {k: args[k] for k in sorted(args) if k != "text"}
        )
    except Exception:
        pass

    return res  # genai_views에서 200자 후처리 적용
