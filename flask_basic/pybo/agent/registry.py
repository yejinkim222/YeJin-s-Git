# pybo/agent/registry.py
from __future__ import annotations
import ast, operator, time, re
from typing import Any, Dict, Callable, Tuple

from flask import current_app

# RAG 툴 (이미 구현된 함수 재사용)
from pybo.rag.engine import (
    rag_search_snippets,
    rag_upsert_texts,
)

# ----------------------------------
# 간단/안전 수식 계산기
# ----------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def _eval_node(node):
    if isinstance(node, ast.Num):               # py<=3.7
        return node.n
    if isinstance(node, ast.Constant):          # py>=3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("constants not allowed")
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")

def calc(expr: str) -> str:
    try:
        tree = ast.parse((expr or "").strip(), mode="eval")
        # 금지 노드 체크
        for n in ast.walk(tree):
            if isinstance(n, (ast.Call, ast.Attribute, ast.Name, ast.Subscript, ast.List, ast.Dict, ast.Tuple)):
                raise ValueError("only numeric expressions are allowed")
        val = _eval_node(tree.body)
        return f"계산 결과: {val}"
    except Exception:
        return "수식을 이해하지 못했어요. 예: 2*(3+4)/5"

# ----------------------------------
# 외부 웹검색(기본 비활성, 링크 힌트만)
# ----------------------------------
def web_search_links(query: str) -> str:
    enabled = str(current_app.config.get("AGENT_WEB_SEARCH", "false")).lower() == "true"
    if not enabled:
        return "외부 웹검색은 현재 비활성화되어 있어요."
    api_key = current_app.config.get("BING_API_KEY")
    if not api_key:
        return "외부 웹검색 키가 설정되지 않았습니다."

    import requests
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    params = {"q": query, "mkt": "ko-KR", "count": 5, "safeSearch": "Strict", "textDecorations": False}
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    try:
        r = requests.get(endpoint, params=params, headers=headers, timeout=6)
        r.raise_for_status()
        web = (r.json().get("webPages") or {}).get("value") or []
        if not web:
            return "웹검색 결과가 충분하지 않습니다."
        hints = []
        for it in web[:3]:
            title = it.get("name", "").strip()
            desc  = it.get("snippet", "").strip()
            url   = it.get("url", "").strip()
            hints.append(f"- {title} — {desc}\n  [링크] {url}")
        return "다음 링크에서 자세히 볼 수 있어요:\n" + "\n".join(hints)
    except Exception:
        return "웹검색 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요."

# ----------------------------------
# 로컬 지식 업서트
# ----------------------------------
def local_upsert_text(title: str, text: str, meta: Dict[str, Any] | None = None) -> str:
    items = [{"title": title or "internal", "text": text or "", "meta": meta or {}}]
    res = rag_upsert_texts(items)
    n = int(res.get("added", 0))
    return f"내부 지식에 {n}개 청크를 추가했습니다."
