# pybo/agent_tools.py
from __future__ import annotations
import json
from types import SimpleNamespace
from flask import current_app, g
from pybo import db

_TOOL_FUNCS = {}
_TOOL_SCHEMAS = []

def tool(name):
    def deco(fn):
        _TOOL_FUNCS[name] = fn
        return fn
    return deco

def add_tool_schema(name: str, description: str, parameters: dict):
    _TOOL_SCHEMAS.append({
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters}
    })

def registry():
    return SimpleNamespace(funcs=_TOOL_FUNCS, schemas=_TOOL_SCHEMAS)

# ─────────────────────────────────────────────────────────────────────────────
# 0) calc / rag_search / web_get 등 기존 등록은 여기로 모읍니다 (생략)
# ─────────────────────────────────────────────────────────────────────────────

# 1) 음성 자가진단: 최신 결과 가져오기
add_tool_schema(
    "audio_fetch_latest_result",
    "로그인 사용자의 최신 '음성 자가진단' 결과 요약을 반환",
    {"type":"object","properties":{}, "additionalProperties": False}
)

@tool("audio_fetch_latest_result")
def _tool_audio_fetch_latest_result() -> str:
    if not getattr(g, "user", None):
        return "[error] 로그인 사용자가 아닙니다."
    from models import ScreeningResult
    row = (ScreeningResult.query
           .filter_by(user_username=g.user.username)
           .order_by(ScreeningResult.created_at.desc())
           .first())
    if not row:
        return "최근 저장된 검사 결과가 없습니다. 'audio_start_session'으로 새 검사를 시작할 수 있어요."
    # 프론트 정책: 짧게 요약 + 키워드/출처 힌트
    summary = f"총점 {row.total_score}/{row.max_score} → {row.result_summary}. 권고: {row.advice or '—'}"
    return summary  # 템플릿이 요약형으로 렌더

# 2) 자가진단 세션 시작(링크 안내)
add_tool_schema(
    "audio_start_session",
    "새로운 음성 자가진단 세션을 시작할 링크를 반환",
    {"type":"object","properties":{}, "additionalProperties": False}
)

@tool("audio_start_session")
def _tool_audio_start_session() -> str:
    # 실제 화면은 기존 /genai/audio
    return "새 검사 시작: /genai/audio  (창을 열어 진행해 주세요)"
