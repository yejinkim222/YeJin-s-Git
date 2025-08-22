# pybo/python/screening.py
from __future__ import annotations
import re, datetime as dt
from typing import Dict

_TARGET_WORDS = {"사과", "열쇠", "기차"}
_SERIAL7_SEQ  = [100, 93, 86, 79, 72, 65]  # 5번 감산 → 확인은 93~65까지

def _norm(s: str) -> str:
    return (s or "").strip()

def _contains_weekday(ans: str) -> bool:
    # 오늘 요일과 매칭
    kweek = ["월","화","수","목","금","토","일"]
    today = dt.date.today().weekday()  # 0=월
    return (kweek[today] in ans) or (kweek[today]+"요일" in ans)

def _contains_season(ans: str) -> bool:
    m = dt.date.today().month
    season = "겨울" if m in (12,1,2) else "봄" if m in (3,4,5) else "여름" if m in (6,7,8) else "가을"
    return season in ans

def _score_date(ans: str) -> int:
    # 대략적: '2025-08-21' 또는 '8월 21일' 등 포함하면 1점
    if re.search(r"\b20\d{2}[-./]?\s*\d{1,2}[-./]?\s*\d{1,2}\b", ans):
        return 1
    if re.search(r"\b\d{1,2}\s*월\s*\d{1,2}\s*일", ans):
        return 1
    return 0

def _score_weekday(ans: str) -> int:
    return 1 if _contains_weekday(ans) else 0

def _score_season(ans: str) -> int:
    return 1 if _contains_season(ans) else 0

def _score_words(ans: str) -> int:
    # 3단어 중 맞춘 개수(순서 무관)
    hits = 0
    for w in _TARGET_WORDS:
        if w in ans:
            hits += 1
    return hits  # 0~3

def _score_serial7(ans: str) -> int:
    # 숫자만 추출해서 시퀀스 접두 길이 평가, 0~3점
    nums = list(map(int, re.findall(r"\d+", ans)))
    # 기대: 100,93,86,79,72,65 → 93~65 5개 중 맞춘 개수 기반 점수
    expected = _SERIAL7_SEQ[1:]  # [93,86,79,72,65]
    k = 0
    for i, v in enumerate(expected):
        if i < len(nums) and nums[i] == v:
            k += 1
        else:
            break
    if k >= 4: return 3
    if k >= 3: return 2
    if k >= 2: return 1
    return 0

def _score_repeat(ans: str) -> int:
    # “봄에는 꽃이 핀다”와의 러프 매칭
    return 1 if ("봄" in ans and "꽃" in ans and ("핀" in ans or "핍니다" in ans)) else 0

def score_screening(answers: Dict[str, str]) -> Dict[str, object]:
    """
    answers: {
      "date": "...", "weekday": "...", "season": "...",
      "encode_words": "...", "serial7": "...", "recall_words": "...", "repeat": "..."
    }
    합계 10점 만점:
      - 날짜/요일/계절 각 1점(3)
      - 즉시 회상(3단어) 0~3점(3)
      - 연산(serial7) 0~3점(3)
      - 문장 따라말하기 1점(1)
    """
    date  = _score_date(_norm(answers.get("date","")))
    week  = _score_weekday(_norm(answers.get("weekday","")))
    seas  = _score_season(_norm(answers.get("season","")))
    enc   = _score_words(_norm(answers.get("encode_words","")))
    ser7  = _score_serial7(_norm(answers.get("serial7","")))
    rep   = _score_repeat(_norm(answers.get("repeat","")))
    total = date + week + seas + enc + ser7 + rep
    # 판정(경험적 기준) — 필요시 조정
    if total >= 8:
        label = "정상 범위"
    elif total >= 6:
        label = "관찰 필요"
    else:
        label = "의심 — 전문가 상담 권장"

    detail = {
        "날짜": date, "요일": week, "계절": seas,
        "즉시 회상(3단어)": enc, "연산(100-7)": ser7, "문장 따라말하기": rep
    }
    return {"score": total, "label": label, "detail": detail}
