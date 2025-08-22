# pybo/python/cogscreen.py
from __future__ import annotations
import math, re
from dataclasses import dataclass
from typing import List, Dict, Optional

# 간단 한국어 토큰화(공백 기준 + 한글/숫자만)
_TOK = re.compile(r"[가-힣0-9A-Za-z]+")

_FILLERS = {"음", "음...", "어", "어...", "그", "그러니까", "뭐지", "저기", "음m", "음…", "어…"}
_REP_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

@dataclass
class Turn:
    role: str          # "user" or "assistant"
    text: str
    start: float = 0.0 # sec (옵션)
    end: float = 0.0   # sec (옵션)

@dataclass
class Features:
    wpm: float
    silence_ratio: float
    filler_ratio: float
    ttr: float
    mattr: float
    mean_sent_len: float
    rep_trigram_ratio: float
    coherence: float  # 0~1
    idea_density: float

@dataclass
class Score:
    total: int           # 0~10
    risk_pct: int        # 0~100
    band: str            # "정상/경계/의심"
    reasons: List[str]
    features: Features

def _tokens(s: str) -> List[str]:
    return [t.lower() for t in _TOK.findall(s)]

def _sentences(s: str) -> List[str]:
    # 매우 단순 분할
    parts = re.split(r"[.!?？！…\n]+", s)
    return [p.strip() for p in parts if p.strip()]

def _ttr(tokens: List[str]) -> float:
    return 0.0 if not tokens else len(set(tokens)) / max(1, len(tokens))

def _mattr(tokens: List[str], window: int = 50) -> float:
    if not tokens: return 0.0
    if len(tokens) <= window: return _ttr(tokens)
    vals = []
    for i in range(0, len(tokens)-window+1):
        vals.append(_ttr(tokens[i:i+window]))
    return sum(vals) / len(vals)

def _trigram_repeat_ratio(tokens: List[str]) -> float:
    if len(tokens) < 6: return 0.0
    trigrams = [" ".join(tokens[i:i+3]) for i in range(len(tokens)-2)]
    total = len(trigrams)
    uniq  = len(set(trigrams))
    return 1.0 - (uniq/total)

def _filler_ratio(tokens: List[str]) -> float:
    if not tokens: return 0.0
    cnt = sum(1 for t in tokens if t in _FILLERS)
    return cnt / len(tokens)

def _idea_density(tokens: List[str]) -> float:
    # 대략: 길이 대비 '내용어' 비율(숫자/한글 영문 길이>=2 토큰)
    if not tokens: return 0.0
    content = [t for t in tokens if len(t) >= 2 and not t.isdigit()]
    return len(content) / len(tokens)

def _coherence(utts: List[str]) -> float:
    # 아주 가벼운 휴리스틱: 인접 문장 간 Jaccard 유사도(토큰 집합)
    if len(utts) < 2: return 1.0
    sims = []
    for a, b in zip(utts[:-1], utts[1:]):
        ta, tb = set(_tokens(a)), set(_tokens(b))
        if not ta or not tb:
            sims.append(0.0); continue
        inter = len(ta & tb); union = len(ta | tb)
        sims.append(inter/union)
    return sum(sims)/len(sims)

def extract_features(turns: List[Turn]) -> Features:
    # user 발화만 취합
    user_turns = [t for t in turns if t.role == "user" and t.text.strip()]
    all_text   = " ".join(t.text for t in user_turns)
    toks       = _tokens(all_text)

    # WPM (총 발화 길이/시간)
    dur = 0.0
    voiced = 0.0
    for t in user_turns:
        seg = max(0.0, t.end - t.start)
        dur += seg
        # 무음 추정이 없으므로 단순히 발화 전체를 유성 시간으로 가정
        voiced += seg
    wpm = (len(toks)/max(1.0, dur/60.0)) if dur > 0 else 0.0
    silence_ratio = 0.0 if dur == 0 else max(0.0, min(1.0, (dur-voiced)/dur))

    sents = []
    for t in user_turns: sents.extend(_sentences(t.text))

    feats = Features(
        wpm = wpm,
        silence_ratio = silence_ratio,
        filler_ratio = _filler_ratio(toks),
        ttr = _ttr(toks),
        mattr = _mattr(toks),
        mean_sent_len = 0.0 if not sents else sum(len(_tokens(s)) for s in sents)/len(sents),
        rep_trigram_ratio = _trigram_repeat_ratio(toks),
        coherence = _coherence(sents),
        idea_density = _idea_density(toks),
    )
    return feats

def score_features(f: Features) -> Score:
    """
    휴리스틱 맵핑(초안)
    - 정상 한국어 대화의 대략적 범위를 기준으로 0~10 점수화
    """
    reasons = []

    # 가이드라인(거칠게): 90~170 wpm
    s_wpm = 2 if 90 <= f.wpm <= 170 else (1 if 70 <= f.wpm <= 190 else 0)
    if s_wpm == 0: reasons.append(f"말 속도 비정상(≈{int(f.wpm)} wpm)")

    s_fill = 2 if f.filler_ratio <= 0.03 else (1 if f.filler_ratio <= 0.08 else 0)
    if s_fill == 0: reasons.append("머뭇거림/군더더기 비율 높음")

    s_ttr  = 2 if f.ttr >= 0.35 else (1 if f.ttr >= 0.25 else 0)
    if s_ttr == 0: reasons.append("어휘 다양도 낮음")

    s_mattr = 2 if f.mattr >= 0.32 else (1 if f.mattr >= 0.24 else 0)

    s_sent = 2 if 8 <= f.mean_sent_len <= 25 else (1 if 6 <= f.mean_sent_len <= 30 else 0)
    if s_sent == 0: reasons.append("문장 길이(정보량) 비정상")

    s_rep = 2 if f.rep_trigram_ratio <= 0.05 else (1 if f.rep_trigram_ratio <= 0.12 else 0)
    if s_rep == 0: reasons.append("반복/순환 발화 많음")

    s_coh = 2 if f.coherence >= 0.25 else (1 if f.coherence >= 0.15 else 0)
    if s_coh == 0: reasons.append("주제 일관성 낮음")

    s_idea = 2 if f.idea_density >= 0.55 else (1 if f.idea_density >= 0.45 else 0)
    if s_idea == 0: reasons.append("내용어 밀도 낮음")

    total = s_wpm + s_fill + s_ttr + s_mattr + s_sent + s_rep + s_coh + s_idea  # 0~16
    # 0~16 → 0~10으로 선형 스케일
    total10 = round(total * (10/16))

    # 위험도(낮을수록 좋음) 역변환
    risk = max(0, 100 - total10*10)

    if   total10 >= 8: band = "정상 범위"
    elif total10 >= 5: band = "경계(경과 관찰 권고)"
    else:              band = "의심(전문 평가 권고)"

    return Score(total10, risk, band, reasons, f)

def evaluate_conversation(turn_dicts: List[Dict]) -> Dict:
    """
    입력: [{"role":"user","text":"...","start":0,"end":3.1}, ...]
    출력: 점수/사유/특징 요약
    """
    turns = [Turn(**{**d}) for d in turn_dicts if d.get("text")]
    feats = extract_features(turns)
    sc = score_features(feats)
    return {
        "total": sc.total,
        "risk_pct": sc.risk_pct,
        "band": sc.band,
        "reasons": sc.reasons,
        "features": sc.features.__dict__,
    }
