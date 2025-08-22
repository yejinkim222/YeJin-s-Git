from __future__ import annotations
from typing import List, Dict
import numpy as np
import re

KOREAN_FILLERS = {"음", "어", "그", "음...", "어...", "저기", "그러니까", "뭐랄까", "어..", "음..", "그..", "저.."}

def compute_acoustic_features(segments: List[Dict], total_sec: float) -> Dict[str, float]:
    """
    음향 기반 지표 계산.
    입력:
      - segments: [{start, end, text}]
      - total_sec: 전체 길이(초)
    출력:
      - dict: speech_rate_wpm, articulation_rate_wpm, pause_ratio, pause_per_min, mean_seg_len
    """
    if total_sec <= 0:
        return dict(speech_rate_wpm=0, articulation_rate_wpm=0, pause_ratio=0, pause_per_min=0, mean_seg_len=0)

    seg_durs = [max(0.0, s["end"] - s["start"]) for s in segments]
    speech_time = sum(seg_durs)
    silence_time = max(0.0, total_sec - speech_time)

    pauses = []
    for i in range(1, len(segments)):
        gap = max(0.0, segments[i]["start"] - segments[i-1]["end"])
        if gap >= 0.25:
            pauses.append(gap)

    words = []
    for s in segments:
        words.extend(s["text"].split())
    word_count = len(words)

    speech_rate = (word_count / (total_sec / 60.0)) if total_sec > 0 else 0.0
    articulation = (word_count / (max(1e-6, speech_time) / 60.0)) if speech_time > 0 else 0.0

    return {
        "speech_rate_wpm": float(speech_rate),
        "articulation_rate_wpm": float(articulation),
        "pause_ratio": float(silence_time / total_sec),
        "pause_per_min": float(len(pauses) / (total_sec / 60.0)),
        "mean_seg_len": float(np.mean(seg_durs) if seg_durs else 0.0),
    }

def compute_text_features(transcript: str) -> Dict[str, float]:
    """
    텍스트 기반 지표(간단).
    입력: 전사 텍스트
    출력: ttr, mean_sentence_len, filler_ratio
    """
    text = (transcript or "").strip()
    if not text:
        return dict(ttr=0, mean_sentence_len=0, filler_ratio=0)

    sentences = [s.strip() for s in re.split(r"(?<=[\.!\?])\s+|\n+", text) if s.strip()]
    tokens = text.split()
    types = set(tokens)

    ttr = (len(types) / len(tokens)) if tokens else 0.0
    mean_sentence_len = float(np.mean([len(s.split()) for s in sentences]) if sentences else 0.0)
    filler_count = sum(1 for tok in tokens if tok in KOREAN_FILLERS)
    filler_ratio = (filler_count / len(tokens)) if tokens else 0.0

    return {"ttr": float(ttr), "mean_sentence_len": float(mean_sentence_len), "filler_ratio": float(filler_ratio)}
