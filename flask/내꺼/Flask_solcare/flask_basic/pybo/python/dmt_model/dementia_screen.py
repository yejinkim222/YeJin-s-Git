from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import os
import numpy as np

from .asr_whisper import transcribe_korean
from .dementia_features import compute_acoustic_features, compute_text_features

FEATURE_ORDER = [
    "speech_rate_wpm", "articulation_rate_wpm",
    "pause_ratio", "pause_per_min", "mean_seg_len",
    "ttr", "mean_sentence_len", "filler_ratio",
]

def build_feature_vector(acoustic: Dict[str, float], textf: Dict[str, float]) -> np.ndarray:
    """
    두 dict를 고정 순서 벡터로 변환.
    입력: acoustic, textf
    출력: np.ndarray shape (len(FEATURE_ORDER),)
    """
    merged = {**acoustic, **textf}
    return np.array([float(merged.get(k, 0.0)) for k in FEATURE_ORDER], dtype=np.float32)

def load_classifier(pkl_path: str) -> Optional[object]:
    """
    joblib pickle 로드(없으면 None).
    입력: 경로
    출력: sklearn estimator 또는 None
    """
    if not pkl_path or not os.path.exists(pkl_path):
        return None
    import joblib
    return joblib.load(pkl_path)

def predict_risk(features: np.ndarray, clf: Optional[object] = None, threshold: float = 0.5):
    """
    분류기 확률→위험도/라벨. 없으면 휴리스틱.
    입력: features(1D), clf, threshold
    출력: (score[0..1], label["high"|"low"])
    """
    if clf is not None and hasattr(clf, "predict_proba"):
        prob = float(clf.predict_proba(features.reshape(1, -1))[0, 1])
    else:
        s = 0.0
        s += (0.8 if features[0] < 80 else 0.2)     # speech_rate_wpm
        s += (0.8 if features[2] > 0.35 else 0.2)   # pause_ratio
        s += (0.7 if features[5] < 0.40 else 0.2)   # ttr
        s += (0.7 if features[6] < 8    else 0.2)   # mean_sentence_len
        prob = max(0.0, min(1.0, s / 3.0))
    label = "high" if prob >= threshold else "low"
    return prob, label

@dataclass
class ScreenResult:
    transcript: str
    features: Dict[str, float]
    risk_score: float
    risk_label: str

def screen_audio_for_dementia(audio_path: str, asr_model: str = "medium", clf_pkl: Optional[str] = None) -> ScreenResult:
    """
    전체 파이프라인: ASR→피처→분류
    입력:
      - audio_path: wav/mp3 등 파일 경로
      - asr_model : faster-whisper 모델 사이즈
      - clf_pkl   : 분류기 pickle 경로(없으면 휴리스틱)
    출력: ScreenResult
    """
    transcript, segments, dur = transcribe_korean(audio_path, model_size=asr_model, beam_size=5)
    acoustic = compute_acoustic_features(segments, dur)
    textf = compute_text_features(transcript)
    x = build_feature_vector(acoustic, textf)
    clf = load_classifier(clf_pkl) if clf_pkl else None
    score, label = predict_risk(x, clf=clf, threshold=0.5)
    return ScreenResult(transcript=transcript, features={**acoustic, **textf}, risk_score=float(score), risk_label=label)
