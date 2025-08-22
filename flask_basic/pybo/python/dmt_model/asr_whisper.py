from __future__ import annotations
from typing import List, Dict, Tuple
import librosa

def transcribe_korean(audio_path: str,
                      model_size: str = "medium",
                      beam_size: int = 5) -> Tuple[str, List[Dict], float]:
    """
    Whisper(faster-whisper)로 한국어 전사를 수행합니다.
    입력:
      - audio_path: 오디오 파일 경로
      - model_size: tiny/base/small/medium/large-v3
      - beam_size : 빔 탐색 크기
    출력:
      - transcript: 전체 텍스트
      - segments  : [{start, end, text}] 리스트
      - duration  : 전체 길이(초)
    """
    from faster_whisper import WhisperModel  # 지연 import
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments_iter, info = model.transcribe(audio_path, language="ko", beam_size=beam_size)

    segments, texts = [], []
    for seg in segments_iter:
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
        texts.append(seg.text)

    transcript = " ".join(texts).strip()
    duration = info.duration if getattr(info, "duration", None) else _safe_duration(audio_path)
    return transcript, segments, float(duration)

def _safe_duration(path: str) -> float:
    """
    librosa로 길이를 계산합니다(Whisper 메타에 없을 때 대비).
    입력: 오디오 경로
    출력: 초 단위 길이
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    return float(len(y) / sr)
