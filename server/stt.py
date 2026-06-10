#!/usr/bin/env python3
"""
STT 모듈: faster-whisper large-v3 (GPU/float16)로 오디오 -> 한국어 텍스트.

- 최초 호출 시 모델 1회 로드(지연 로딩, 스레드 안전), 이후 재사용.
- 브라우저가 보낸 webm/ogg/wav/mp3 등은 av(내장 ffmpeg)가 디코딩.
- vad_filter=True: 무음·비음성 구간 제거 -> 환각("감사합니다" 류) 감소.
- transcribe_bytes(audio_bytes, filename) -> {"text","language","duration","elapsed","segments"}

환경변수로 모델/디바이스 조정 가능: STT_MODEL, STT_DEVICE, STT_COMPUTE
"""
import os, time, tempfile, threading

_MODEL = None
_LOCK = threading.Lock()
_MODEL_NAME = os.environ.get("STT_MODEL", "large-v3")
_DEVICE     = os.environ.get("STT_DEVICE", "cuda")
_COMPUTE    = os.environ.get("STT_COMPUTE", "float16")

def get_model():
    """지연 로딩 + 더블체크 락 (동시 요청 안전)."""
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                from faster_whisper import WhisperModel
                _MODEL = WhisperModel(_MODEL_NAME, device=_DEVICE, compute_type=_COMPUTE)
    return _MODEL

def transcribe_bytes(audio_bytes, filename="audio.bin", language="ko"):
    suffix = os.path.splitext(filename)[1] or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(audio_bytes); tmp.flush(); tmp.close()
        model = get_model()
        t0 = time.time()
        segments, info = model.transcribe(
            tmp.name,
            language=language,
            beam_size=5,
            vad_filter=True,   # 무음/비음성 제거 -> 환각 감소
        )
        seg_list = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
        ]
        text = " ".join(s["text"] for s in seg_list).strip()
        return {
            "text": text,
            "language": info.language,
            "duration": round(info.duration, 2),
            "elapsed": round(time.time() - t0, 2),
            "segments": seg_list,
        }
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass

def is_loaded():
    return _MODEL is not None
