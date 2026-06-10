#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice.py — STT 결과로 음성 전달력 '보조 지표'를 계산하는 순수 함수 (GPU 불필요).

입력: /interview/stt 의 result 딕셔너리
      {"text": str, "duration": float, "segments": [{"start","end","text"}], ...}
출력: {wpm, word_count, duration_sec, speaking_sec, speaking_ratio,
       filler_count, filler_ratio, pause_count, long_pause_count, longest_pause,
       total_pause_sec, delivery_score, pace_label, note}

근거/한계:
- 단어 타임스탬프가 없어 '문장(segment) 사이 공백'으로 휴지를 추정한다.
  stt.py는 vad_filter=True라 무음이 제거되므로 segment 간 갭 = 실제 휴지에 가깝다.
- 필러는 STT 텍스트 기반이라 근사치다(Whisper가 필러를 누락/병합할 수 있음).
- delivery_score는 0~100 휴리스틱 보조 지표이며 합격 예측이 아니다.
"""
import re

# 언어별 필러 (standalone 토큰). 한국어 단음절 중 위험 토큰(그/저/뭐/막 등)은 오탐을 줄이려 제외.
FILLERS = {
    "ko": {"음", "어", "에", "그니까", "그러니까", "저기", "뭐랄까", "음음", "어어"},
    "en": {"um", "uh", "er", "erm", "uhh", "umm", "hmm"},
}
# 다어절 필러(구) — 부분 문자열로 카운트
FILLER_PHRASES = {
    "ko": ["그 뭐냐", "뭐랄까"],
    "en": ["you know", "i mean", "kind of", "sort of"],
}
# 적정 말속도 밴드 (분당 어절/단어) — 면접 권장치(대략)
PACE_BAND = {"ko": (110, 180), "en": (120, 160)}
PAUSE_SEC = 0.7       # 이 이상 공백이면 '휴지'
LONG_PAUSE_SEC = 1.5  # 이 이상이면 '긴 휴지'


def _tokens(text):
    cleaned = re.sub(r"[^\w가-힣]+", " ", text)   # 구두점 제거
    return [t for t in cleaned.split() if t]


def _count_fillers(text, lang):
    toks = _tokens(text.lower() if lang == "en" else text)
    fset = FILLERS.get(lang, set())
    n = sum(1 for t in toks if t in fset)
    low = text.lower()
    for ph in FILLER_PHRASES.get(lang, []):
        n += low.count(ph)
    return n


def _clamp(x, lo=0, hi=100):
    return int(round(max(lo, min(hi, x))))


def voice_metrics(result, lang="ko"):
    lang = lang if lang in ("ko", "en") else "ko"
    result = result or {}
    text = result.get("text", "") or ""
    try:
        duration = float(result.get("duration", 0) or 0)
    except (TypeError, ValueError):
        duration = 0.0
    segments = result.get("segments", []) or []

    words = _tokens(text)
    word_count = len(words)
    wpm = round(word_count / (duration / 60.0)) if duration > 0 and word_count else 0

    # 발화시간 / 휴지 (segment 기반)
    speaking_sec = 0.0
    gaps = []
    prev_end = None
    for s in segments:
        try:
            st = float(s.get("start", 0)); en = float(s.get("end", 0))
        except (TypeError, ValueError):
            continue
        if en > st:
            speaking_sec += (en - st)
        if prev_end is not None:
            gap = st - prev_end
            if gap > 0:
                gaps.append(gap)
        prev_end = en
    speaking_sec = round(speaking_sec, 2)
    speaking_ratio = round(speaking_sec / duration, 3) if duration > 0 else 0.0
    pause_gaps = [g for g in gaps if g >= PAUSE_SEC]
    long_pauses = [g for g in gaps if g >= LONG_PAUSE_SEC]
    pause_count = len(pause_gaps)
    long_pause_count = len(long_pauses)
    longest_pause = round(max(gaps), 2) if gaps else 0.0
    total_pause_sec = round(sum(pause_gaps), 2)

    # 필러
    filler_count = _count_fillers(text, lang)
    filler_ratio = round(filler_count / word_count, 3) if word_count else 0.0

    # ----- delivery_score (0~100 휴리스틱) -----
    lo, hi = PACE_BAND[lang]
    if wpm == 0:
        pace = 0
    elif lo <= wpm <= hi:
        pace = 100
    elif wpm < lo:
        pace = _clamp(100 - (lo - wpm) * 1.2)   # 너무 느림
    else:
        pace = _clamp(100 - (wpm - hi) * 1.0)   # 너무 빠름
    fluency = _clamp(100 - filler_ratio * 600) if word_count else 0   # 필러 10% → 40
    flow = _clamp(speaking_ratio * 100 - long_pause_count * 8) if word_count else 0
    delivery = _clamp(0.40 * pace + 0.30 * fluency + 0.30 * flow)

    if wpm == 0:
        pace_label = "측정 불가" if lang == "ko" else "N/A"
    elif wpm < lo:
        pace_label = "느린 편" if lang == "ko" else "Slow"
    elif wpm > hi:
        pace_label = "빠른 편" if lang == "ko" else "Fast"
    else:
        pace_label = "적정" if lang == "ko" else "Good"

    note = ("음성 지표는 전달력 참고용 보조 지표입니다. 필러는 STT 특성상 근사치이며, 휴지는 문장 단위로 추정됩니다."
            if lang == "ko" else
            "Voice metrics are a supplementary delivery indicator. Filler counts are approximate (STT-dependent); pauses are estimated at the segment level.")

    return {
        "wpm": wpm,
        "word_count": word_count,
        "duration_sec": round(duration, 2),
        "speaking_sec": speaking_sec,
        "speaking_ratio": speaking_ratio,
        "filler_count": filler_count,
        "filler_ratio": filler_ratio,
        "pause_count": pause_count,
        "long_pause_count": long_pause_count,
        "longest_pause": longest_pause,
        "total_pause_sec": total_pause_sec,
        "delivery_score": delivery,
        "pace_label": pace_label,
        "note": note,
    }


if __name__ == "__main__":
    import json
    cases = [
        ("정상 ko",
         "ko",
         {"text": "안녕하세요 저는 백엔드 개발자입니다 지난 프로젝트에서 결제 시스템을 설계하고 운영했습니다 트래픽이 몰릴 때 캐시와 큐로 부하를 분산했습니다",
          "duration": 14.0,
          "segments": [{"start": 0.2, "end": 4.0, "text": "안녕하세요 저는 백엔드 개발자입니다"},
                       {"start": 4.8, "end": 9.0, "text": "지난 프로젝트에서 결제 시스템을 설계하고 운영했습니다"},
                       {"start": 10.5, "end": 13.8, "text": "트래픽이 몰릴 때 캐시와 큐로 부하를 분산했습니다"}]}),
        ("필러 많음 ko",
         "ko",
         {"text": "음 그니까 어 저는 그 음 그니까 자바를 좀 했고 어 스프링도 음 조금",
          "duration": 12.0,
          "segments": [{"start": 0.0, "end": 12.0, "text": "음 그니까 어 저는 그 음 그니까 자바를 좀 했고 어 스프링도 음 조금"}]}),
        ("느리고 긴 휴지 ko",
         "ko",
         {"text": "네 그건 잘 모르겠습니다",
          "duration": 18.0,
          "segments": [{"start": 1.0, "end": 3.0, "text": "네"},
                       {"start": 8.0, "end": 11.0, "text": "그건"},
                       {"start": 15.0, "end": 17.0, "text": "잘 모르겠습니다"}]}),
        ("영어 정상 en",
         "en",
         {"text": "I led the backend team and designed a payment system that handled high traffic using caching and message queues",
          "duration": 9.0,
          "segments": [{"start": 0.1, "end": 4.5, "text": "I led the backend team and designed a payment system"},
                       {"start": 5.0, "end": 8.8, "text": "that handled high traffic using caching and message queues"}]}),
        ("빈 입력", "ko", {"text": "", "duration": 0, "segments": []}),
    ]
    for name, lang, r in cases:
        print(f"\n[{name}]  (lang={lang})")
        print(json.dumps(voice_metrics(r, lang), ensure_ascii=False))
