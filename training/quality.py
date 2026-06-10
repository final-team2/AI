#!/usr/bin/env python3
"""자가증류 표본 품질 게이트 — gen/filter 공용 단일 소스.
구체성·논리·전달 3축은 모든 답변에 적용(0 불가). 기술정확도는 인성문항 N/A(0) 허용."""
SCORE_KEYS = ["technical_accuracy", "specificity", "logic", "communication"]
CORE_KEYS  = ["specificity", "logic", "communication"]

def is_korean(text, thr=0.3):
    if not text: return False
    kr = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
    return kr / max(len(text), 1) >= thr

def _good_list(x):
    return (isinstance(x, list) and len(x) > 0 and
            all(isinstance(i, str) and len(i.strip()) >= 3 and "..." not in i for i in x))

def check_quality(ev, reasoning):
    sc = ev.get("scores")
    if not isinstance(sc, dict): return False, "스키마"
    vals = [sc.get(k) for k in SCORE_KEYS]
    if any((not isinstance(v, (int, float))) or isinstance(v, bool) for v in vals): return False, "점수형식"
    if max(vals) <= 10: return False, "10점스케일"
    if max(vals) > 100: return False, "점수범위초과"
    if any(sc.get(k) == 0 for k in CORE_KEYS): return False, "핵심축0점"
    fb = ev.get("feedback", "")
    if not (isinstance(fb, str) and len(fb.strip()) >= 15 and "..." not in fb and is_korean(fb)):
        return False, "피드백부실"
    if not _good_list(ev.get("strengths")): return False, "강점부실"
    if not _good_list(ev.get("improvements")): return False, "개선부실"
    if len((reasoning or "").strip()) < 150: return False, "사고과단"
    return True, "ok"
