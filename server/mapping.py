"""
mapping.py — EXAONE 평가(4축) → 프론트 리포트 표시 스키마 변환 (순수 후처리, 모델/GPU 불필요)

[입력] /evaluate 가 내는 4축 점수 (0~100)
    technical_accuracy(기술정확도) · specificity(구체성) · logic(논리) · communication(의사소통)

[출력 1] 문항별 표시 3점   : logic(논리) · clarity(명확성) · depth(깊이)
[출력 2] 리포트 종합 5점   : tech · comm · problem · attitude · logic  (+ overall, grade)

* 매핑 규칙(프론트 InterviewReport.tsx 스키마에 맞춤):
    문항별  logic   ← logic
            clarity ← communication
            depth   ← (specificity + technical_accuracy) / 2
    종합    tech     ← avg(technical_accuracy)
            comm     ← avg(communication)
            logic    ← avg(logic)
            problem  ← avg( (logic + specificity) / 2 )      # 문제해결 = 논리 + 구체적 접근
            attitude ← 표정/음성 지표 있으면 그것, 없으면 (comm+logic)/2 휴리스틱
    overall ← 5개 카테고리 평균,  grade ← 프론트와 동일 임계값
"""

# ----- 안전 유틸 -----
def _clamp(x):
    """숫자로 변환 + 0~100 범위로 클램프 + 반올림 정수. 비정상 입력은 0."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0
    return int(round(max(0.0, min(100.0, x))))


def _mean(xs):
    xs = [v for v in xs if v is not None]
    return sum(xs) / len(xs) if xs else 0.0


# ----- 등급 (프론트 InterviewReport.tsx GRADE 기준과 동일) -----
def grade_of(overall):
    o = overall
    if o >= 90:
        return "A+"
    if o >= 85:
        return "A"
    if o >= 80:
        return "B+"
    if o >= 75:
        return "B"
    if o >= 70:
        return "C+"
    return "C"


# ----- 문항별 4축 → 표시 3점 -----
def to_question_scores(ev):
    """단일 답변 평가(4축 dict) → {logic, clarity, depth}. 프론트 QA.scores 와 동일 키."""
    ev = ev or {}
    ta = _clamp(ev.get("technical_accuracy"))
    sp = _clamp(ev.get("specificity"))
    lo = _clamp(ev.get("logic"))
    co = _clamp(ev.get("communication"))
    return {
        "logic": lo,
        "clarity": co,
        "depth": _clamp((sp + ta) / 2),
    }


# ----- 표정/음성 → 태도·열정 점수 (없으면 None) -----
def _attitude_from_signals(voice, expression):
    vals = []
    if expression:
        # 표정 집계: 0~100 숫자 지표만 평균 (예: 시선안정/미소/자신감/고개움직임 또는 confidence 등)
        vals += [v for v in expression.values() if isinstance(v, (int, float))]
    if voice:
        # 음성에서 점수형 지표만 (말속도 WPM·필러워드 횟수 같은 비점수 값은 제외)
        for k in ("clarity", "clarity_score", "confidence"):
            v = voice.get(k)
            if isinstance(v, (int, float)):
                vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


# ----- 여러 문항 4축 → 리포트 종합 5점 -----
CATEGORY_ORDER = ["tech", "comm", "problem", "attitude", "logic"]
CATEGORY_LABELS = {
    "tech": "기술 지식",
    "comm": "커뮤니케이션",
    "problem": "문제 해결",
    "attitude": "태도·열정",
    "logic": "논리적 사고",
}


def to_report_scores(evaluations, voice=None, expression=None):
    """문항 4축 평가 리스트 → {overall, grade, categories[5]}.
    voice/expression(선택): 태도·열정 산출에 사용. 없으면 텍스트 기반 휴리스틱."""
    evaluations = [e for e in (evaluations or []) if e]
    if not evaluations:
        return {"overall": 0, "grade": "C", "categories": []}

    avg_ta = _mean([_clamp(e.get("technical_accuracy")) for e in evaluations])
    avg_sp = _mean([_clamp(e.get("specificity")) for e in evaluations])
    avg_lo = _mean([_clamp(e.get("logic")) for e in evaluations])
    avg_co = _mean([_clamp(e.get("communication")) for e in evaluations])

    attitude = _attitude_from_signals(voice, expression)
    if attitude is None:
        attitude = (avg_co + avg_lo) / 2.0  # 휴리스틱 (표정/음성 미연동 시)

    score = {
        "tech": _clamp(avg_ta),
        "comm": _clamp(avg_co),
        "problem": _clamp((avg_lo + avg_sp) / 2.0),
        "attitude": _clamp(attitude),
        "logic": _clamp(avg_lo),
    }
    overall = _clamp(_mean([score[k] for k in CATEGORY_ORDER]))
    categories = [
        {"key": k, "label": CATEGORY_LABELS[k], "score": score[k], "max": 100}
        for k in CATEGORY_ORDER
    ]
    return {"overall": overall, "grade": grade_of(overall), "categories": categories}


# ============================== 자체 검증 ==============================
if __name__ == "__main__":
    import json

    # 샘플: /evaluate 가 냈을 법한 4축 결과 3문항
    sample_evals = [
        {"technical_accuracy": 80, "specificity": 75, "logic": 80, "communication": 85},
        {"technical_accuracy": 78, "specificity": 70, "logic": 78, "communication": 82},
        {"technical_accuracy": 60, "specificity": 55, "logic": 70, "communication": 72},
    ]

    print("===== 문항별 표시 점수 (4축 → logic/clarity/depth) =====")
    for i, ev in enumerate(sample_evals, 1):
        qs = to_question_scores(ev)
        headline = round((qs["logic"] + qs["clarity"] + qs["depth"]) / 3)  # 프론트 접힘 점수와 동일
        print(f"Q{i} {ev}")
        print(f"   → {qs}  (헤드라인 {headline})")

    print("\n===== 리포트 종합 (표정/음성 미연동, 휴리스틱 attitude) =====")
    rep = to_report_scores(sample_evals)
    print(json.dumps(rep, ensure_ascii=False, indent=2))

    print("\n===== 리포트 종합 (표정 집계 연동 시 attitude override) =====")
    rep2 = to_report_scores(
        sample_evals,
        expression={"gaze": 78, "smile": 65, "confidence": 72, "head": 84},  # 표정 4종 예시
    )
    print(json.dumps(rep2, ensure_ascii=False, indent=2))

    # 엣지: 빈 리스트 / 비정상 값
    print("\n===== 엣지 케이스 =====")
    print("empty :", to_report_scores([]))
    print("dirty :", to_question_scores({"technical_accuracy": "n/a", "logic": None, "specificity": 130, "communication": 50}))
