#!/usr/bin/env python3
"""sft_data.jsonl -> 품질 게이트 통과분만 sft_data_clean.jsonl.
gen이 돌고 있어도 안전(입력 읽기전용, 출력은 별도 파일). 사유별 탈락 통계 출력."""
import json, argparse, collections
from quality import check_quality, SCORE_KEYS

PREFILL = "먼저 지원자의 답변을 평가 항목별로 살펴보겠습니다."

def reasoning_of(thought):
    t = (thought or "").strip()
    return t[len(PREFILL):].lstrip() if t.startswith(PREFILL) else t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="/workspace/interview_ai/sft_data.jsonl")
    ap.add_argument("--out", default="/workspace/interview_ai/sft_data_clean.jsonl")
    a = ap.parse_args()
    kept = 0; drop = collections.Counter(); overalls = []; seen = set()
    with open(a.src, encoding="utf-8") as fin, open(a.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: drop["json깨짐"] += 1; continue
            q = (r.get("question") or "").strip()
            if not q or q in seen: drop["중복/빈질문"] += 1; continue
            ev = r.get("evaluation") or {}
            ok, why = check_quality(ev, reasoning_of(r.get("thought")))
            if not ok: drop[why] += 1; continue
            seen.add(q)
            vals = [ev["scores"][k] for k in SCORE_KEYS]
            ev["overall"] = round(sum(vals) / len(vals))
            fout.write(json.dumps(r, ensure_ascii=False) + "\n"); kept += 1; overalls.append(ev["overall"])
    tot = kept + sum(drop.values())
    print(f"입력 {tot}줄 -> 통과 {kept} / 탈락 {sum(drop.values())} ({round(100*kept/max(tot,1))}% 유지)")
    print("탈락 사유:", dict(drop.most_common()))
    if overalls:
        overalls.sort()
        print(f"overall 분포: 평균 {round(sum(overalls)/len(overalls))} | 중앙 {overalls[len(overalls)//2]} | 범위 {overalls[0]}~{overalls[-1]}")
    print(f"-> {a.out}")

if __name__ == "__main__":
    main()
