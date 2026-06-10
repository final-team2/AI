"""
compare_base_vs_lora.py — base vs QLoRA 파인튜닝 비교 실험 (발표용)

같은 (질문, 답변)에 대해 EXAONE 를 어댑터 ON / OFF 로 동일 프롬프트·동일 greedy 디코딩으로 돌려,
파인튜닝이 '평가 출력의 포맷/스키마/스케일/한국어 추론 일관성'을 얼마나 개선했는지 수치화한다.
(어댑터의 목적은 점수 정확도가 아니라 출력 형식·행동의 안정화이므로 그 축으로 측정.)

- 모델 로딩·프롬프트·파서는 server.py 와 동일.
- 결과: 콘솔 요약표 + compare_results.md(발표용) + compare_results.json.

권장 실행: 서버를 잠시 내리고(pkill -f uvicorn) 실행 -> 끝나면 bash start.sh 로 복구.
N=8 쌍 × (어댑터+base) ≈ 15~20분.
"""
import os, json, re, time
import torch
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float32
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from json_repair import repair_json

LLM_NAME = "LGAI-EXAONE/EXAONE-Deep-7.8B"
LLM_REV  = "e3f42b18f6b1"
import sys as _sys
ADAPTER  = _sys.argv[1] if len(_sys.argv) > 1 else "/workspace/interview_ai/lora_adapter_v2"
TRAIN    = "/workspace/interview_ai/train.jsonl"
PREFILL  = "먼저 지원자의 답변을 평가 항목별로 살펴보겠습니다. "
MAX_NEW  = 3072
SCORE_KEYS = ["technical_accuracy", "specificity", "logic", "communication"]

OUT_MD   = "/workspace/interview_ai/compare_results.md"
OUT_JSON = "/workspace/interview_ai/compare_results.json"

# =========================== 테스트 세트 (다양한 난이도/유형) ===========================
TEST_PAIRS = [
    # (유형 메모, 질문, 답변)
    ("기술·강함", "프로세스와 스레드의 차이를 설명해 주세요.",
     "프로세스는 독립된 메모리 공간을 가지며, 스레드는 한 프로세스 내에서 메모리를 공유합니다. 스레드는 생성 비용이 적지만 공유 자원 접근 시 동기화가 필요해 레이스 컨디션에 주의해야 합니다."),
    ("기술·모호", "데이터베이스 인덱스가 무엇인가요?",
     "음 인덱스는 데이터를 빠르게 찾게 해주는 거예요. 정확히는 잘 모르겠지만 검색이 빨라지는 것 같습니다."),
    ("기술·오답", "REST API의 특징을 설명해 주세요.",
     "REST는 항상 서버에 클라이언트 상태를 저장하고, 쿠키로만 통신하는 방식입니다."),
    ("인성·좋음(tech=0)", "팀원과 갈등이 생겼을 때 어떻게 해결하시나요?",
     "먼저 상대의 입장을 충분히 들어보고 사실 관계를 정리한 뒤, 공동의 목표를 기준으로 절충안을 찾습니다. 이전 프로젝트에서 코드 컨벤션 충돌을 이런 방식의 합의로 해결한 경험이 있습니다."),
    ("인성·약함", "본인의 가장 큰 강점은 무엇인가요?",
     "그냥 열심히 합니다."),
    ("기술·중간", "HTTP와 HTTPS의 차이는 무엇인가요?",
     "HTTPS는 HTTP에 암호화가 추가된 것입니다. SSL/TLS로 통신 내용을 암호화해서 도청과 위변조를 막아 보안이 강화됩니다."),
    ("기술·짧음", "가비지 컬렉션이 무엇인가요?",
     "메모리를 자동으로 정리해 주는 기능이요."),
    ("기술·장황", "동시성과 병렬성의 차이를 설명해 주세요.",
     "동시성은 여러 작업을 아주 빠르게 번갈아 처리해서 마치 동시에 진행되는 것처럼 보이게 하는 개념이고요, 병렬성은 실제로 여러 코어에서 물리적으로 동시에 실행하는 것입니다. 싱글 코어에서도 동시성은 가능하지만 병렬성은 멀티코어가 필요합니다. 예를 들어 요리할 때 한 사람이 여러 냄비를 번갈아 보는 게 동시성, 여러 요리사가 각자 냄비를 맡는 게 병렬성이라고 볼 수 있습니다."),
]

# =========================== 파서 (server.py 동일: 3단 + json_repair) ===========================
def parse_json_lenient(text):
    tail = text.split("</thought>")[-1] if "</thought>" in text else text
    tail = re.sub(r"```(?:json)?", "", tail)
    m = re.search(r"\{.*\}", tail, re.DOTALL)
    if not m:
        return None
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
    except Exception:
        pass
    try:
        obj = repair_json(raw, return_objects=True)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

# =========================== 지표 ===========================
def hangul_ratio(text):
    hangul = sum(1 for c in text if "\uac00" <= c <= "\ud7a3")
    latin  = sum(1 for c in text if c.isascii() and c.isalpha())
    return hangul / (hangul + latin) if (hangul + latin) else 0.0

def evaluate_output(raw):
    """원문에서 지표 추출."""
    thought = raw.split("</thought>")[0] if "</thought>" in raw else raw
    parsed = parse_json_lenient(raw)
    m = {
        "thought_complete": "</thought>" in raw,
        "korean_reasoning": hangul_ratio(thought) > 0.30,
        "parse_ok": parsed is not None,
        "schema_ok": False,
        "scale_ok": False,         # 0~100 범위 + 0~10 스케일 오용 아님
        "low_scale_suspect": False,
        "out_of_range": False,
        "feedback_ok": False,
        "scores": None,
        "overall": None,
    }
    if parsed and isinstance(parsed.get("scores"), dict):
        sc = parsed["scores"]
        vals, ok = [], True
        for k in SCORE_KEYS:
            if k not in sc:
                ok = False; break
            try:
                vals.append(float(sc[k]))
            except (TypeError, ValueError):
                ok = False; break
        if ok and len(vals) == 4:
            m["schema_ok"] = True
            m["scores"] = {k: vals[i] for i, k in enumerate(SCORE_KEYS)}
            m["out_of_range"] = any(v < 0 or v > 100 for v in vals)
            m["low_scale_suspect"] = (max(vals) <= 10 and max(vals) > 0)   # 8/8/8/8 같은 0~10 오용
            m["scale_ok"] = (not m["out_of_range"]) and (not m["low_scale_suspect"])
        fb = parsed.get("feedback")
        m["feedback_ok"] = isinstance(fb, str) and len(fb.strip()) > 0
        try:
            m["overall"] = parsed.get("overall")
        except Exception:
            pass
    return m, (raw[-700:] if len(raw) > 700 else raw)

# =========================== 모델 ===========================
def load_model():
    print(">>> EXAONE + LoRA 로딩 (1~2분)...", flush=True)
    tok = AutoTokenizer.from_pretrained(LLM_NAME, revision=LLM_REV, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(LLM_NAME, revision=LLM_REV,
        quantization_config=bnb, device_map="auto", trust_remote_code=True).eval()
    model = PeftModel.from_pretrained(base, ADAPTER).eval()
    print(f">>> 로드 완료. GPU {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)
    return tok, model

def run_llm(tok, model, prompt, use_adapter):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + PREFILL
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    plen = enc["input_ids"].shape[1]
    with torch.no_grad():
        if use_adapter:
            o = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        else:
            with model.disable_adapter():
                o = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
    return tok.decode(o[0][plen:], skip_special_tokens=False)

# =========================== 메인 ===========================
def main():
    rubric = json.loads(open(TRAIN, encoding="utf-8").readline())["messages"][0]["content"].split("\n\n[면접 질문]")[0]
    tok, model = load_model()

    rows = []
    for idx, (kind, q, a) in enumerate(TEST_PAIRS, 1):
        prompt = f"{rubric}\n\n[면접 질문]\n{q}\n\n[지원자 답변]\n{a}"
        print(f"\n[{idx}/{len(TEST_PAIRS)}] {kind} — '{q[:24]}…'", flush=True)
        entry = {"kind": kind, "question": q, "answer": a}
        for mode, use_ad in (("lora", True), ("base", False)):
            t0 = time.time()
            raw = run_llm(tok, model, prompt, use_ad)
            m, tail = evaluate_output(raw)
            m["elapsed"] = round(time.time() - t0, 1)
            entry[mode] = {"metrics": m, "raw_tail": tail}
            flags = "".join([
                "P" if m["parse_ok"] else "·",
                "S" if m["schema_ok"] else "·",
                "R" if m["scale_ok"] else ("!" if m["low_scale_suspect"] else "·"),
                "K" if m["korean_reasoning"] else "·",
            ])
            print(f"    {mode:4s} [{flags}] overall={m['overall']} scores={m['scores']} ({m['elapsed']}s)", flush=True)
        rows.append(entry)

    # ---- 집계 ----
    keys = ["parse_ok", "schema_ok", "scale_ok", "thought_complete", "korean_reasoning", "feedback_ok"]
    labels = {"parse_ok": "JSON 파싱 성공", "schema_ok": "스키마 4축 완비", "scale_ok": "점수 0~100 범위",
              "thought_complete": "</thought> 완료", "korean_reasoning": "한국어 추론 유지", "feedback_ok": "feedback 존재"}
    N = len(rows)
    agg = {"lora": {}, "base": {}}
    for mode in ("lora", "base"):
        for k in keys:
            agg[mode][k] = sum(1 for r in rows if r[mode]["metrics"][k])
    low_lora = sum(1 for r in rows if r["lora"]["metrics"]["low_scale_suspect"])
    low_base = sum(1 for r in rows if r["base"]["metrics"]["low_scale_suspect"])

    print("\n" + "=" * 58)
    print(f"  비교 결과 요약  (N={N})")
    print("=" * 58)
    print(f"  {'지표':<18} {'LoRA(어댑터)':>14} {'base':>10}")
    print("  " + "-" * 50)
    for k in keys:
        print(f"  {labels[k]:<18} {agg['lora'][k]:>8}/{N:<5} {agg['base'][k]:>5}/{N}")
    print(f"  (0~10 스케일 의심)    LoRA {low_lora} / base {low_base}")
    print("=" * 58)

    # ---- 마크다운 리포트 ----
    def pctline(k):
        l, b = agg["lora"][k], agg["base"][k]
        return f"| {labels[k]} | {l}/{N} ({round(100*l/N)}%) | {b}/{N} ({round(100*b/N)}%) |"

    md = []
    md.append("# EXAONE 평가기: base vs QLoRA 파인튜닝 비교\n")
    md.append(f"- 모델: `{LLM_NAME}` (rev `{LLM_REV}`, 4-bit nf4)\n- 어댑터: `{ADAPTER}`\n- 표본: {N}개 (질문·답변 다양 유형) · 동일 프롬프트/루브릭 · greedy 디코딩 · 어댑터 ON/OFF만 차이\n")
    md.append("\n## 요약: 파인튜닝이 개선한 출력 일관성\n")
    md.append("어댑터의 목적은 점수 정확도가 아니라 **평가 출력의 형식·행동 안정화**다. 아래는 동일 입력에 대한 출력 품질 지표.\n")
    md.append("\n| 지표 | LoRA(어댑터) | base |\n|---|---|---|")
    for k in keys:
        md.append(pctline(k))
    md.append(f"| (0~10 스케일 오용 의심) | {low_lora}/{N} | {low_base}/{N} |")
    md.append("\n*P=파싱, S=스키마, R=스케일 정상, K=한국어 추론*\n")

    md.append("\n## 문항별 결과 (overall 점수 비교)\n")
    md.append("| # | 유형 | LoRA overall | base overall | 비고 |\n|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        lm, bm = r["lora"]["metrics"], r["base"]["metrics"]
        notes = []
        if not bm["parse_ok"]: notes.append("base 파싱 실패")
        elif not bm["schema_ok"]: notes.append("base 스키마 불완전")
        elif bm["low_scale_suspect"]: notes.append("base 0~10 스케일")
        if not bm["korean_reasoning"]: notes.append("base 영어 추론")
        md.append(f"| {i} | {r['kind']} | {lm['overall']} | {bm['overall']} | {', '.join(notes) or '-'} |")

    # base 가 실패하고 lora 가 성공한 대표 사례 1~2개 원문 첨부
    md.append("\n## 대표 사례 (base 실패 → LoRA 정상)\n")
    examples = [r for r in rows if (not r["base"]["metrics"]["parse_ok"] or not r["base"]["metrics"]["schema_ok"]
                                    or r["base"]["metrics"]["low_scale_suspect"] or not r["base"]["metrics"]["korean_reasoning"])][:2]
    if not examples:
        md.append("_이번 표본에서는 base 도 대체로 형식을 지켰습니다 (지표표의 비율로 차이 확인)._\n")
    for r in examples:
        md.append(f"\n### [{r['kind']}] {r['question']}\n")
        md.append(f"**base 출력(말미):**\n```\n{r['base']['raw_tail']}\n```\n")
        md.append(f"**LoRA 출력(말미):**\n```\n{r['lora']['raw_tail']}\n```\n")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n>>> 저장: {OUT_MD}")
    print(f">>> 저장: {OUT_JSON}")


if __name__ == "__main__":
    main()
