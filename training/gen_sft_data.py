#!/usr/bin/env python3
"""
self-distillation 평가 데이터 생성 (면접 답변 -> 한국어 사고 + 점수 JSON).

실무 수준 특징:
- argparse: 직군/목표수/출력경로/새로시작을 인자로 (다른 직군·언어 재사용)
- 중단점 재개: 기존 출력의 이미 만든 질문은 건너뛰고 이어서 생성
- 매 건 즉시 flush 저장(체크포인트) + Ctrl+C 안전 종료 후 재개 명령 안내
- --fresh: 기존 파일을 타임스탬프 백업 후 새로 시작 (rm 데이터 유실 방지)
- 경로/파일 사전 검증(가정 금지) + 주기적 진행 통계 + 최종 요약
- 품질 게이트: 사고완주 / JSON파싱 / 한국어사고 / 스키마 / 점수형식 / 10점스케일·전부0 제외

사용 예:
  nohup python gen_sft_data.py --target 200 > gen_sft.log 2>&1 &
  nohup python gen_sft_data.py --occupation sm --target 150 --out sft_sm.jsonl > gen_sm.log 2>&1 &
  python gen_sft_data.py --fresh
"""
import os, sys, json, re, glob, random, time, argparse, shutil
import torch

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float32

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------- 설정 ----------------
DATA_DIR = "/workspace/interview_ai/data"
LLM_NAME = "LGAI-EXAONE/EXAONE-Deep-7.8B"
LLM_REV  = "e3f42b18f6b1"          # 절대 빼지 말 것 (main은 transformers v5 전제)
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MIN_ANS, MAX_ANS = 50, 1200
PREFILL  = "먼저 지원자의 답변을 평가 항목별로 살펴보겠습니다. "
SCORE_KEYS = ["technical_accuracy", "specificity", "logic", "communication"]
GEN_KW = dict(max_new_tokens=4096, do_sample=True, temperature=0.6, top_p=0.95)

RUBRIC = """당신은 웹개발자 채용 면접관입니다. [면접 질문]에 대한 [지원자 답변]을 평가하세요.

먼저 이 질문이 '기술 질문'인지 '인성·경험·동기 질문'인지 판단하세요.
- 기술 질문(알고리즘, 프로그래밍 언어, 설계, 개발 경험 등): 기술적 정확성을 평가합니다.
- 인성·경험·동기 질문(협업, 갈등 해결, 지원 동기, 부서 배치, 강점/약점 등): 기술 지식을 요구하지 말고, 답변의 태도·경험·사고방식이 타당하고 설득력 있는지를 평가합니다. 이런 질문에는 "코드, 알고리즘 등 기술적 접근이 필요하다"는 식의 피드백을 절대 쓰지 마세요.

각 항목을 0~100점 정수로 채점하세요 (10점 만점이 아닙니다). 모든 항목에 실제 점수를 매기고, 0으로 두지 마세요.
- technical_accuracy: 답변 내용의 정확성과 타당성. 기술 질문이면 기술적 정확성을, 인성·경험 질문이면 답변이 사실에 부합하고 설득력 있는지를 평가하세요. 기술 용어가 없다는 이유만으로 0점을 주지 마세요.
- specificity: 구체성과 깊이
- logic: 논리적 구조
- communication: 전달의 명확성
점수 기준: 우수 80~95 / 양호 60~75 / 보통 40~55 / 미흡 15~35 / 매우 미흡 0~10.

사고는 한국어로 핵심만 간결하게 쓰고, 사고 안에는 JSON을 쓰지 마세요.
사고를 마친 뒤 마지막에 아래 JSON 하나만 출력하세요 (JSON 외 텍스트 금지, 모든 내용 한국어):
{"scores":{"technical_accuracy":0,"specificity":0,"logic":0,"communication":0},"strengths":["..."],"improvements":["..."],"feedback":"..."}"""

def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

def is_korean(text, thr=0.3):
    if not text:
        return False
    kr = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
    return kr / max(len(text), 1) >= thr

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--occupation", default="ict", help="직군 코드 (ict/sm/ps/bm/mm/ard/rnd)")
    p.add_argument("--target", type=int, default=200, help="목표 성공 샘플 수")
    p.add_argument("--out", default="/workspace/interview_ai/sft_data.jsonl", help="출력 jsonl 경로")
    p.add_argument("--fresh", action="store_true", help="기존 출력을 백업 후 처음부터 생성")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    OUT_FILE, OCC = args.out, args.occupation

    # --- 경로/파일 사전 검증 (가정 금지) ---
    if not os.path.isdir(DATA_DIR):
        log(f"[중단] 데이터 폴더가 없습니다: {DATA_DIR}")
        sys.exit(1)
    files = glob.glob(os.path.join(DATA_DIR, "**", f"ckmk_*_{OCC}_*.json"), recursive=True)
    if not files:
        log(f"[중단] 패턴에 맞는 파일이 0개입니다 (occupation={OCC}). 직군 코드를 확인하세요.")
        sys.exit(1)
    random.seed(args.seed); random.shuffle(files)
    log(f">>> 직군={OCC} 파일 {len(files)}개 | 목표 {args.target} | 출력 {OUT_FILE}")

    # --- --fresh: 기존 파일 백업 후 제거 (rm 데이터 유실 방지) ---
    if args.fresh and os.path.exists(OUT_FILE):
        bak = f"{OUT_FILE}.{time.strftime('%Y%m%d_%H%M%S')}.bak"
        shutil.copy2(OUT_FILE, bak)
        os.remove(OUT_FILE)
        log(f">>> --fresh: 기존 파일을 백업하고 새로 시작합니다 -> {bak}")

    # --- 모델 로드 ---
    log(">>> EXAONE 로드 (4bit, revision 고정)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(LLM_NAME, revision=LLM_REV, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(LLM_NAME, revision=LLM_REV,
        quantization_config=bnb, device_map="auto", trust_remote_code=True).eval()
    log(f">>> 로드 완료. GPU {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # --- 중단점 재개: 이미 만든 질문 수집 ---
    done_q = set()
    if os.path.exists(OUT_FILE):
        for line in open(OUT_FILE, encoding="utf-8"):
            try:
                done_q.add(json.loads(line)["question"])
            except Exception:
                pass
        if done_q:
            log(f">>> 기존 {len(done_q)}개 발견 — 이어서 생성합니다")

    out = open(OUT_FILE, "a", encoding="utf-8")
    success = len(done_q); tried = 0; overalls = []
    interrupted = False
    try:
        for f in files:
            if success >= args.target:
                break
            try:
                d = json.load(open(f, encoding="utf-8"))["dataSet"]
            except Exception:
                continue
            q = (((d.get("question") or {}).get("raw") or {}).get("text") or "").strip()
            a = (((d.get("answer") or {}).get("raw") or {}).get("text") or "").strip()
            if not q or not a or q in done_q:
                continue
            if not (MIN_ANS <= len(a) <= MAX_ANS):
                continue
            done_q.add(q); tried += 1

            prompt = f"{RUBRIC}\n\n[면접 질문]\n{q}\n\n[지원자 답변]\n{a}"
            msgs = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + PREFILL
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to(llm.device)
            plen = enc["input_ids"].shape[1]
            try:
                with torch.no_grad():
                    o = llm.generate(**enc, **GEN_KW)
            except Exception as e:
                log(f"  [생성오류] {type(e).__name__}: {e}")
                continue
            gen = tok.decode(o[0][plen:], skip_special_tokens=False)

            # --- 품질 게이트 ---
            if "</thought>" not in gen:
                log(f"  [탈락:사고미완] ok={success} tried={tried}"); continue
            gen_thought = gen.split("</thought>")[0].strip()
            mm = re.search(r'```|\{[\s\S]*?"scores"', gen_thought)
            if mm:
                gen_thought = gen_thought[:mm.start()].strip()
            tail = gen.split("</thought>")[-1]
            m = re.search(r"\{.*\}", tail, re.DOTALL)
            if not m:
                log(f"  [탈락:JSON없음] ok={success} tried={tried}"); continue
            try:
                ev = json.loads(m.group(0))
            except Exception:
                log(f"  [탈락:JSON파싱] ok={success} tried={tried}"); continue
            if not is_korean(gen_thought):
                log(f"  [탈락:사고비한국어] ok={success} tried={tried}"); continue
            sc = ev.get("scores")
            if not isinstance(sc, dict):
                log(f"  [탈락:스키마] ok={success} tried={tried}"); continue
            vals = [sc.get(k) for k in SCORE_KEYS]
            if any(not isinstance(v, (int, float)) for v in vals):
                log(f"  [탈락:점수형식] ok={success} tried={tried}"); continue
            if sum(vals) == 0:
                log(f"  [탈락:점수전부0] ok={success} tried={tried}"); continue
            if max(vals) <= 10:
                log(f"  [탈락:10점스케일] ok={success} tried={tried} vals={vals}"); continue
            ev["overall"] = round(sum(vals) / len(vals))

            full_thought = (PREFILL + gen_thought).strip()
            rec = {"question": q, "answer": a, "thought": full_thought, "evaluation": ev,
                   "experience": (d.get("info") or {}).get("experience", ""),
                   "occupation": OCC}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            success += 1; overalls.append(ev["overall"])
            log(f"  [성공 {success}/{args.target}] overall={ev['overall']} scores={vals} len={len(full_thought)} (tried={tried})")

            if overalls and len(overalls) % 20 == 0:
                rate = 100 * len(overalls) / max(tried, 1)
                log(f"  --- 진행 통계: 성공률 {rate:.0f}% | overall avg {sum(overalls)//len(overalls)} | range {min(overalls)}~{max(overalls)}")
    except KeyboardInterrupt:
        interrupted = True
        log(">>> 사용자 중단(Ctrl+C) — 지금까지 저장된 분량은 안전합니다")
    finally:
        out.close()

    log(f"\n>>> {'중단됨' if interrupted else '완료'}: {success}개 저장 -> {OUT_FILE}")
    if overalls:
        log(f">>> 이번 세션 신규 {len(overalls)}개 | 성공률 {100*len(overalls)//max(tried,1)}% | overall avg {sum(overalls)//len(overalls)} | range {min(overalls)}~{max(overalls)}")
    if success < args.target:
        log(">>> 이어서 채우려면 같은 명령을 다시 실행하세요(자동 재개): "
            f"nohup python gen_sft_data.py --occupation {OCC} --target {args.target} --out {OUT_FILE} > gen_sft.log 2>&1 &")

if __name__ == "__main__":
    main()
