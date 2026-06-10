#!/usr/bin/env python3
"""
QLoRA SFT (4bit EXAONE) — completion-only 손실, 라벨 직접 마스킹.

기존 문제: TRL DataCollatorForCompletionOnlyLM가 응답마커([|assistant|]->[3])를
실제 시퀀스에서 못 찾아 라벨 전부 -100 -> loss 0.0(학습 안 됨).

이 버전:
- TRL 미사용. sft_data.jsonl을 직접 읽어 프롬프트(RUBRIC+Q+A, 생성프롬프트, +PREFILL)는 -100,
  완성부(추론 + </thought> + JSON + EOT)만 학습하도록 라벨을 손수 구성(경계 모호성 없음).
- 추론 경로와 동일한 프롬프트/프리필 -> 학습=추론 정합.
- 검증 분할 + epoch별 eval_loss, save_safetensors=False(EXAONE 공유텐서 .bin), 체크포인트 재개.
- [개선] eval 있으면 load_best_model_at_end + EarlyStopping(patience) -> eval 최저 체크포인트가
  자동으로 최종 어댑터가 됨(수동 체크포인트 선택 불필요). save_total_limit로 디스크 절약.
- 첫 샘플의 마스킹 경계/완성부 디코드를 출력해 눈으로 검증.

사용:
  python3 train_qlora.py --data sft_data_clean.jsonl --out lora_adapter_v3 --epochs 5 --patience 2
  python3 train_qlora.py --out lora_adapter_smoke --limit 180 --epochs 5   # 스모크
  nohup python3 train_qlora.py --data sft_data_clean.jsonl --out lora_adapter_v3 --epochs 5 > train_v3.log 2>&1 &
  nohup python3 train_qlora.py --data sft_data_clean.jsonl --out lora_adapter_v3 --resume > train_v3.log 2>&1 &
"""
import os, sys, json, re, glob, time, random, argparse
import torch

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float32

from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

LLM_NAME = "LGAI-EXAONE/EXAONE-Deep-7.8B"
LLM_REV  = "e3f42b18f6b1"          # 절대 빼지 말 것 (main은 transformers v5 전제)
PREFILL  = "먼저 지원자의 답변을 평가 항목별로 살펴보겠습니다. "
SCORE_KEYS = ["technical_accuracy", "specificity", "logic", "communication"]

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

def log(m): print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)

def latest_checkpoint(d):
    cks = [c for c in glob.glob(os.path.join(d, "checkpoint-*")) if re.search(r"checkpoint-(\d+)$", c)]
    if not cks: return None
    cks.sort(key=lambda p: int(re.search(r"checkpoint-(\d+)$", p).group(1)))
    return cks[-1]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/interview_ai/sft_data_clean.jsonl")
    p.add_argument("--out",  default="/workspace/interview_ai/lora_adapter_v3")
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    a = parse_args(); random.seed(a.seed)
    log("="*60)
    log(">>> QLoRA SFT (completion-only, 직접 마스킹)")
    log(">>> [개선] eval 있으면 load_best_model_at_end + EarlyStopping(patience) -> best 자동 저장")
    log(">>> [운영주의] 학습 중 Pod stop 금지(끊기면 --resume) · 다른 GPU 작업 금지(OOM) · 고정 스택 변경 금지")
    log("="*60)
    if not os.path.exists(a.data):
        log(f"[중단] 데이터 없음: {a.data}"); sys.exit(1)

    tok = AutoTokenizer.from_pretrained(LLM_NAME, revision=LLM_REV, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    EOT = tok.convert_tokens_to_ids("[|endofturn|]")
    if EOT is None or EOT == tok.unk_token_id:
        EOT = tok.eos_token_id
    pcore = PREFILL.strip()

    probe = tok.apply_chat_template([{"role":"user","content":"x"}], tokenize=False, add_generation_prompt=True)
    if "<thought>" not in probe[-60:]:
        log(">>> [경고] 생성프롬프트 끝에 <thought>가 안 보임 — 완성부 포맷을 점검하세요")

    def build(rec):
        q  = (rec.get("question") or "").strip()
        an = (rec.get("answer") or "").strip()
        th = (rec.get("thought") or "").strip()
        ev = rec.get("evaluation") or {}
        sc = ev.get("scores")
        if not q or not an or not th or not isinstance(sc, dict): return None
        scores = {k: sc.get(k) for k in SCORE_KEYS}
        if any(scores[k] is None for k in SCORE_KEYS): return None
        out_json = {"scores": scores, "strengths": ev.get("strengths", []),
                    "improvements": ev.get("improvements", []), "feedback": ev.get("feedback", "")}
        user = f"{RUBRIC}\n\n[면접 질문]\n{q}\n\n[지원자 답변]\n{an}"
        prompt = tok.apply_chat_template([{"role":"user","content":user}], tokenize=False, add_generation_prompt=True)
        if th.startswith(pcore):
            reasoning = th[len(pcore):].lstrip(); prompt = prompt + PREFILL
        else:
            reasoning = th
        completion = reasoning.rstrip() + "\n</thought>\n" + json.dumps(out_json, ensure_ascii=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"] + [EOT]
        ids = p_ids + c_ids
        return {"input_ids": ids, "attention_mask": [1]*len(ids),
                "labels": [-100]*len(p_ids) + c_ids, "plen": len(p_ids)}

    recs, sk_long, sk_bad = [], 0, 0
    for line in open(a.data, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: sk_bad += 1; continue
        ex = build(r)
        if ex is None: sk_bad += 1; continue
        if len(ex["input_ids"]) > a.max_len: sk_long += 1; continue
        recs.append(ex)
    if not recs:
        log("[중단] 유효 샘플 0개"); sys.exit(1)
    random.shuffle(recs)
    if a.limit > 0:
        recs = recs[:a.limit]
        log(f">>> [스모크] --limit {a.limit} 적용")
    n_val = int(len(recs)*a.val_ratio) if a.val_ratio > 0 else 0
    val, train = (recs[:n_val], recs[n_val:]) if n_val > 0 else ([], recs)
    log(f">>> 샘플 {len(recs)} (학습 {len(train)}/검증 {len(val)}) | 스킵: 길이초과 {sk_long}, 불량 {sk_bad}")

    s0 = train[0]; lab_n = sum(1 for x in s0["labels"] if x != -100)
    log(f">>> [검증] 샘플0 총토큰 {len(s0['input_ids'])} | 프롬프트(마스킹) {s0['plen']} | 학습대상 {lab_n}")
    log(">>> [검증] 프롬프트 끝: " + tok.decode(s0["input_ids"][max(0,s0['plen']-26):s0['plen']]).replace("\n","\\n"))
    log(">>> [검증] 완성부 앞: " + tok.decode(s0["input_ids"][s0['plen']:s0['plen']+34]).replace("\n","\\n"))
    if lab_n == 0:
        log("[중단] 학습대상 토큰 0 — 마스킹 오류"); sys.exit(1)

    keys = ("input_ids","attention_mask","labels")
    train_ds = Dataset.from_list([{k:e[k] for k in keys} for e in train])
    eval_ds  = Dataset.from_list([{k:e[k] for k in keys} for e in val]) if val else None

    log(">>> 모델 로드 (4bit, revision 고정)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(LLM_NAME, revision=LLM_REV,
        quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM", bias="none"))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    pad_id = tok.pad_token_id
    def collate(feats):
        m = max(len(f["input_ids"]) for f in feats)
        ii, am, lb = [], [], []
        for f in feats:
            d = m - len(f["input_ids"])
            ii.append(f["input_ids"] + [pad_id]*d)
            am.append(f["attention_mask"] + [0]*d)
            lb.append(f["labels"] + [-100]*d)
        return {"input_ids": torch.tensor(ii), "attention_mask": torch.tensor(am), "labels": torch.tensor(lb)}

    # eval 있을 때만 best-model 자동 저장 + 조기종료 (없으면 기존처럼 마지막 epoch 저장)
    best_kwargs = {}
    callbacks = None
    if eval_ds is not None:
        best_kwargs = dict(load_best_model_at_end=True, metric_for_best_model="eval_loss",
                           greater_is_better=False, save_total_limit=2)
        callbacks = [EarlyStoppingCallback(early_stopping_patience=a.patience)]
        log(f">>> best 체크포인트 자동 채택 ON (metric=eval_loss, patience={a.patience}, save_total_limit=2)")

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr,
        warmup_ratio=0.03, lr_scheduler_type="cosine", bf16=True,
        logging_steps=5, save_strategy="epoch",
        eval_strategy=("epoch" if eval_ds is not None else "no"),
        optim="paged_adamw_8bit", save_safetensors=False,
        seed=a.seed, report_to="none", remove_unused_columns=False,
        label_names=["labels"], **best_kwargs)

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=eval_ds, data_collator=collate, callbacks=callbacks)
    ckpt = latest_checkpoint(a.out) if a.resume else None
    if ckpt: log(f">>> 체크포인트 재개: {ckpt}")
    log(">>> 학습 시작")
    t0 = time.time(); res = trainer.train(resume_from_checkpoint=ckpt); dt = time.time()-t0

    # load_best_model_at_end=True 이면 이 시점 model은 eval 최저 체크포인트 -> save_model이 곧 best 저장
    trainer.save_model(a.out); tok.save_pretrained(a.out)
    summ = {"base_model": LLM_NAME, "revision": LLM_REV,
            "train_samples": len(train), "val_samples": len(val),
            "epochs": a.epochs, "lr": a.lr, "lora_r": a.r, "lora_alpha": a.alpha,
            "batch": a.batch, "grad_accum": a.grad_accum, "max_len": a.max_len,
            "patience": (a.patience if eval_ds is not None else None),
            "load_best_model_at_end": (eval_ds is not None),
            "train_runtime_sec": round(dt,1), "train_loss": round(float(res.training_loss),4),
            "completion_only": True, "save_safetensors": False,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        if eval_ds is not None:
            summ["eval_loss_best"] = round(float(trainer.evaluate().get("eval_loss",0)),4)
    except Exception as e:
        log(f"[경고] 최종 eval 실패: {e}")
    json.dump(summ, open(os.path.join(a.out,"training_summary.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f">>> 완료. train_loss={summ['train_loss']} eval_loss(best)={summ.get('eval_loss_best','-')} | {dt/60:.1f}분 | {a.out}")
    if eval_ds is not None:
        log(">>> (best 모델이 자동 로드되어 최종 어댑터로 저장됨 — 체크포인트 수동 선택 불필요)")

if __name__ == "__main__":
    main()
