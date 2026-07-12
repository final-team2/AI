import os, json, re, time
import torch
import torch.nn.functional as F
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float32
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from contextlib import asynccontextmanager
from json_repair import repair_json
from mapping import to_question_scores, to_report_scores

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
RAG_DIR   = "/workspace/interview_ai/rag"
LLM_NAME  = "LGAI-EXAONE/EXAONE-Deep-7.8B"
LLM_REV   = "e3f42b18f6b1"          # 절대 빼지 말 것 (main은 transformers v5 전제라 스택이 깨짐)
EMB_MODEL = "BAAI/bge-m3"
ADAPTER   = "/workspace/interview_ai/lora_adapter_v3"
TRAIN     = "/workspace/interview_ai/train.rubric.jsonl"

# ---- 추론 예산 강제 (폭주/524 방지) ----
BUDGET_FORCE  = True    # 추론이 예산 넘으면 </thought> 강제 후 답변 생성 (False면 기존 동작)
REASON_BUDGET = 500    # 추론 단계 토큰 예산 (측정된 정상 최대 1156 위)
ANSWER_BUDGET = 768     # 강제 종료 후 답변(JSON/텍스트) 생성 예산


LANGS = ("ko", "en")
def norm_lang(lang):
    return lang if lang in LANGS else "ko"

# ---- prefills (언어별) ----
PREFILL = {
    "ko": "먼저 지원자의 답변을 평가 항목별로 살펴보겠습니다. ",
    "en": "First, let me assess the candidate's answer against each criterion. ",
}
GEN_PREFILL = {
    "ko": "먼저 지원자의 이력서와 채용공고를 살펴보고, 어떤 질문이 적합할지 항목별로 생각해보겠습니다. ",
    "en": "First, let me review the candidate's resume and the job posting and think about which questions fit. ",
}
FU_PREFILL = {
    "ko": "먼저 지원자의 답변이 구체적인지 두루뭉실한지, 근거가 충분한지 진단한 뒤 후속 질문을 정하겠습니다. ",
    "en": "First, let me diagnose whether the candidate's answer is specific or vague and well-grounded, then decide the follow-up. ",
}
RP_PREFILL = {
    "ko": "먼저 지원자의 면접 결과 전체를 살펴보겠습니다. ",
    "en": "First, let me review the candidate's overall interview results. ",
}

# ---- 영어 채점 rubric (한국어는 train.jsonl에서 M['rubric']로 로드) ----
RUBRIC_EN = """You are a web-developer hiring interviewer. Evaluate the [Candidate Answer] to the [Interview Question].

First decide whether this is a 'technical question' or a 'behavioral / experience / motivation question' (collaboration, conflict, motivation, team fit, strengths/weaknesses). For behavioral questions do NOT demand technical knowledge or say "a code/technical approach is needed"; assess whether the attitude, experience, and reasoning are sound and convincing.

Score each of the five axes as an INTEGER from 0 to 100. This is a 0-100 scale, NOT a 0-10 scale. A strong answer is about 85 (NOT 8); an average answer is about 55 (NOT 5); a weak answer is about 25. Be discriminating: vague or hedging answers ("I'm not sure, maybe...") should score low on specificity and technical_accuracy.
- technical_accuracy: correctness and validity of the content
- specificity: concreteness and depth
- logic: logical structure
- depth: understanding of underlying principles, root causes, and deeper reasoning (the "why" behind the answer)
- communication: clarity of delivery
Also score the answer's STAR structure on a 1-5 scale per item (NOT 0-100). Score each item independently; do not average or sum.
- situation: 5=concrete, complex context (when, which project), 3=present but generic, 1=no situation given
- task: 5=own role/goal clear, 3=present but passive ("was asked to"), 1=unclear
- action: 5=own concrete actions step by step ("I did"), 3=present but lots of "we" or abstract, 1=unclear or hypothetical ("I would")
- result: 5=quantified outcome (numbers/effect), 3=present but qualitative, 1=no result
STAR matters for behavioral/experience questions; for a pure technical/knowledge question it is natural for STAR to be weak, so assign a low score (1-2). Distribution: a typical or generic answer scores mostly 2-3 per item. Give 5 only when truly specific and exemplary, 4 only when clearly strong. Do NOT give high scores to all items; discriminate per item by the answer's actual level. This 1-5 scale is separate from the 0-100 axis scores.
Bands: excellent 80-95 / good 60-75 / fair 40-55 / weak 15-35 / very weak 0-10.

Think concisely in English. Do NOT write any JSON inside your thinking. Apply the criteria carefully to decide each score, but once decided, do not repeat the same deliberation; output the JSON immediately. Do not repeat identical second-guessing ("but", "recalculating"). After thinking, output ONLY one JSON object (all content in English), exactly in this shape:
{"scores":{"technical_accuracy":85,"specificity":70,"logic":80,"depth":78,"communication":75},"star":{"situation":4,"task":4,"action":5,"result":4},"strengths":["..."],"improvements":["..."],"feedback":"..."}

Worked example (for format and scale calibration only):
[Interview Question] Explain what an index is in a database.
[Candidate Answer] An index is like a book's table of contents; it lets the database find rows without scanning the whole table, speeding up reads but slightly slowing writes.
{"scores":{"technical_accuracy":82,"specificity":68,"logic":80,"depth":75,"communication":85},"star":{"situation":1,"task":1,"action":1,"result":1},"strengths":["Clear analogy","Notes the read/write trade-off"],"improvements":["Could mention B-tree structure or which columns to index"],"feedback":"Accurate and well-communicated; add concrete detail on index internals to score higher."}"""

# ---- 면접관 페르소나 (언어별) ----
PERSONAS = {
    "ko": {
        "default":     "당신은 IT 직무 면접관입니다.",
        "senior_tech": "당신은 구현 디테일과 기술적 트레이드오프를 끝까지 파고드는 시니어 기술 면접관입니다.",
        "culture_fit": "당신은 협업·태도·지원 동기 같은 컬처핏을 중점적으로 평가하는 면접관입니다.",
        "pressure":    "당신은 날카롭고 도전적인 질문으로 지원자를 압박하는 면접관입니다.",
        "mentor":      "당신은 편안한 분위기에서 지원자의 강점을 끌어내는 친근한 멘토형 면접관입니다.",
    },
    "en": {
        "default":     "You are an IT job interviewer.",
        "senior_tech": "You are a senior technical interviewer who probes implementation details and technical trade-offs in depth.",
        "culture_fit": "You are an interviewer who focuses on culture fit such as collaboration, attitude, and motivation.",
        "pressure":    "You are an interviewer who challenges the candidate with sharp, demanding questions.",
        "mentor":      "You are a friendly, mentor-style interviewer who draws out the candidate's strengths in a relaxed atmosphere.",
    },
}

SCORE_KEYS = ["technical_accuracy", "specificity", "logic", "depth", "communication"]
LABEL = {
    "ko": {"technical_accuracy": "기술 정확도", "specificity": "답변 구체성", "logic": "논리성", "depth": "심화이해", "communication": "의사소통"},
    "en": {"technical_accuracy": "Technical accuracy", "specificity": "Specificity", "logic": "Logic", "depth": "Depth", "communication": "Communication"},
}
LIMITS = {"question": 2000, "answer": 6000, "topic": 500,
          "resume": 12000, "job_posting": 12000}

# ---- 운영 메시지 (언어별) ----
MSG = {
    "ko": {
        "loading": "서버가 아직 모델을 로딩 중입니다. 잠시 후 다시 시도하세요.",
        "empty": "'{name}' 값이 비어 있습니다.",
        "too_long": "'{name}' 값이 너무 깁니다 (최대 {max}자, 현재 {len}자).",
        "eval_fail": "평가 결과 JSON 파싱 실패",
        "gen_fail": "질문 생성 JSON 파싱 실패",
        "fu_fail": "꼬리질문 JSON 파싱 실패",
        "results_empty": "'results'가 비어 있습니다. 최소 1개 문항 결과가 필요합니다.",
        "qbank_missing": "영어 질문 은행이 아직 구축되지 않았습니다. /interview/generate를 사용하세요.",
    },
    "en": {
        "loading": "The server is still loading the model. Please try again shortly.",
        "empty": "'{name}' is empty.",
        "too_long": "'{name}' is too long (max {max} chars, got {len}).",
        "eval_fail": "Failed to parse evaluation JSON",
        "gen_fail": "Failed to parse generated-questions JSON",
        "fu_fail": "Failed to parse follow-up JSON",
        "results_empty": "'results' is empty. At least one question result is required.",
        "qbank_missing": "The English question bank is not built yet. Use /interview/generate.",
    },
}

M = {}

# ---------------- 공통 유틸 ----------------
def clamp_scores(scores):
    out = {}
    for k in SCORE_KEYS:
        try:
            v = int(round(float(scores.get(k, 0))))
        except (TypeError, ValueError):
            v = 0
        out[k] = max(0, min(100, v))
    return out

_STAR_GRADE = {5: "A", 4: "B", 3: "C", 2: "D", 1: "F"}

def _star_overall_grade(avg100):
    o = avg100
    if o >= 90: return "A+"
    if o >= 83: return "A"
    if o >= 80: return "A-"
    if o >= 73: return "B+"
    if o >= 67: return "B"
    if o >= 60: return "B-"
    if o >= 53: return "C+"
    if o >= 47: return "C"
    if o >= 40: return "C-"
    if o >= 20: return "D"
    return "F"

def clamp_star(star):
    """LLM이 낸 star(항목별 1~5) -> 개별 점수(x20)+등급(A~F) + 종합 평균+세분화 등급.
    미평가(0)는 종합에서 제외. 프론트 e.star.S 호환. 반환: (score, grade, overall, overall_grade)."""
    star = star or {}
    keys = {"S": "situation", "T": "task", "A": "action", "R": "result"}
    score, grade, valid = {}, {}, []
    for short, full in keys.items():
        try:
            v = int(round(float(star.get(full, 0))))
        except (TypeError, ValueError):
            v = 0
        if v <= 0:
            score[short], grade[short] = 0, "N/A"
        else:
            v = max(1, min(5, v))
            score[short], grade[short] = v * 20, _STAR_GRADE[v]
            valid.append(v)
    if valid:
        overall = round(sum(valid) / len(valid) * 20)
        og = _star_overall_grade(overall)
    else:
        overall, og = 0, "N/A"
    return score, grade, overall, og

def fix_overall(scores):
    vals = [int(scores.get(k, 0)) for k in SCORE_KEYS]
    if vals[0] == 0:
        rest = vals[1:]
        return round(sum(rest) / len(rest)) if rest else 0
    return round(sum(vals) / len(vals))

def parse_json_lenient(text):
    """</thought> 이후 마지막 JSON을 관대하게 파싱.
    엄격 json.loads -> 트레일링 콤마 보정 -> json_repair(폴백) 순. 한·영 공통."""
    tail = text.split("</thought>")[-1] if "</thought>" in text else text
    tail = re.sub(r"```(?:json)?", "", tail)
    m = re.search(r"\{.*\}", tail, re.DOTALL)
    if not m:
        return None
    raw = m.group(0)
    try:
        return json.loads(raw)                                  # 1) 엄격
    except Exception:
        pass
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", raw))   # 2) 트레일링 콤마 보정
    except Exception:
        pass
    try:
        obj = repair_json(raw, return_objects=True)             # 3) json_repair 폴백(키 따옴표 누락 등)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def vlen(name, val, lang="ko", required=True):
    max_len = LIMITS.get(name)
    if required and (val is None or (isinstance(val, str) and not val.strip())):
        return MSG[lang]["empty"].format(name=name)
    if isinstance(val, str) and max_len and len(val) > max_len:
        return MSG[lang]["too_long"].format(name=name, max=max_len, len=len(val))
    return None

def not_ready(lang="ko"):
    return None if "llm" in M else {"ok": False, "error": MSG[lang]["loading"]}

import threading
GEN_LOCK = threading.Lock()   # GPU 생성 직렬화: 동시 요청이 같은 모델에 동시에 generate -> CUDA 충돌/오염 방지

def run_llm(prompt, prefill, use_adapter=True, max_new_tokens=2048,
            do_sample=False, temperature=0.8, top_p=0.9, reason_budget=None):
    msgs = [{"role": "user", "content": prompt}]
    text = M["llm_tok"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill
    enc = M["llm_tok"](text, return_tensors="pt", add_special_tokens=False).to(M["llm"].device)
    plen = enc["input_ids"].shape[1]

    gkw = {"do_sample": do_sample}
    if do_sample:
        gkw["temperature"] = temperature
        gkw["top_p"] = top_p

    def _gen(inp, attn, n_tok):
        if use_adapter:
            return M["llm"].generate(input_ids=inp, attention_mask=attn, max_new_tokens=n_tok, **gkw)
        with M["llm"].disable_adapter():
            return M["llm"].generate(input_ids=inp, attention_mask=attn, max_new_tokens=n_tok, **gkw)

    forced = False
    with GEN_LOCK:                          # 한 번에 하나의 generate만 GPU에서 실행 (직렬화)
        t0 = time.time()
        with torch.no_grad():
            if not BUDGET_FORCE:
                o = _gen(enc["input_ids"], enc["attention_mask"], max_new_tokens)
            else:
                _rb_src = reason_budget if reason_budget is not None else REASON_BUDGET
                rb = min(_rb_src, max_new_tokens)
                o1 = _gen(enc["input_ids"], enc["attention_mask"], rb)
                n1 = int(o1.shape[1] - plen)
                if n1 < rb:
                    o = o1                                  # 자연 종료(추론+답변 완료)
                else:
                    gen1 = M["llm_tok"].decode(o1[0][plen:], skip_special_tokens=True)
                    if "</thought>" in gen1:
                        seq = o1
                        rem = max(0, max_new_tokens - rb)   # 남은 원래 예산으로 답변 마저
                    else:
                        close = M["llm_tok"]("\n</thought>\n", return_tensors="pt",
                                             add_special_tokens=False).input_ids.to(M["llm"].device)
                        seq = torch.cat([o1, close], dim=1)
                        rem = ANSWER_BUDGET                 # 추론 강제 종료 -> 답변 생성
                        forced = True
                    o = _gen(seq, torch.ones_like(seq), rem) if rem > 0 else seq
        dt = time.time() - t0
    gen_len = int(o.shape[1] - plen)
    hit = " [CAP]" if gen_len >= max_new_tokens else ""
    fmark = " [FORCED]" if forced else ""
    print(f">>> [gen] {gen_len}tok / {dt:.1f}s = {gen_len/max(dt,1e-9):.1f} tok/s "
          f"(prompt {plen}, cap {max_new_tokens}{hit}{fmark}, adapter={use_adapter}, sample={do_sample})", flush=True)
    return M["llm_tok"].decode(o[0][plen:], skip_special_tokens=False)

# ---------------- 프롬프트 빌더 (언어별) ----------------
def eval_prompt(lang, question, answer):
    if lang == "en":
        return f"{RUBRIC_EN}\n\n[Interview Question]\n{question}\n\n[Candidate Answer]\n{answer}"
    return f"{M['rubric']}\n\n[면접 질문]\n{question}\n\n[지원자 답변]\n{answer}"

def gen_prompt(lang, intro, n, resume, job_posting):
    if lang == "en":
        return f"""{intro} Based on the candidate's resume and the job posting below, create {n} interview questions you would actually ask this candidate.

Rules:
- Tailored questions connecting the resume's experience/skills with the posting's requirements
- Mix technical questions with experience/behavioral questions appropriately
- Each question one clear sentence, in English
- After your thinking, output ONLY this JSON: {{"questions": ["q1", "q2", "..."]}}

[Resume]
{resume}

[Job Posting]
{job_posting}"""
    return f"""{intro} 아래 지원자의 이력서와 채용공고를 바탕으로, 이 지원자에게 실제로 물어볼 면접 질문 {n}개를 만드세요.

규칙:
- 이력서의 경험·기술과 공고의 요구사항을 연결한 맞춤형 질문일 것
- 기술 질문과 경험·인성 질문을 적절히 섞을 것
- 각 질문은 한 문장으로 명확하게, 한국어로 작성
- 사고 과정을 마친 뒤, 마지막에 JSON만 출력: {{"questions": ["질문1", "질문2", "..."]}}

[이력서]
{resume}

[채용공고]
{job_posting}"""

def fu_prompt(lang, intro, question, answer, history=None):
    hist_ko = hist_en = ""
    if history:
        try:
            ls = []
            for h in history:
                q = (h.get("question") or "").strip()
                a = (h.get("answer") or "").strip()
                if q or a:
                    ls.append("- Q: " + q + "\n  A: " + a)
            if ls:
                body = "\n".join(ls)
                hist_ko = "[\uc774\uc804 \ub300\ud654 \ud750\ub984] (\uc774\ubbf8 \ub2e4\ub8ec \ub0b4\uc6a9\uc740 \ubc18\ubcf5\ud558\uc9c0 \ub9d0\uace0 \uc774\uc5b4\uc11c \uc2ec\ud654\ud560 \uac83)\n" + body + "\n\n"
                hist_en = "[Prior conversation] (do not repeat what was covered; build on it)\n" + body + "\n\n"
        except Exception:
            hist_ko = hist_en = ""
    if lang == "en":
        return f"""{intro} Below are an interview question and the candidate's answer. Analyze the answer, then create ONE natural follow-up question.

[Step 1] Diagnose the answer silently: length (sufficient or too short), specificity (concrete examples/numbers/tech vs vague), basis (reasons given or not), depth (surface vs deep).
[Step 2] Adapt the follow-up to the diagnosis:
- If the answer is short, vague, or lacks basis -> directly ask for a concrete example, situation, number, or the reason behind the choice.
- If the answer is specific and solid -> probe one level deeper: the reason for the choice, comparison with alternatives, trade-offs, or edge cases.

Rules:
- Ground it strictly in what the candidate actually said (do not invent content)
- Keep technical terms (Redux, React Query, JPA, etc.) in their original form; do not transliterate
- One clear sentence, in English
- After your thinking, output ONLY this JSON: {{"followup": "follow-up question"}}

{hist_en}[Interview Question]
{question}
[Candidate Answer]
{answer}"""
    return f"""{intro} \uc544\ub798\ub294 \uba74\uc811 \uc9c8\ubb38\uacfc \uc9c0\uc6d0\uc790\uc758 \ub2f5\ubcc0\uc785\ub2c8\ub2e4. \uc9c0\uc6d0\uc790\uc758 \ub2f5\ubcc0\uc744 \ubd84\uc11d\ud55c \ub4a4 \uc790\uc5f0\uc2a4\ub7ec\uc6b4 \ud6c4\uc18d(\uaf2c\ub9ac) \uc9c8\ubb38 1\uac1c\ub97c \ub9cc\ub4dc\uc138\uc694.

[1\ub2e8\uacc4] \ub2f5\ubcc0\uc744 \uc18d\uc73c\ub85c \uc9c4\ub2e8\ud558\uc138\uc694: \uae38\uc774(\ucda9\ubd84/\ub108\ubb34 \uc9e7\uc74c), \uad6c\uccb4\uc131(\uad6c\uccb4\uc801 \uc0ac\ub840\u00b7\uc218\uce58\u00b7\uae30\uc220 vs \ub450\ub8e8\ubb49\uc2e4), \uadfc\uac70(\uc774\uc720 \uc81c\uc2dc \uc5ec\ubd80), \uae4a\uc774(\ud45c\uba74\uc801 vs \uae4a\uc74c).
[2\ub2e8\uacc4] \uc9c4\ub2e8\uc5d0 \ub530\ub77c \uaf2c\ub9ac\uc9c8\ubb38\uc744 \ub2e4\ub974\uac8c \ub9cc\ub4dc\uc138\uc694:
- \ub2f5\ubcc0\uc774 \uc9e7\uac70\ub098 \ub450\ub8e8\ubb49\uc2e4\ud558\uac70\ub098 \uadfc\uac70\uac00 \ubd80\uc871\ud558\uba74 -> \uad6c\uccb4\uc801\uc778 \uc0ac\ub840\u00b7\uc0c1\ud669\u00b7\uc218\uce58, \ub610\ub294 \uadf8\ub807\uac8c \ud55c \uc774\uc720\ub97c \uc9c1\uc811 \uc694\uad6c\ud558\uc138\uc694. (\uc608: \ubc29\uae08 '\uc801\ub2f9\ud788 \ud588\ub2e4'\uace0 \ud558\uc168\ub294\ub370, \uc5b4\ub5a4 \uae30\uc900\uc73c\ub85c \uacb0\uc815\ud558\uc168\uace0 \uc2e4\uc81c \uc608\ub97c \ub4e4\uc5b4\uc8fc\uc2e4 \uc218 \uc788\ub098\uc694?)
- \ub2f5\ubcc0\uc774 \uad6c\uccb4\uc801\uc774\uace0 \ucda9\uc2e4\ud558\uba74 -> \ud55c \ub2e8\uacc4 \ub354 \uae4a\uc774 \ud30c\uace0\ub4dc\uc138\uc694: \uc120\ud0dd\ud55c \uc774\uc720, \ub2e4\ub978 \ubc29\ubc95\uacfc\uc758 \ube44\uad50, \ud2b8\ub808\uc774\ub4dc\uc624\ud504, \ud55c\uacc4 \uc0c1\ud669.

\uaddc\uce59:
- \ubc18\ub4dc\uc2dc \uc9c0\uc6d0\uc790\uac00 \uc2e4\uc81c\ub85c \ud55c \ub9d0\uc5d0 \uadfc\uac70\ud560 \uac83 (\uc5c6\ub294 \ub0b4\uc6a9\uc744 \uc9c0\uc5b4\ub0b4\uc9c0 \ub9d0 \uac83)
- \uae30\uc220 \uc6a9\uc5b4(Redux, React Query, JPA \ub4f1)\ub294 \uc6d0\ubb38(\uc601\ubb38) \uadf8\ub300\ub85c \ud45c\uae30\ud560 \uac83 (\uc74c\ucc28 \ubcc0\ud658 \uae08\uc9c0)
- \ud55c \ubb38\uc7a5\uc73c\ub85c \uba85\ud655\ud558\uac8c, \ud55c\uad6d\uc5b4\ub85c \uc791\uc131
- \uc0ac\uace0 \uacfc\uc815\uc744 \ub9c8\uce5c \ub4a4, \ub9c8\uc9c0\ub9c9\uc5d0 JSON\ub9cc \ucd9c\ub825: {{"followup": "\uaf2c\ub9ac\uc9c8\ubb38"}}

{hist_ko}[\uba74\uc811 \uc9c8\ubb38]
{question}
[\uc9c0\uc6d0\uc790 \ub2f5\ubcc0]
{answer}"""
def report_lines(lang, results):
    lines = []
    for i, r in enumerate(results, 1):
        ev = r.get("evaluation") or {}
        q = r.get("question", "")
        sc = clamp_scores(ev.get("scores", {})) if ev.get("scores") else {}
        fb = ev.get("feedback", "")
        if sc.get("technical_accuracy", 0) == 0:
            sc.pop("technical_accuracy", None)
        lab = LABEL[lang]
        loc_sc = {lab.get(k, k): v for k, v in sc.items() if k in SCORE_KEYS}
        if lang == "en":
            lines.append(f"[Q{i}] Question: {q}\nScores: {loc_sc}\nFeedback: {fb}")
        else:
            lines.append(f"[문항 {i}] 질문: {q}\n점수: {loc_sc}\n피드백: {fb}")
    return "\n\n".join(lines)

def report_prompt(lang, joined):
    if lang == "en":
        return f"""You are an interviewer summarizing the results of an IT-job mock interview. Below are one candidate's full results (per-question scores and feedback). Synthesize them into a report.

Rules:
- Summarize overall strengths, areas to improve, and a preparation guide for passing
- Do not mention evaluation axes that are not in the scores
- Be concrete and actionable, in English
- After your thinking, output ONLY this JSON: {{"summary": "one-paragraph overview", "strengths": ["strength", "..."], "weaknesses": ["area to improve", "..."], "guide": ["prep guide", "..."]}}

[Full interview results]
{joined}"""
    return f"""당신은 IT 직무 모의면접 결과를 종합하는 면접관입니다. 아래는 한 지원자의 면접 전체 결과(문항별 점수·피드백)입니다. 이를 종합해 리포트를 작성하세요.

규칙:
- 전반적 강점, 보완점, 합격을 위한 준비 가이드를 각각 정리
- 점수에 없는 평가 항목은 언급하지 말 것
- 구체적이고 실행 가능하게, 한국어로 작성
- 사고 과정을 마친 뒤, 마지막에 JSON만 출력: {{"summary": "총평 한 문단", "strengths": ["강점", "..."], "weaknesses": ["보완점", "..."], "guide": ["준비 가이드", "..."]}}

[면접 전체 결과]
{joined}"""

# ---------------- 모델 로딩 ----------------
@asynccontextmanager
async def lifespan(app):
    print(">>> 모델 로딩 시작 (1~2분 소요)...", flush=True)
    M["emb_tok"]   = AutoTokenizer.from_pretrained(EMB_MODEL)
    M["emb_model"] = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE).half().eval()
    M["index"]     = faiss.read_index(os.path.join(RAG_DIR, "ict_questions.index"))
    M["records"]   = json.load(open(os.path.join(RAG_DIR, "ict_questions.json"), encoding="utf-8"))
    print(f">>> RAG(ko) 로드 완료: 질문 {M['index'].ntotal}개", flush=True)
    # --- FAQ RAG 인덱스 로딩 (챗봇용) — 있으면 로딩, 없으면 건너뜀(하위호환) ---
    _faq_idx = os.path.join(RAG_DIR, "faq.index")
    _faq_jsn = os.path.join(RAG_DIR, "faq.json")
    if os.path.exists(_faq_idx) and os.path.exists(_faq_jsn):
        M["faq_index"]   = faiss.read_index(_faq_idx)
        M["faq_records"] = json.load(open(_faq_jsn, encoding="utf-8"))
        print(f">>> FAQ RAG 로드 완료: {M['faq_index'].ntotal}개", flush=True)
    else:
        print(">>> (FAQ 인덱스 없음 — 챗봇은 면접RAG/폴백으로만 동작)", flush=True)
    en_idx  = os.path.join(RAG_DIR, "ict_questions_en.index")
    en_json = os.path.join(RAG_DIR, "ict_questions_en.json")
    if os.path.exists(en_idx) and os.path.exists(en_json):
        M["index_en"]   = faiss.read_index(en_idx)
        M["records_en"] = json.load(open(en_json, encoding="utf-8"))
        print(f">>> RAG(en) 로드 완료: 질문 {M['index_en'].ntotal}개", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    M["llm_tok"] = AutoTokenizer.from_pretrained(LLM_NAME, revision=LLM_REV, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(LLM_NAME, revision=LLM_REV,
        quantization_config=bnb, device_map="auto", trust_remote_code=True).eval()
    M["llm"] = PeftModel.from_pretrained(base, ADAPTER).eval()
    print(f">>> EXAONE + LoRA 로드 완료. GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)
    first = json.loads(open(TRAIN, encoding="utf-8").readline())
    M["rubric"] = first["messages"][0]["content"].split("\n\n[면접 질문]")[0]
    print(">>> ✅ 서버 준비 완료 — 이제 요청을 받을 수 있습니다", flush=True)
    yield
    M.clear()

app = FastAPI(title="Interview AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ===== API 키 인증 (X-API-Key) — 환경변수 API_KEY가 있을 때만 활성 =====
import os as _os
API_KEY = _os.environ.get("API_KEY", "").strip()

@app.middleware("http")
async def _api_key_guard(request, call_next):
    if API_KEY and request.method != "OPTIONS":
        _p = request.url.path
        _exempt = (_p in ("/", "/health", "/openapi.json") or _p.startswith("/docs") or _p.startswith("/redoc"))
        if not _exempt and request.headers.get("X-API-Key") != API_KEY:
            return JSONResponse(status_code=401, content={"ok": False, "error": "인증 실패: X-API-Key 헤더가 없거나 올바르지 않습니다. / Unauthorized."})
    return await call_next(request)

# ---------------- 미들웨어 / 에러 핸들러 ----------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        print(f">>> [{request.method}] {request.url.path} -> {status} ({time.time()-t0:.1f}s)", flush=True)

@app.exception_handler(Exception)
async def on_exception(request: Request, exc: Exception):
    print(f">>> [ERROR] {request.url.path}: {type(exc).__name__}: {exc}", flush=True)
    return JSONResponse(status_code=500, content={"ok": False, "error": f"{type(exc).__name__}: {exc}"})

@app.exception_handler(RequestValidationError)
async def on_validation(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"ok": False, "error": "요청 형식이 올바르지 않습니다. / Invalid request format."})

def embed_query(text, max_len=256):
    enc = M["emb_tok"]([text], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        v = M["emb_model"](**enc).last_hidden_state[:, 0]
        v = F.normalize(v, p=2, dim=1)
    return v.float().cpu().numpy().astype("float32")

# ---------------- 요청 스키마 ----------------
class ChatReq(BaseModel):
    message: str
    lang: str = "ko"

class QuestionReq(BaseModel):
    topic: str
    k: int = 3
    lang: str = "ko"

class EvaluateReq(BaseModel):
    question: str
    answer: str
    lang: str = "ko"

class GenerateReq(BaseModel):
    resume: str
    job_posting: str
    n: int = 5
    persona: str = "default"
    lang: str = "ko"

class FollowupReq(BaseModel):
    question: str
    answer: str
    persona: str = "default"
    lang: str = "ko"
    history: list = []
class ReportReq(BaseModel):
    results: list
    lang: str = "ko"
    voice: dict = None
    expression: dict = None

# ---------------- 엔드포인트 ----------------
@app.get("/health")
def health():
    ready = "llm" in M
    info = {"status": "ok", "ready": ready, "languages": list(LANGS)}
    if ready:
        info["rag_questions"] = M["index"].ntotal
        info["en_question_bank"] = "index_en" in M
        info["adapter_loaded"] = isinstance(M["llm"], PeftModel)
        if torch.cuda.is_available():
            info["gpu_memory_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
    return info

# ---- 데모 페이지(single HTML) same-origin 서빙 ----
# 공개URL/ 로 접속하면 데모와 API가 동일 출처(same-origin)라 CORS 문제 없음.
# server.py 위치를 기준으로 demo/index.html을 여러 후보 경로에서 탐색(레포/플랫 배포 모두 호환).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEMO_CANDIDATES = [
    os.path.join(_HERE, "demo", "index.html"),         # server.py 와 같은 폴더의 demo/
    os.path.join(_HERE, "..", "demo", "index.html"),   # 레포 루트의 demo/ (server/ 의 상위)
    os.path.join(_HERE, "index.html"),                 # server.py 옆에 평탄화된 경우
]

def _demo_path():
    for p in _DEMO_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None

@app.get("/")
def demo_index():
    p = _demo_path()
    if p:
        return FileResponse(p, media_type="text/html; charset=utf-8")
    return JSONResponse(status_code=404, content={"ok": False,
        "error": "demo/index.html을 찾을 수 없습니다. server.py 와 같은 위치(또는 상위)에 demo/index.html을 두세요."})

@app.get("/interview/personas")
def list_personas(lang: str = "ko"):
    lang = norm_lang(lang)
    return {"ok": True, "personas": [{"key": k, "description": v} for k, v in PERSONAS[lang].items()]}

@app.post("/interview/question")
def get_question(req: QuestionReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr: return nr
    err = vlen("topic", req.topic, lang)
    if err: return {"ok": False, "error": err}
    if lang == "en":
        if "index_en" not in M:
            return {"ok": False, "error": MSG["en"]["qbank_missing"]}
        index, records = M["index_en"], M["records_en"]
    else:
        index, records = M["index"], M["records"]
    k = max(1, min(10, req.k))
    scores, ids = index.search(embed_query(req.topic), k)
    out = [{"question": records[i]["question"], "score": float(s)}
           for s, i in zip(scores[0], ids[0])]
    return {"ok": True, "topic": req.topic, "questions": out}

@app.post("/interview/evaluate")
def evaluate(req: EvaluateReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr: return nr
    err = vlen("question", req.question, lang) or vlen("answer", req.answer, lang)
    if err: return {"ok": False, "error": err}
    prompt = eval_prompt(lang, req.question, req.answer)
    # [어댑터 미사용] QLoRA 어댑터가 5축을 0~100 대신 1~5로 왜곡 + tok/s 8.2로 저하 →
    # base 모델(EXAONE)로 채점(척도 정상 0~100 + ~50 tok/s). 어댑터 원인규명·재학습은 발표 후 과제(DEVLOG 참조).
    use_adapter = False
    gen = run_llm(prompt, PREFILL[lang], use_adapter=use_adapter, max_new_tokens=2048)
    ev = parse_json_lenient(gen)
    if ev and isinstance(ev.get("scores"), dict):
        ev["scores"] = clamp_scores(ev["scores"])
        ev["overall"] = fix_overall(ev["scores"])
        ev["display_scores"] = to_question_scores(ev["scores"])
        _ss, _sg, _so, _sog = clamp_star(ev.get("star"))
        ev["star"] = _ss
        ev["star_grade"] = _sg
        ev["star_overall"] = _so
        ev["star_overall_grade"] = _sog
        return {"ok": True, "evaluation": ev}
    return {"ok": False, "error": MSG[lang]["eval_fail"], "raw": gen[-1500:]}


# ---- 스트리밍 채점 (SSE): 출력은 비스트리밍과 동일, 토큰만 실시간으로 흘려보냄 ----
def _eval_stream_gen(prompt, prefill, use_adapter, lang):
    from transformers import TextIteratorStreamer
    msgs = [{"role": "user", "content": prompt}]
    text = M["llm_tok"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill
    enc = M["llm_tok"](text, return_tensors="pt", add_special_tokens=False).to(M["llm"].device)
    plen = enc["input_ids"].shape[1]
    streamer = TextIteratorStreamer(M["llm_tok"], skip_prompt=True, skip_special_tokens=False)
    holder = {}
    def _gen():
        try:
            with torch.no_grad():
                if use_adapter:
                    holder["out"] = M["llm"].generate(**enc, max_new_tokens=2048, do_sample=False, streamer=streamer)
                else:
                    with M["llm"].disable_adapter():
                        holder["out"] = M["llm"].generate(**enc, max_new_tokens=2048, do_sample=False, streamer=streamer)
        except Exception as e:
            holder["err"] = e
    GEN_LOCK.acquire()                      # 비스트리밍과 같은 락으로 직렬화
    t = threading.Thread(target=_gen, daemon=True)
    t0 = time.time()
    t.start()
    full = []
    try:
        for chunk in streamer:
            if not chunk:
                continue
            full.append(chunk)
            piece = chunk
            for _tok in ("[|endofturn|]", "[|assistant|]", "[|system|]", "[|user|]", "[|endoftext|]"):
                piece = piece.replace(_tok, "")
            if piece:
                yield "data: " + json.dumps({"type": "token", "text": piece}, ensure_ascii=False) + "\n\n"
        t.join()
        if "err" in holder:
            raise holder["err"]
        gen_text = "".join(full)
        out = holder.get("out")
        if out is not None:
            gl = int(out.shape[1] - plen); dt = time.time() - t0
            print(f">>> [gen-stream] {gl}tok / {dt:.1f}s = {gl/max(dt,1e-9):.1f} tok/s (cap 2048, adapter={use_adapter})", flush=True)
        ev = parse_json_lenient(gen_text)
        if ev and isinstance(ev.get("scores"), dict):
            ev["scores"] = clamp_scores(ev["scores"])
            ev["overall"] = fix_overall(ev["scores"])
            ev["display_scores"] = to_question_scores(ev["scores"])
            _ss, _sg, _so, _sog = clamp_star(ev.get("star"))
            ev["star"] = _ss
            ev["star_grade"] = _sg
            ev["star_overall"] = _so
            ev["star_overall_grade"] = _sog
            payload = {"type": "done", "ok": True, "evaluation": ev}
        else:
            payload = {"type": "done", "ok": False, "error": MSG[lang]["eval_fail"]}
        yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
    except Exception as e:
        yield "data: " + json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False) + "\n\n"
    finally:
        if t.is_alive():
            t.join()                        # 끊겨도 generate 끝까지 기다린 뒤 락 해제 (다음 요청과 GPU 충돌 방지)
        try:
            GEN_LOCK.release()
        except RuntimeError:
            pass

@app.post("/interview/evaluate/stream")
def evaluate_stream(req: EvaluateReq):
    lang = norm_lang(req.lang)
    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    def _one(obj):
        def _g():
            yield "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
        return _g()
    nr = not_ready(lang)
    if nr:
        return StreamingResponse(_one({"type": "error", "error": nr["error"]}), media_type="text/event-stream", headers=sse_headers)
    err = vlen("question", req.question, lang) or vlen("answer", req.answer, lang)
    if err:
        return StreamingResponse(_one({"type": "error", "error": err}), media_type="text/event-stream", headers=sse_headers)
    prompt = eval_prompt(lang, req.question, req.answer)
    use_adapter = False  # [어댑터 미사용] 위 evaluate와 동일 이유(척도 왜곡+속도) — base 사용
    return StreamingResponse(_eval_stream_gen(prompt, PREFILL[lang], use_adapter, lang), media_type="text/event-stream", headers=sse_headers)

@app.post("/interview/generate")
def generate_questions(req: GenerateReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr: return nr
    err = vlen("resume", req.resume, lang) or vlen("job_posting", req.job_posting, lang)
    if err: return {"ok": False, "error": err}
    n = max(1, min(15, req.n))
    intro = PERSONAS[lang].get(req.persona, PERSONAS[lang]["default"])
    prompt = gen_prompt(lang, intro, n, req.resume, req.job_posting)
    gen = run_llm(prompt, GEN_PREFILL[lang], use_adapter=False, max_new_tokens=2048, do_sample=True, reason_budget=350)
    d = parse_json_lenient(gen)
    if d and isinstance(d.get("questions"), list):
        return {"ok": True, "questions": d["questions"]}
    return {"ok": False, "error": MSG[lang]["gen_fail"], "raw": gen[-1500:]}

@app.post("/interview/followup")
def followup(req: FollowupReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr: return nr
    err = vlen("question", req.question, lang) or vlen("answer", req.answer, lang)
    if err: return {"ok": False, "error": err}
    intro = PERSONAS[lang].get(req.persona, PERSONAS[lang]["default"])
    prompt = fu_prompt(lang, intro, req.question, req.answer, req.history)
    gen = run_llm(prompt, FU_PREFILL[lang], use_adapter=False, max_new_tokens=1536, do_sample=True, reason_budget=350)
    d = parse_json_lenient(gen)
    if d and "followup" in d:
        return {"ok": True, "followup": d.get("followup", "")}
    return {"ok": False, "error": MSG[lang]["fu_fail"], "raw": gen[-1500:]}

@app.post("/interview/report")
def report(req: ReportReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr: return nr
    if not isinstance(req.results, list) or len(req.results) == 0:
        return {"ok": False, "error": MSG[lang]["results_empty"]}
    sums = {k: 0 for k in SCORE_KEYS}
    cnts = {k: 0 for k in SCORE_KEYS}
    for r in req.results:
        sc = (r.get("evaluation") or {}).get("scores") or {}
        if not sc:
            continue
        sc = clamp_scores(sc)
        for k in SCORE_KEYS:
            v = sc[k]
            if k == "technical_accuracy" and v == 0:
                continue
            sums[k] += v
            cnts[k] += 1
    axis_avg = {k: (round(sums[k] / cnts[k]) if cnts[k] else 0) for k in SCORE_KEYS}
    overall = fix_overall(axis_avg)
    joined = report_lines(lang, req.results)
    prompt = report_prompt(lang, joined)
    gen = run_llm(prompt, RP_PREFILL[lang], use_adapter=False, max_new_tokens=2048)
    body = parse_json_lenient(gen) or {}
    _expr_in = {k: v for k, v in (req.expression or {}).items() if k != "overall" and isinstance(v, (int, float))}
    _voice_in = {"clarity_score": req.voice.get("delivery_score")} if isinstance(req.voice, dict) and isinstance(req.voice.get("delivery_score"), (int, float)) else None
    fe = to_report_scores([(r.get("evaluation") or {}).get("scores") or {} for r in req.results],
                          voice=_voice_in, expression=(_expr_in or None))
    return {"ok": bool(body), "overall": overall, "axis_averages": axis_avg,
            "categories": fe["categories"], "grade": fe["grade"], "overall_categories": fe["overall"],
            "report": body, "raw": (None if body else gen[-1500:])}

# ===== STT (음성 -> 텍스트) =====
# ===== 표정 분석 점수 수신 (브라우저 face-api.js → 백엔드, 점수만) =====
EXP_KEYS = ["confidence", "composure", "attention", "expressiveness"]
EXP_LABEL = {
    "ko": {"good": "안정적", "mid": "보통", "low": "개선 필요"},
    "en": {"good": "Stable", "mid": "Moderate", "low": "Needs work"},
}

def _exp_clamp(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0
    return int(round(max(0, min(100, v))))

class ExpressionReq(BaseModel):
    scores: dict = {}
    lang: str = "ko"

@app.post("/interview/expression")
def interview_expression(req: ExpressionReq):
    lang = norm_lang(req.lang)
    src = req.scores if isinstance(req.scores, dict) else {}
    norm = {k: _exp_clamp(src.get(k)) for k in EXP_KEYS}
    # 클라이언트 overall은 신뢰하지 않고 재계산 (자신감·안정·주의 0.3 + 표현력 0.1)
    norm["overall"] = _exp_clamp(0.30 * norm["confidence"] + 0.30 * norm["composure"]
                                 + 0.30 * norm["attention"] + 0.10 * norm["expressiveness"])
    tier = "good" if norm["overall"] >= 75 else ("mid" if norm["overall"] >= 50 else "low")
    note = ("표정 점수는 전달력 참고용 보조 지표이며 합격 예측이 아닙니다." if lang == "ko"
            else "Expression scores are a supplementary delivery indicator, not a pass/fail prediction.")
    return {"ok": True, "lang": lang, "expression": norm, "label": EXP_LABEL[lang][tier], "note": note}


# ===== 학습용 객관식 퀴즈 생성 (EXAONE, Claude API 대체) =====
import random as _random
QUIZ_PREFILL = {
    "ko": "먼저 주제의 핵심 개념을 정리하고, 정답이 하나로 분명한 짧은 4지선다 문제로 어떻게 낼지 생각하겠습니다. ",
    "en": "First, let me organize the key concepts and think about clear single-answer multiple-choice questions with short options. ",
}

class QuizReq(BaseModel):
    topic: str
    n: int = 5
    difficulty: str = "중"   # 하 / 중 / 상
    lang: str = "ko"

def quiz_prompt(lang, topic, n, difficulty):
    if lang == "en":
        return f"""You write multiple-choice (4-option) quiz items for web-development study. Create {n} questions on the topic below at '{difficulty}' difficulty.

Rules:
- Each item is ONE short factual question with a single unambiguous answer (concept, definition, behavior, difference). Never write open-ended "explain/describe" prompts.
- Stay strictly within the given topic (no unrelated content).
- All 4 options are SHORT (a phrase, about 6 words). Keep the four options similar in length INCLUDING the correct one; the correct option must not be the longest or most detailed.
- The 3 distractors are plausible but clearly wrong and mutually exclusive.
- Exactly one correct answer. Explanation: 1-2 sentences.
- After thinking, output ONLY this JSON (answer as TEXT, not an index):
{{"items":[{{"question":"...","correct":"...","distractors":["...","...","..."],"explanation":"..."}}]}}

Example:
{{"items":[{{"question":"What does an async function return?","correct":"A Promise","distractors":["A callback","undefined","The value immediately"],"explanation":"An async function always returns a Promise wrapping its return value."}}]}}

[Topic]
{topic}"""
    return f"""당신은 웹 개발 학습용 객관식(4지선다) 문제를 출제합니다. 아래 주제로 난이도 '{difficulty}' 문제 {n}개를 만드세요.

규칙:
- 각 문제는 정답이 하나로 분명한 '한 문장짜리 사실 확인형'(개념·정의·동작·차이). "설명해주세요/서술하시오/방법을 쓰시오" 같은 서술형은 절대 금지.
- 반드시 주어진 주제 범위 안에서만 출제(주제와 무관한 내용 금지).
- 보기 4개는 모두 짧게(명사구나 한 구절, 대체로 20자 내외). 정답을 포함해 4개의 길이를 비슷하게 맞추고, 정답이 가장 길거나 가장 자세한 보기가 되지 않게.
- 오답 3개는 그럴듯하지만 분명히 틀리고 서로 겹치지 않게.
- 정답은 정확히 하나. 해설은 1~2문장.
- 사고를 마친 뒤, 마지막에 JSON만 출력(정답은 인덱스가 아니라 '텍스트'):
{{"items":[{{"question":"...","correct":"...","distractors":["...","...","..."],"explanation":"..."}}]}}

예시:
{{"items":[{{"question":"async 함수가 반환하는 것은?","correct":"Promise","distractors":["콜백 함수","undefined","즉시 실행 결과"],"explanation":"async 함수는 항상 Promise를 반환하며, 반환값은 그 Promise로 감싸집니다."}}]}}

[주제]
{topic}"""


def _build_quiz_items(parsed, n):
    """모델 출력(정답=텍스트)을 받아 보기 셔플 + 정답 인덱스 계산. 프론트 스키마로 반환."""
    items = parsed.get("items") if isinstance(parsed, dict) else None
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        q = str(it.get("question") or "").strip()
        correct = str(it.get("correct") or "").strip()
        expl = str(it.get("explanation") or "").strip()
        dis = [str(d).strip() for d in (it.get("distractors") or []) if str(d).strip()]
        dis = [d for d in dis if d != correct]
        seen, uniq = set(), []
        for d in dis:
            if d not in seen:
                seen.add(d); uniq.append(d)
        if not q or not correct or len(uniq) < 3:
            continue
        opts = [correct] + uniq[:3]
        _random.shuffle(opts)
        out.append({"q": q, "options": opts, "answer": opts.index(correct), "explanation": expl})
        if len(out) >= n:
            break
    return out

@app.post("/education/quiz")
def education_quiz(req: QuizReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr:
        return nr
    err = vlen("topic", req.topic, lang)
    if err:
        return {"ok": False, "error": err}
    n = max(1, min(10, req.n))
    prompt = quiz_prompt(lang, req.topic, n, req.difficulty)
    gen = run_llm(prompt, QUIZ_PREFILL[lang], use_adapter=False, max_new_tokens=3072, do_sample=True)
    parsed = parse_json_lenient(gen)
    items = _build_quiz_items(parsed, n)
    if items:
        return {"ok": True, "topic": req.topic, "count": len(items), "quiz": items}
    return {"ok": False, "error": MSG[lang]["gen_fail"], "raw": gen[-1500:]}


# ===== 이력서 자동화 (자기소개서 생성 + 항목 다듬기) — base EXAONE, Claude API 대체 =====

CL_PREFILL = {
    "ko": "먼저 제공된 정보의 핵심을 파악하고, 자연스러운 자기소개서 흐름을 어떻게 구성할지 생각하겠습니다. ",
    "en": "First, let me organize the candidate's provided info (role, experience, skills, projects) and plan how to weave their strengths into the cover letter. ",
}
POLISH_PREFILL = {
    "ko": "먼저 원문이 말하려는 핵심 성과를 파악하고, 사실은 그대로 둔 채 더 명확하고 임팩트 있게 다듬을 방법을 생각하겠습니다. ",
    "en": "First, let me identify the core achievement in the original text and plan how to make it clearer and more impactful without changing the facts. ",
}
_LEN_GUIDE = {"단": "약 300자", "중": "약 550자", "장": "약 800자",
              "short": "about 150 words", "medium": "about 280 words", "long": "about 420 words"}

def _strip_thought(text):
    """추론 모델 출력에서 </thought> 이후 본문만 추출(없으면 전체)."""
    if not text:
        return ""
    t = text
    if "</thought>" in t:
        t = t.rsplit("</thought>", 1)[-1]
    for _tok in ("[|endofturn|]", "[|assistant|]", "[|system|]", "[|user|]", "[|endoftext|]"):
        t = t.replace(_tok, "")
    lines = [ln.rstrip() for ln in t.strip().splitlines()]
    def _is_meta(s):
        s = s.strip()
        return (s == "" or set(s) <= set("-—=*_ ")
                or (s.startswith("(") and s.endswith(")"))
                or s in ("자기소개서", "자소서", "Cover Letter", "커버레터"))
    while lines and _is_meta(lines[0]):
        lines.pop(0)
    while lines and _is_meta(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


class CoverLetterReq(BaseModel):
    role: str = ""        # 지원 직무
    company: str = ""     # 지원 회사
    applicant: str = ""   # 지원자 이름
    experience: str = ""  # 경력(자유 텍스트/요약)
    skills: str = ""      # 보유 스킬
    education: str = ""   # 학력
    projects: str = ""    # 프로젝트
    focus: str = ""       # 강조점/지원 동기 키워드
    tone: str = ""        # 톤(예: 정중하고 진솔하게)
    length: str = "중"    # 단/중/장
    lang: str = "ko"

def cover_letter_prompt(req, lang):
    given = []
    def add(ko, en, val):
        if val and val.strip():
            given.append(f"- {(ko if lang=='ko' else en)}: {val.strip()}")
    add("지원자", "Applicant", req.applicant)
    add("지원 직무", "Target role", req.role)
    add("지원 회사", "Company", req.company)
    add("경력", "Experience", req.experience)
    add("스킬", "Skills", req.skills)
    add("학력", "Education", req.education)
    add("프로젝트", "Projects", req.projects)
    add("강조점/동기", "Focus/Motivation", req.focus)
    given_block = "\n".join(given) if given else ("(제공된 정보 없음)" if lang == "ko" else "(no info provided)")
    tone = (req.tone.strip() or ("정중하고 진솔하게" if lang == "ko" else "professional and sincere"))
    length = _LEN_GUIDE.get(req.length, "약 550자" if lang == "ko" else "about 280 words")
    if lang == "en":
        return f"""You write a job-application cover letter (self-introduction) in English, based ONLY on the information provided.

Rules:
- Use ONLY the given facts. Do NOT invent companies, certifications, or experience that were not provided.
- MOST IMPORTANT: use quantitative figures (percentages, multipliers, amounts, durations) ONLY if present in [Provided information]. Otherwise write qualitatively without numbers; never invent figures like "20% reduction" or "3x growth".
- If info is sparse, write sincere general sentences without fabricating specifics.
- Tone: {tone}. Length: {length}. Natural paragraphs (no bullet lists, no headings).
- Output ONLY the cover letter text. Do NOT output a length note, separators (---), parenthetical remarks, preamble, or JSON.

[Provided information]
{given_block}"""
    return f"""당신은 채용 지원용 자기소개서를 작성합니다. 아래 '제공된 정보'만 근거로 씁니다.

규칙:
- 제공된 사실만 사용하세요. 주어지지 않은 회사명·자격증·경력을 지어내지 마세요.
- 가장 중요: 퍼센트·배수·금액·기간 등 정량 수치는 [제공된 정보]에 적힌 것만 쓰세요. 없으면 숫자 없이 '응답 속도를 개선'처럼 정성적으로 쓰고, '20% 절감'·'3배 증가' 같은 임의 수치를 절대 만들지 마세요.
- 정보가 적으면 사실을 날조하지 말고 진솔한 일반 문장으로 채우세요.
- 톤: {tone}. 분량: {length}. 자연스러운 문단(불릿/제목 없이).
- 자기소개서 본문만 출력하세요. 분량 표기('(약 550자)'), 구분선('---'), 괄호 주석, 머리말, JSON을 출력하지 마세요.

[제공된 정보]
{given_block}"""

@app.post("/resume/cover-letter")
def resume_cover_letter(req: CoverLetterReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr:
        return nr
    prompt = cover_letter_prompt(req, lang)
    gen = run_llm(prompt, CL_PREFILL[lang], use_adapter=False, max_new_tokens=2560, do_sample=True)
    text = _strip_thought(gen)
    if not text:
        return {"ok": False, "error": MSG[lang]["gen_fail"], "raw": gen[-1500:]}
    return {"ok": True, "lang": lang, "cover_letter": text}


class PolishReq(BaseModel):
    text: str           # 다듬을 이력서 항목/문장 (필수)
    role: str = ""      # 지원 직무(선택)
    style: str = ""     # 스타일(예: 간결하고 성과 중심)
    lang: str = "ko"

def polish_prompt(req, lang):
    role = req.role.strip()
    style = (req.style.strip() or ("간결하고 성과 중심으로" if lang == "ko" else "concise and achievement-focused"))
    if lang == "en":
        role_line = (f"\n- Target role: {role}" if role else "")
        return f"""You refine a resume bullet/sentence to be clearer and more impactful.

Rules:
- Keep the facts EXACTLY. Do NOT add achievements, numbers, skills, results, or effects not in the original (rephrase only).
- Style: {style}. Use strong verbs and the concrete outcomes already present.
- Output ONLY the refined text (1-3 lines). No parenthetical notes, separators, preamble, or JSON.{role_line}

[Original]
{req.text.strip()}"""
    role_line = (f"\n- 지원 직무: {role}" if role else "")
    return f"""당신은 이력서 항목(불릿/문장)을 더 명확하고 임팩트 있게 다듬습니다.

규칙:
- 사실은 그대로 유지하세요. 원문에 없는 성과·수치·스킬·결과·효과를 절대 추가하지 마세요(표현만 다듬기).
- 스타일: {style}. 원문에 있는 동작·성과를 강한 동사와 구체적 표현으로.
- 다듬은 문장만 출력하세요. 괄호 주석('(사실 유지...)'), 구분선, 머리말, JSON 없이 1~3줄만.{role_line}

[원문]
{req.text.strip()}"""

@app.post("/resume/polish")
def resume_polish(req: PolishReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr:
        return nr
    if not (req.text or "").strip():
        return {"ok": False, "error": ("다듬을 텍스트를 입력하세요." if lang == "ko" else "Provide text to polish.")}
    prompt = polish_prompt(req, lang)
    gen = run_llm(prompt, POLISH_PREFILL[lang], use_adapter=False, max_new_tokens=1024, do_sample=True)
    text = _strip_thought(gen)
    if not text:
        return {"ok": False, "error": MSG[lang]["gen_fail"], "raw": gen[-1500:]}
    return {"ok": True, "lang": lang, "original": req.text.strip(), "polished": text}


# ===== 공고 자동화 (A: 공고 생성 + B: 공고 분석) — base EXAONE, Claude API 대체 =====

POSTING_GEN_PREFILL = {
    "ko": "먼저 직무와 요구 기술을 정리하고, 채용공고의 주요 업무·자격요건·우대사항을 어떻게 구성할지 생각하겠습니다. ",
    "en": "First, let me organize the role and required skills, then plan the responsibilities, requirements, and preferred qualifications for the posting. ",
}
POSTING_ANALYZE_PREFILL = {
    "ko": "먼저 채용공고 원문을 읽고, 요구 기술·자격요건·우대사항·핵심 키워드를 원문에서 그대로 뽑아 정리하겠습니다. ",
    "en": "First, let me read the posting text and extract the required skills, qualifications, preferred points, and key terms exactly as written. ",
}

def _as_str_list(v, limit=12):
    out = []
    if isinstance(v, list):
        for x in v:
            s = str(x).strip(" -•\t")
            if s:
                out.append(s)
    elif isinstance(v, str) and v.strip():
        for part in re.split(r"[\n;]+", v):
            s = part.strip(" -•\t")
            if s:
                out.append(s)
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq[:limit]

def _build_posting(parsed):
    if not isinstance(parsed, dict):
        return None
    posting = {
        "title": str(parsed.get("title") or "").strip(),
        "summary": str(parsed.get("summary") or "").strip(),
        "responsibilities": _as_str_list(parsed.get("responsibilities")),
        "requirements": _as_str_list(parsed.get("requirements")),
        "preferred": _as_str_list(parsed.get("preferred")),
        "conditions": _as_str_list(parsed.get("conditions")),
    }
    if not posting["title"] and not posting["responsibilities"] and not posting["requirements"]:
        return None
    return posting

def _build_analysis(parsed):
    if not isinstance(parsed, dict):
        return None
    analysis = {
        "role": str(parsed.get("role") or "").strip(),
        "summary": str(parsed.get("summary") or "").strip(),
        "requirements": _as_str_list(parsed.get("requirements") or parsed.get("required_skills") or parsed.get("qualifications")),
        "preferred": _as_str_list(parsed.get("preferred")),
        "keywords": _as_str_list(parsed.get("keywords"), limit=20),
    }
    if not (analysis["role"] or analysis["requirements"]):
        return None
    return analysis


class PostingGenReq(BaseModel):
    role: str = ""             # 직무/포지션
    company: str = ""          # 회사(선택)
    skills: str = ""           # 요구 기술
    responsibilities: str = "" # 주요 업무 힌트
    level: str = ""            # 경력 수준(신입/주니어/시니어)
    employment_type: str = ""  # 고용 형태
    notes: str = ""            # 기타
    lang: str = "ko"

def posting_gen_prompt(req, lang):
    given = []
    def add(ko, en, val):
        if val and val.strip():
            given.append(f"- {(ko if lang=='ko' else en)}: {val.strip()}")
    add("직무/포지션", "Role", req.role)
    add("회사", "Company", req.company)
    add("요구 기술", "Required skills", req.skills)
    add("주요 업무(힌트)", "Responsibilities (hints)", req.responsibilities)
    add("경력 수준", "Level", req.level)
    add("고용 형태", "Employment type", req.employment_type)
    add("기타", "Notes", req.notes)
    given_block = "\n".join(given) if given else ("(제공 정보 없음)" if lang == "ko" else "(no info provided)")
    if lang == "en":
        return f"""You help a recruiter draft a job posting. Build a reasonable posting based on the information below.

Rules:
- Center the draft on the provided role/skills/info. Do NOT assert unprovided specifics such as exact salary, benefits, or confidential company facts (use general wording if needed).
- Short, clear phrases per item. English.
- After thinking, output ONLY this JSON:
{{"title": "...", "summary": "...", "responsibilities": ["..."], "requirements": ["..."], "preferred": ["..."], "conditions": ["..."]}}

[Provided information]
{given_block}"""
    return f"""당신은 채용 담당자를 도와 '채용공고 초안'을 작성합니다. 아래 제공 정보를 바탕으로 합리적인 공고를 구성하세요.

규칙:
- 제공된 직무·기술·정보를 중심으로 작성하세요. 연봉·복지·회사 기밀처럼 제공되지 않은 구체 수치·사실을 단정하지 마세요(필요하면 일반적 표현).
- 각 항목은 짧고 명확한 구/문장으로. 한국어로.
- 사고를 마친 뒤 JSON만 출력:
{{"title": "...", "summary": "...", "responsibilities": ["..."], "requirements": ["..."], "preferred": ["..."], "conditions": ["..."]}}

[제공 정보]
{given_block}"""

@app.post("/posting/generate")
def posting_generate(req: PostingGenReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr:
        return nr
    prompt = posting_gen_prompt(req, lang)
    gen = run_llm(prompt, POSTING_GEN_PREFILL[lang], use_adapter=False, max_new_tokens=2560, do_sample=True)
    posting = _build_posting(parse_json_lenient(gen))
    if posting:
        return {"ok": True, "lang": lang, "posting": posting}
    return {"ok": False, "error": MSG[lang]["gen_fail"], "raw": gen[-1500:]}


class PostingAnalyzeReq(BaseModel):
    text: str          # 붙여넣은 채용공고 원문 (필수)
    lang: str = "ko"

def posting_analyze_prompt(text, lang):
    if lang == "en":
        return f"""You analyze a job-posting text and structure its key items.

Rules:
- Extract ONLY what is actually in the text. Do NOT add items not present.
- requirements = required/must-have, preferred = nice-to-have. Keep them distinct (do not put preferred items under requirements).
- Split into short items; keywords = key terms.
- After thinking, output ONLY this JSON:
{{"role": "...", "summary": "...", "requirements": ["..."], "preferred": ["..."], "keywords": ["..."]}}

[Job posting text]
{text}"""
    return f"""당신은 채용공고 원문을 분석해 핵심 항목을 구조화합니다.

규칙:
- 원문에 실제로 있는 내용만 추출하세요. 원문에 없는 항목·기술·자격을 추가하지 마세요.
- requirements = '자격요건/필수', preferred = '우대사항' 으로 정확히 구분하세요(우대사항을 requirements에 넣지 말 것).
- 각 항목은 짧게 분리하고, keywords에는 핵심 단어를 담으세요.
- 사고를 마친 뒤 JSON만 출력:
{{"role": "...", "summary": "...", "requirements": ["..."], "preferred": ["..."], "keywords": ["..."]}}

[채용공고 원문]
{text}"""

@app.post("/posting/analyze")
def posting_analyze(req: PostingAnalyzeReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr:
        return nr
    if not (req.text or "").strip():
        return {"ok": False, "error": ("분석할 공고 원문을 입력하세요." if lang == "ko" else "Provide posting text to analyze.")}
    prompt = posting_analyze_prompt(req.text.strip(), lang)
    gen = run_llm(prompt, POSTING_ANALYZE_PREFILL[lang], use_adapter=False, max_new_tokens=2048)
    analysis = _build_analysis(parse_json_lenient(gen))
    if analysis:
        return {"ok": True, "lang": lang, "analysis": analysis}
    return {"ok": False, "error": MSG[lang]["gen_fail"], "raw": gen[-1500:]}


from fastapi import UploadFile, File, Form
import stt as _stt
from voice import voice_metrics

@app.post("/interview/stt")
async def interview_stt(file: UploadFile = File(...), lang: str = Form("ko")):
    try:
        audio = await file.read()
        if not audio:
            return {"ok": False, "error": "오디오 파일이 비어 있습니다. / Empty audio file."}
        if len(audio) > 25 * 1024 * 1024:
            return {"ok": False, "error": "오디오가 너무 큽니다(최대 25MB). / Audio too large (max 25MB)."}
        result = _stt.transcribe_bytes(audio, file.filename or "audio.bin", language=norm_lang(lang))
        result["voice"] = voice_metrics(result, norm_lang(lang))
        return {"ok": True, **result}
    except Exception as e:
        return {"ok": False, "error": f"STT 처리 실패 / STT failed: {type(e).__name__}: {e}"}

# ===== whisper 웜 스타트 =====
import threading as _th
_th.Thread(target=_stt.get_model, daemon=True, name="whisper-warmup").start()

# ============================================================
# /resume/analyze — 자소서/이력서 분석 (v5: 하이브리드 + 환각필터)
#   점수 = 파이썬 휴리스틱, 질적 = LLM(점수주입+근거강제+고구체성 가드+후처리 필터)
# ============================================================
import re as _re_ra

_RA_TECH = ["react","vue","angular","svelte","next.js","nuxt","typescript","javascript",
    "node","express","nestjs","python","django","flask","fastapi","java","spring","kotlin",
    "c++","c#","golang","rust","php","laravel","ruby","rails","sql","mysql","postgresql",
    "postgres","mongodb","redis","graphql","rest","api","redux","mobx","zustand","recoil",
    "tailwind","sass","webpack","vite","babel","docker","kubernetes","k8s","aws","gcp",
    "azure","git","github","gitlab","jenkins","ci/cd","jest","cypress","playwright","html",
    "css","jquery","websocket","oauth","jwt","kafka","rabbitmq","elasticsearch","nginx"]

_RA_METRIC = ["감소","증가","단축","개선","향상","절감","달성","상승","하락","절약","축소","확대",
    "reduced","increased","improved","decreased","achieved","boosted","cut","saved","grew",
    "optimized","raised","lowered"]

_RA_STAR = {
    "S": ["상황","배경","문제","이슈","당시","현황","situation","problem","context","challenge","issue"],
    "T": ["과제","목표","역할","담당","미션","task","goal","responsib","objective","mission","role"],
    "A": ["주도","수행","구현","개발","적용","진행","도입","리팩","마이그","설계","구축","action",
          "implement","led","built","develop","designed","migrat","refactor"],
    "R": ["결과","성과","달성","개선","감소","증가","단축","절감","효과","result","achiev","improv",
          "reduc","increas","impact","outcome"],
}

# 고구체성 문서에 대한 '지표 부족' 류 환각 약점을 거르는 키워드
_RA_METRIC_KW = ["지표","수치","정량","계량","측정","metric","quantif"]
_RA_LACK_KW = ["부족","부재","없","미흡","약함","lack","missing","absent","insufficient","limited"]

def _ra_clamp(x, lo=1, hi=10):
    return max(lo, min(hi, int(round(x))))

def _ra_spec_score(text):
    tl = text.lower()
    nums = len(_re_ra.findall(r"\d+", text))
    pcts = len(_re_ra.findall(r"%|퍼센트|percent", tl))
    tech = sum(1 for t in _RA_TECH if t in tl)
    metrics = sum(1 for m in _RA_METRIC if m in tl)
    raw = min(nums, 8) + min(pcts, 5) * 2 + min(tech, 7) * 1.5 + min(metrics, 5)
    return _ra_clamp(2 + raw * 0.45)

def _ra_star_score(text):
    tl = text.lower()
    present = sum(1 for kws in _RA_STAR.values() if any(kw.lower() in tl for kw in kws))
    base = {0: 1, 1: 3, 2: 5, 3: 7, 4: 8}[present]
    if present >= 3 and _re_ra.search(r"\d", text):
        base = min(10, base + 1)
    return base

def _ra_job_fit(doc, posting):
    if not (posting or "").strip():
        return 0
    def _toks(s):
        return set(_re_ra.findall(r"[A-Za-z가-힣]{2,}", s.lower()))
    pt, dt = _toks(posting), _toks(doc)
    if not pt:
        return 0
    return _ra_clamp(2 + (len(pt & dt) / len(pt)) * 10)

def _ra_overall(spec, star, jobfit, has_posting):
    if has_posting:
        return _ra_clamp(0.4 * spec + 0.35 * star + 0.25 * jobfit)
    return _ra_clamp(0.55 * spec + 0.45 * star)

def _ra_filter_weaknesses(weaknesses, spec, lang):
    # 구체성이 높은(>=8) 문서에 '정량 지표 부족' 류 약점이 달리면 모순 → 제거
    if spec < 8 or not isinstance(weaknesses, list):
        return weaknesses
    kept = []
    for w in weaknesses:
        s = str(w).lower()
        metric_lack = any(m in s for m in _RA_METRIC_KW) and any(l in s for l in _RA_LACK_KW)
        if not metric_lack:
            kept.append(w)
    if not kept:
        kept = ["내용의 깊이나 직무 연관성 측면에서 보강 여지" if lang == "ko"
                else "Could deepen content or strengthen role-relevance"]
    return kept

RESUME_QUAL_PREFILL = {
    "ko": "먼저 문서에 실제로 적힌 내용을 근거로 강점과 약점을 정리하겠습니다. ",
    "en": "First, let me organize strengths and weaknesses grounded strictly in what the document states. ",
}

class ResumeAnalyzeReq(BaseModel):
    document: str
    job_posting: str = ""
    role: str = ""
    lang: str = "ko"

def resume_qual_prompt(lang, document, job_posting, role, spec, star, jobfit, has_posting):
    high = spec >= 8
    if lang == "en":
        role_line = f"\n- Target role: {role}" if role.strip() else ""
        posting_line = f"\n\n[Job Posting]\n{job_posting}" if has_posting else ""
        jf = f", job-fit {jobfit}/10" if has_posting else ""
        hi = ("\n- IMPORTANT: This document has a HIGH specificity score (it already contains sufficient quantitative metrics). Do NOT write weaknesses like 'lacks metrics', 'insufficient numbers', or 'needs quantitative results'. Find weaknesses in depth of content, role-relevance, clarity of technical explanation, or differentiation instead." if high else "")
        return f"""You are a career coach giving QUALITATIVE feedback on a candidate's resume/cover letter. (Scores are computed separately.)

Auto-scores for THIS document (reference): specificity {spec}/10, STAR {star}/10{jf}. Your feedback MUST NOT contradict these.

Rules:
- Ground everything STRICTLY in what the document actually says. Do NOT invent content that is not there.{hi}
- strengths (2-3), weaknesses (2-3), suggestions (2-3 actionable), summary (ONE sentence)
- Output ONE complete JSON only. No markdown, no preamble.
{{"strengths": ["item", "item"], "weaknesses": ["item", "item"], "suggestions": ["tip", "tip"], "summary": "one sentence"}}

[Document]{role_line}
{document}{posting_line}"""
    else:
        role_line = f"\n- 지원 직무: {role}" if role.strip() else ""
        posting_line = f"\n\n[채용공고]\n{job_posting}" if has_posting else ""
        jf = f", 공고 적합도 {jobfit}/10" if has_posting else ""
        hi = ("\n- ★중요: 이 문서는 구체성 점수가 높습니다(이미 정량적 지표·수치가 충분함). '성과 지표 부족', '수치 부족', '정량적 결과 부족' 같은 약점은 절대 쓰지 마세요. 약점은 내용의 깊이, 직무 연관성, 기술 설명의 명확성, 차별성에서 찾으세요." if high else "")
        return f"""당신은 지원자의 자소서/이력서에 질적 피드백을 주는 커리어 코치입니다. (점수는 별도로 계산됩니다.)

이 문서의 자동 평가 점수(참고): 구체성 {spec}/10, STAR {star}/10{jf}. 피드백은 이 점수와 모순되면 안 됩니다.

규칙:
- 모든 내용을 문서에 실제로 적힌 것에만 근거하세요. 문서에 없는 내용을 지어내지 마세요.{hi}
- strengths(강점 2~3개), weaknesses(약점 2~3개), suggestions(실행 가능한 개선 제안 2~3개), summary(한 문장 총평)
- 완전한 JSON 하나만 출력. 마크다운·서론 금지.
{{"strengths": ["항목", "항목"], "weaknesses": ["항목", "항목"], "suggestions": ["조언", "조언"], "summary": "한 문장 총평"}}

[자소서/이력서]{role_line}
{document}{posting_line}"""

@app.post("/resume/analyze")
def resume_analyze(req: ResumeAnalyzeReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr:
        return nr
    doc_text = (req.document or "").strip()
    if not doc_text:
        return {"ok": False, "error": ("분석할 자소서/이력서를 입력하세요." if lang == "ko" else "Provide a resume/cover letter to analyze.")}
    if len(doc_text) > 12000:
        return {"ok": False, "error": ("자소서는 12,000자 이내여야 합니다." if lang == "ko" else "Resume must be under 12,000 characters.")}
    has_posting = bool((req.job_posting or "").strip())
    spec = _ra_spec_score(doc_text)
    star = _ra_star_score(doc_text)
    jobfit = _ra_job_fit(doc_text, req.job_posting)
    overall = _ra_overall(spec, star, jobfit, has_posting)
    prompt = resume_qual_prompt(lang, doc_text, req.job_posting, req.role, spec, star, jobfit, has_posting)
    gen = run_llm(prompt, RESUME_QUAL_PREFILL[lang], use_adapter=False, max_new_tokens=1536, do_sample=False)
    d = parse_json_lenient(gen) or {}
    weaknesses = _ra_filter_weaknesses(d.get("weaknesses", []), spec, lang)
    analysis = {
        "overall_score": overall,
        "specificity_score": spec,
        "star_score": star,
        "job_fit_score": jobfit,
        "strengths": d.get("strengths", []),
        "weaknesses": weaknesses,
        "suggestions": d.get("suggestions", []),
        "summary": d.get("summary", ""),
    }
    return {"ok": True, "lang": lang, "analysis": analysis}


# ===== 자기소개서 자동 작성 (스트리밍) — /resume/cover-letter/stream =====
def _coverletter_stream_gen(prompt, prefill, lang):
    from transformers import TextIteratorStreamer
    msgs = [{"role": "user", "content": prompt}]
    text = M["llm_tok"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill
    enc = M["llm_tok"](text, return_tensors="pt", add_special_tokens=False).to(M["llm"].device)
    streamer = TextIteratorStreamer(M["llm_tok"], skip_prompt=True, skip_special_tokens=False)
    holder = {}
    def _gen():
        try:
            with torch.no_grad():
                with M["llm"].disable_adapter():
                    holder["out"] = M["llm"].generate(
                        **enc, max_new_tokens=2560, do_sample=True,
                        temperature=0.8, top_p=0.9, streamer=streamer)
        except Exception as e:
            holder["err"] = e
    GEN_LOCK.acquire()
    t = threading.Thread(target=_gen, daemon=True)
    t.start()
    full, buf, started = [], "", False
    try:
        for chunk in streamer:
            if not chunk:
                continue
            full.append(chunk)
            piece = chunk
            for _tok in ("[|endofturn|]", "[|assistant|]", "[|system|]", "[|user|]", "[|endoftext|]"):
                piece = piece.replace(_tok, "")
            if not piece:
                continue
            if started:
                yield "data: " + json.dumps({"type": "token", "text": piece}, ensure_ascii=False) + "\n\n"
            else:
                buf += piece
                k = buf.find("</thought>")
                if k >= 0:
                    started = True
                    after = buf[k + len("</thought>"):].lstrip()
                    if after:
                        yield "data: " + json.dumps({"type": "token", "text": after}, ensure_ascii=False) + "\n\n"
                    buf = ""
        t.join()
        if "err" in holder:
            raise holder["err"]
        cover = _strip_thought("".join(full))
        payload = {"type": "done", "ok": True, "cover_letter": cover} if cover \
            else {"type": "done", "ok": False, "error": MSG[lang]["gen_fail"]}
        yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
    except Exception as e:
        yield "data: " + json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False) + "\n\n"
    finally:
        if t.is_alive():
            t.join()
        try:
            GEN_LOCK.release()
        except RuntimeError:
            pass


@app.post("/resume/cover-letter/stream")
def resume_cover_letter_stream(req: CoverLetterReq):
    lang = norm_lang(req.lang)
    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    def _one(obj):
        def _g():
            yield "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
        return _g()
    nr = not_ready(lang)
    if nr:
        return StreamingResponse(_one({"type": "error", "error": nr["error"]}), media_type="text/event-stream", headers=sse_headers)
    prompt = cover_letter_prompt(req, lang)
    return StreamingResponse(_coverletter_stream_gen(prompt, CL_PREFILL[lang], lang), media_type="text/event-stream", headers=sse_headers)


# ============================================================================
#  적응형 레벨테스트 (Adaptive Level Test)  — server.py 하단에 append
#  설계: 난이도 숫자/문항 선택 = 규칙(stateless, 환각 0)
#        추천 텍스트 = base EXAONE(계산된 결과 수치만 근거, 후처리 폴백)
#  - POST /leveltest/next          : 객관식 적응 루프(즉시채점→문항마다 난이도 변경)
#  - POST /leveltest/session-next  : 서술형 면접 세션 점수 → 다음 난이도 + 로드맵
#  상태는 요청에 실려 옴(프론트 보관) → 서버 무상태. qid만 주고받아 정답은 서버에만.
# ============================================================================

# ---- 내장 객관식 문제 은행 (topic: fe/be/cs, diff: 1=하 2=중 3=상) ----
# answer = options 내 정답 인덱스. (정답 위치는 문항마다 분산)
_LT_BANK = [
    # ===== 난이도 1 (하/기초) =====
    {"qid":"q101","topic":"fe","diff":1,"q":"HTML에서 가장 큰 제목을 나타내는 태그는?",
     "options":["<head>","<h1>","<h6>","<title>"],"answer":1,
     "explanation":"<h1>이 최상위(가장 큰) 제목 태그이며 <h6>로 갈수록 작아집니다."},
    {"qid":"q102","topic":"fe","diff":1,"q":"CSS에서 글자 색을 지정하는 속성은?",
     "options":["background","font-size","color","text"],"answer":2,
     "explanation":"글자 색은 color 속성으로 지정합니다."},
    {"qid":"q103","topic":"be","diff":1,"q":"HTTP 상태 코드 404가 의미하는 것은?",
     "options":["요청한 리소스 없음","서버 내부 오류","정상 처리","권한 없음"],"answer":0,
     "explanation":"404 Not Found는 요청한 리소스를 찾을 수 없음을 뜻합니다."},
    {"qid":"q104","topic":"be","diff":1,"q":"SQL에서 데이터를 조회하는 명령은?",
     "options":["INSERT","SELECT","DELETE","UPDATE"],"answer":1,
     "explanation":"SELECT는 데이터를 조회(읽기)하는 명령입니다."},
    {"qid":"q105","topic":"cs","diff":1,"q":"2진수 1010을 10진수로 바꾸면?",
     "options":["8","5","12","10"],"answer":3,
     "explanation":"1010(2) = 8+0+2+0 = 10 입니다."},
    {"qid":"q106","topic":"cs","diff":1,"q":"1바이트(byte)는 몇 비트(bit)인가?",
     "options":["4비트","8비트","16비트","1024비트"],"answer":1,
     "explanation":"1바이트는 8비트입니다."},
    {"qid":"q107","topic":"fe","diff":1,"q":"웹페이지의 '구조'를 담당하는 언어는?",
     "options":["CSS","JavaScript","HTML","SQL"],"answer":2,
     "explanation":"HTML은 구조, CSS는 스타일, JavaScript는 동작을 담당합니다."},
    {"qid":"q108","topic":"be","diff":1,"q":"REST API에서 새 데이터 '생성'에 주로 쓰는 HTTP 메서드는?",
     "options":["GET","POST","DELETE","PATCH"],"answer":1,
     "explanation":"POST는 새 리소스 생성에 주로 사용됩니다."},
    {"qid":"q109","topic":"fe","diff":1,"q":"다음 중 JavaScript의 '변수 선언' 키워드가 아닌 것은?",
     "options":["let","const","function","var"],"answer":2,
     "explanation":"let/const/var가 변수 선언 키워드이며 function은 함수 선언 키워드입니다."},
    {"qid":"q110","topic":"cs","diff":1,"q":"프로그램 실행 중 데이터를 임시로 저장하는 휘발성 메모리는?",
     "options":["HDD","RAM","SSD","ROM"],"answer":1,
     "explanation":"RAM은 실행 중 임시 데이터를 담는 휘발성 주기억장치입니다."},

    # ===== 난이도 2 (중) =====
    {"qid":"q201","topic":"fe","diff":2,"q":"React에서 컴포넌트의 '상태'를 관리하는 기본 Hook은?",
     "options":["useEffect","useState","useMemo","useContext"],"answer":1,
     "explanation":"useState는 컴포넌트 상태값과 갱신 함수를 제공합니다."},
    {"qid":"q202","topic":"fe","diff":2,"q":"CSS Flexbox에서 '주축(main axis)' 방향 정렬 속성은?",
     "options":["align-items","justify-content","flex-wrap","order"],"answer":1,
     "explanation":"justify-content는 주축 정렬, align-items는 교차축 정렬입니다."},
    {"qid":"q203","topic":"be","diff":2,"q":"관계형 DB에서 두 테이블을 연결하는 데 쓰는 키는?",
     "options":["기본 키","후보 키","외래 키","슈퍼 키"],"answer":2,
     "explanation":"외래 키(Foreign Key)는 다른 테이블의 기본 키를 참조해 연결합니다."},
    {"qid":"q204","topic":"be","diff":2,"q":"JS에서 비동기 작업의 '미래 결과'를 표현하기 위해 반환되는 객체는?",
     "options":["Promise","Callback","Array","Object"],"answer":0,
     "explanation":"Promise는 비동기 작업의 완료/실패와 결과를 표현합니다."},
    {"qid":"q205","topic":"cs","diff":2,"q":"스택(Stack)의 데이터 처리 방식은?",
     "options":["FIFO(선입선출)","LIFO(후입선출)","우선순위 순","무작위"],"answer":1,
     "explanation":"스택은 마지막에 넣은 것이 먼저 나오는 LIFO 구조입니다."},
    {"qid":"q206","topic":"cs","diff":2,"q":"시간복잡도 O(log n)에 해당하는 대표적인 알고리즘은?",
     "options":["선형 탐색","버블 정렬","이진 탐색","완전 탐색"],"answer":2,
     "explanation":"정렬된 배열에서의 이진 탐색은 O(log n)입니다."},
    {"qid":"q207","topic":"fe","diff":2,"q":"다른 출처(도메인)로의 브라우저 요청을 허용·제어하는 메커니즘은?",
     "options":["CORS","CSRF","XSS","JWT"],"answer":0,
     "explanation":"CORS(교차 출처 리소스 공유)는 다른 출처 요청 허용 정책을 정의합니다."},
    {"qid":"q208","topic":"be","diff":2,"q":"같은 요청을 여러 번 보내도 결과가 동일하게 유지되는 성질은?",
     "options":["원자성","멱등성","일관성","지속성"],"answer":1,
     "explanation":"멱등성(idempotent)은 동일 요청을 반복해도 상태 변화가 같음을 뜻합니다(예: PUT, DELETE)."},
    {"qid":"q209","topic":"cs","diff":2,"q":"DB 트랜잭션의 ACID 중 '전부 반영 아니면 전부 취소'를 뜻하는 것은?",
     "options":["일관성","원자성","격리성","지속성"],"answer":1,
     "explanation":"원자성(Atomicity)은 트랜잭션이 전부 실행되거나 전혀 실행되지 않음을 보장합니다."},
    {"qid":"q210","topic":"be","diff":2,"q":"HTTP에서 '리소스를 부분 수정'할 때 의미상 적절한 메서드는?",
     "options":["GET","PATCH","HEAD","OPTIONS"],"answer":1,
     "explanation":"PATCH는 리소스의 일부만 수정할 때 사용합니다(전체 교체는 PUT)."},

    # ===== 난이도 3 (상) =====
    {"qid":"q301","topic":"fe","diff":3,"q":"React에서 함수를 메모이제이션해 불필요한 재생성을 막는 Hook은?",
     "options":["useState","useCallback","useEffect","useReducer"],"answer":1,
     "explanation":"useCallback은 의존성이 같으면 같은 함수 참조를 유지해 재생성을 막습니다."},
    {"qid":"q302","topic":"be","diff":3,"q":"범위 검색에 유리해 DB 인덱스로 널리 쓰이는 자료구조는?",
     "options":["해시 테이블","B-Tree","연결 리스트","스택"],"answer":1,
     "explanation":"B-Tree(B+Tree)는 정렬을 유지해 범위 검색·정렬 조회에 유리합니다."},
    {"qid":"q303","topic":"cs","diff":3,"q":"교착상태(Deadlock)의 4가지 '필요조건'이 아닌 것은?",
     "options":["상호 배제","점유와 대기","선점 가능","순환 대기"],"answer":2,
     "explanation":"교착상태는 '비선점'이 조건입니다. 자원을 선점할 수 있으면 교착이 풀립니다."},
    {"qid":"q304","topic":"be","diff":3,"q":"분산 시스템에서 일관성·가용성·분단내성을 동시에 만족할 수 없다는 이론은?",
     "options":["ACID","CAP 정리","BASE","SOLID"],"answer":1,
     "explanation":"CAP 정리는 네트워크 분단 시 일관성과 가용성 중 하나를 포기해야 함을 말합니다."},
    {"qid":"q305","topic":"fe","diff":3,"q":"브라우저 렌더링에서 레이아웃을 다시 계산하는, 비용이 큰 작업은?",
     "options":["리페인트","리플로우","컴포지팅","하이드레이션"],"answer":1,
     "explanation":"리플로우(레이아웃 재계산)는 리페인트보다 비용이 큽니다."},
    {"qid":"q306","topic":"cs","diff":3,"q":"TCP 3-way handshake의 올바른 순서는?",
     "options":["ACK → SYN → FIN","SYN → SYN-ACK → ACK","FIN → ACK → SYN","SYN → ACK → FIN"],"answer":1,
     "explanation":"연결 수립은 SYN → SYN-ACK → ACK 순서로 이뤄집니다."},
    {"qid":"q307","topic":"be","diff":3,"q":"한 트랜잭션이 '아직 커밋되지 않은' 데이터를 읽는 동시성 문제는?",
     "options":["Dirty Read","Phantom Read","Non-repeatable Read","Lost Update"],"answer":0,
     "explanation":"Dirty Read는 커밋 전 데이터를 읽어 롤백 시 오류가 되는 문제입니다."},
    {"qid":"q308","topic":"fe","diff":3,"q":"JS 이벤트 루프에서 Promise의 then 콜백이 들어가는 큐는?",
     "options":["매크로태스크 큐","마이크로태스크 큐","콜 스택","렌더 큐"],"answer":1,
     "explanation":"Promise 콜백은 마이크로태스크 큐에 들어가 매크로태스크보다 먼저 처리됩니다."},
    {"qid":"q309","topic":"cs","diff":3,"q":"다음 중 '해시 충돌' 해결 방법이 아닌 것은?",
     "options":["체이닝","개방 주소법","이진 탐색","이중 해싱"],"answer":2,
     "explanation":"이진 탐색은 충돌 해결법이 아닙니다(체이닝·개방주소법·이중해싱이 해당)."},
    {"qid":"q310","topic":"be","diff":3,"q":"쓰기 시 캐시와 DB를 '동시에' 갱신하는 캐시 전략은?",
     "options":["Write-Back","Write-Through","Cache-Aside","Write-Around"],"answer":1,
     "explanation":"Write-Through는 캐시와 DB를 동시에 갱신해 일관성이 높습니다(쓰기 지연은 큼)."},
]
_LT_BY_QID = {it["qid"]: it for it in _LT_BANK}
_LT_TOTAL = 8                      # 한 회 레벨테스트 문항 수
_LT_DIFF_KO = {1: "하", 2: "중", 3: "상"}

def _lt_topics_for(role):
    r = (role or "").lower()
    if "front" in r or "프론트" in r:  return ["fe", "cs"]
    if "back" in r or "백엔드" in r:   return ["be", "cs"]
    return ["fe", "be", "cs"]          # 풀스택/미지정

def _lt_next_diff(cur, correct):
    """규칙: 맞으면 +1, 틀리면 -1 (1~3 클램프). 환각 없음."""
    return max(1, min(3, cur + (1 if correct else -1)))

def _lt_pick(target, topics, used, step):
    """target 난이도·허용 토픽·미사용 중에서 결정론적으로 1개 선택(없으면 인접 난이도로 완화)."""
    order = [target] + sorted([1, 2, 3], key=lambda d: abs(d - target))
    seen = set()
    for d in order:
        if d in seen:
            continue
        seen.add(d)
        cand = [it for it in _LT_BANK
                if it["diff"] == d and it["topic"] in topics and it["qid"] not in used]
        if not cand:
            continue
        cand.sort(key=lambda it: it["qid"])
        return cand[(step * 7 + target) % len(cand)]   # 상태로부터 결정론적 → 재생성 시 동일
    # 토픽 무시하고라도 아무거나
    cand = [it for it in _LT_BANK if it["qid"] not in used]
    cand.sort(key=lambda it: it["qid"])
    return cand[step % len(cand)] if cand else None

def _lt_replay(profile, answers):
    """제출된 answers(qid·chosen)를 은행으로 채점하며 난이도 경로를 재구성(무상태)."""
    topics = _lt_topics_for(profile.get("role"))
    cur = 2                       # 시작 난이도 '중'
    graded, used, path = [], set(), []
    for a in answers:
        qid = str((a or {}).get("qid") or "")
        it = _LT_BY_QID.get(qid)
        if not it:
            continue
        chosen = (a or {}).get("chosen", -1)
        try:
            chosen = int(chosen)
        except (TypeError, ValueError):
            chosen = -1
        correct = (chosen == it["answer"])
        path.append(cur)
        graded.append({"qid": qid, "topic": it["topic"], "diff": it["diff"],
                       "chosen": chosen, "answer": it["answer"], "correct": correct,
                       "explanation": it["explanation"]})
        used.add(qid)
        cur = _lt_next_diff(cur, correct)
    return topics, cur, graded, used, path

def _lt_level_label(acc, diff_reached):
    """정답률 + 도달 난이도 → 레벨 라벨(규칙)."""
    if acc >= 0.85 and diff_reached >= 3:   return "고급"
    if acc >= 0.65 and diff_reached >= 2:   return "중급"
    if acc >= 0.45:                          return "초급"
    return "입문"

def _lt_summary(graded):
    n = len(graded)
    correct = sum(1 for g in graded if g["correct"])
    acc = (correct / n) if n else 0.0
    diff_reached = max((g["diff"] for g in graded if g["correct"]), default=1)
    by = {}
    for g in graded:
        t = g["topic"]; by.setdefault(t, [0, 0])
        by[t][1] += 1
        if g["correct"]:
            by[t][0] += 1
    by_topic = {t: round(c / max(tot, 1), 2) for t, (c, tot) in by.items()}
    weak = sorted([t for t, r in by_topic.items() if r < 0.6], key=lambda t: by_topic[t])
    label = _lt_level_label(acc, diff_reached)
    return {"level": label, "accuracy": round(acc, 2), "correct": correct, "total": n,
            "difficulty_reached": _LT_DIFF_KO.get(diff_reached, "중"),
            "by_topic": by_topic, "weak_topics": weak}

_LT_TOPIC_KO = {"fe": "프론트엔드", "be": "백엔드", "cs": "CS 기초/공통"}

def _lt_fallback_rec(stats):
    """LLM 실패/미준비 시 쓰는 규칙 기반 추천(항상 사실에 근거)."""
    weak = stats.get("weak_topics") or []
    if weak:
        w = ", ".join(_LT_TOPIC_KO.get(t, t) for t in weak)
        focus = f"특히 {w} 영역의 정답률이 낮으니 이 부분을 우선 보강하세요. "
    else:
        focus = "전반적으로 고른 편입니다. 한 단계 높은 난이도로 도전해 보세요. "
    return (f"현재 레벨은 '{stats['level']}'(정답률 {int(stats['accuracy']*100)}%, "
            f"도달 난이도 '{stats['difficulty_reached']}')입니다. {focus}"
            f"다음 단계로는 약점 주제의 개념을 정리한 뒤, 같은 난이도 문제로 정답률을 끌어올리고 "
            f"안정되면 난이도를 한 칸 올리는 순서를 권합니다.")

LT_PREFILL = {
    "ko": "먼저 이 학습자의 레벨테스트 결과(정답률·약점 토픽·도달 난이도)를 살펴보고, 가장 도움이 될 다음 학습 방향을 생각하겠습니다. ",
    "en": "First, let me review this learner's level-test results (accuracy, weak topics, difficulty reached) and plan the most helpful next learning steps. ",
}

def _lt_ai_recommend(profile, stats, lang):
    """base EXAONE로 로드맵 추천 생성. 계산된 수치만 근거로 강제. 실패하면 규칙 폴백."""
    try:
        by = ", ".join(f"{_LT_TOPIC_KO.get(t,t)} {int(r*100)}%" for t, r in (stats.get("by_topic") or {}).items())
        weak = ", ".join(_LT_TOPIC_KO.get(t, t) for t in (stats.get("weak_topics") or [])) or "없음"
        role = profile.get("role") or "미지정"
        langs = ", ".join(profile.get("languages") or []) or "미지정"
        prompt = (
            "당신은 개발자 학습 코치입니다. 아래 '레벨테스트 결과 수치'만 근거로, 다음 학습 로드맵을 한국어 3~4문장으로 제시하세요.\n"
            "절대 규칙: 아래에 없는 점수·수치·통계를 새로 지어내지 마세요. 약점 주제를 먼저 보강하는 방향으로, 다음에 풀 난이도도 한 단계 제안하세요.\n\n"
            f"[지원자] 희망직군: {role} / 사용언어: {langs}\n"
            f"[레벨] {stats['level']} (정답률 {int(stats['accuracy']*100)}%, 도달 난이도 '{stats['difficulty_reached']}')\n"
            f"[주제별 정답률] {by}\n"
            f"[약점 주제] {weak}\n"
        )
        gen = run_llm(prompt, LT_PREFILL.get(lang, LT_PREFILL["ko"]),
                      use_adapter=False, max_new_tokens=1024, do_sample=False)
        body = _strip_thought(gen).strip()
        # 후처리 가드: 비었거나 너무 짧으면(생성 실패로 간주) 규칙 폴백
        if len(body) < 30:
            return _lt_fallback_rec(stats), "fallback"
        return body, "llm"
    except Exception as e:
        print(f">>> [leveltest] recommend fallback: {e}", flush=True)
        return _lt_fallback_rec(stats), "fallback"


class LevelTestReq(BaseModel):
    profile: dict = {}        # {role, career, goal, languages:[...]}
    answers: list = []        # [{qid, chosen}] 지금까지 제출한 답(빈 배열이면 첫 문항 요청)
    lang: str = "ko"

@app.post("/leveltest/next")
def leveltest_next(req: LevelTestReq):
    lang = norm_lang(req.lang)
    profile = req.profile if isinstance(req.profile, dict) else {}
    answers = req.answers if isinstance(req.answers, list) else []

    topics, cur, graded, used, _path = _lt_replay(profile, answers)
    prev = None
    if graded:
        g = graded[-1]
        prev = {"correct": g["correct"], "answer": g["answer"], "explanation": g["explanation"]}

    # 종료 조건
    if len(graded) >= _LT_TOTAL:
        stats = _lt_summary(graded)
        nr = not_ready(lang)               # 추천에 LLM 필요 → 미준비면 규칙 폴백
        if nr:
            rec, src = _lt_fallback_rec(stats), "fallback"
        else:
            rec, src = _lt_ai_recommend(profile, stats, lang)
        stats["recommendation"] = rec
        stats["recommendation_source"] = src
        return {"ok": True, "done": True, "total": _LT_TOTAL, "prev": prev, "result": stats}

    # 다음 문항 선택(규칙·결정론적)
    nxt = _lt_pick(cur, topics, used, len(graded))
    if not nxt:
        stats = _lt_summary(graded)
        stats["recommendation"] = _lt_fallback_rec(stats)
        stats["recommendation_source"] = "fallback"
        return {"ok": True, "done": True, "total": len(graded), "prev": prev,
                "result": stats, "note": "문제 은행 소진"}
    return {"ok": True, "done": False, "q_no": len(graded) + 1, "total": _LT_TOTAL,
            "difficulty": _LT_DIFF_KO[cur], "prev": prev,
            "question": {"qid": nxt["qid"], "q": nxt["q"], "options": nxt["options"]}}


# ---- 서술형 면접 세션 → 다음 난이도 + 로드맵 (둘 다 준비: 세션 단위 적응) ----
class SessionDiffReq(BaseModel):
    scores: dict = {}         # 5축 점수(0~100): technical_accuracy/logic/specificity/depth/communication 등
    current_difficulty: str = "중"   # 하/중/상
    profile: dict = {}
    lang: str = "ko"

_SD_ORDER = {"하": 1, "중": 2, "상": 3}

@app.post("/leveltest/session-next")
def leveltest_session_next(req: SessionDiffReq):
    lang = norm_lang(req.lang)
    scores = req.scores if isinstance(req.scores, dict) else {}
    profile = req.profile if isinstance(req.profile, dict) else {}
    vals = []
    for v in scores.values():
        try:
            vals.append(max(0.0, min(100.0, float(v))))
        except (TypeError, ValueError):
            pass
    avg = round(sum(vals) / len(vals), 1) if vals else 0.0
    cur = _SD_ORDER.get(str(req.current_difficulty).strip(), 2)
    # 규칙: 세션 평균 70+ → 난이도 ↑, 50 미만 → ↓, 그 외 유지 (1~3 클램프)
    nxt = cur + (1 if avg >= 70 else (-1 if avg < 50 else 0))
    nxt = max(1, min(3, nxt))
    stats = {"level": _lt_level_label(avg / 100.0, nxt),
             "accuracy": round(avg / 100.0, 2),
             "difficulty_reached": _LT_DIFF_KO[cur],
             "by_topic": {}, "weak_topics": []}
    # 약점 축 = 가장 낮은 점수 축
    if scores:
        weak_axis = min(scores.items(), key=lambda kv: kv[1])[0] if vals else None
        stats["weak_axis"] = weak_axis
    nr = not_ready(lang)
    if nr:
        rec, src = _lt_fallback_rec(stats), "fallback"
    else:
        rec, src = _lt_ai_recommend(profile, stats, lang)
    return {"ok": True, "session_avg": avg,
            "current_difficulty": _LT_DIFF_KO[cur], "next_difficulty": _LT_DIFF_KO[nxt],
            "recommendation": rec, "recommendation_source": src}


# ============================================================================
#  학습 노트 생성 (AI Lesson)  — server.py 하단에 append
#  - 커리큘럼 지도의 노드(토픽)를 누르면 그 자리에서 EXAONE이 '핵심 학습 노트'를 생성
#  - RAG(면접 질문)로 '면접에서 실제로 묻는 개념' 중심으로 근거를 깔고, 범위는 요약으로 한정
#  - <thought> 추론은 건너뛰고 본문만 스트리밍 (자소서 스트림과 동일 방식)
#  - POST /education/lesson         : 비스트리밍 {ok, lesson, related}
#  - POST /education/lesson/stream  : SSE (token... -> done{lesson, related})
# ============================================================================

LESSON_PREFILL = {
    "ko": "먼저 이 주제에서 면접에 자주 나오는 핵심 개념과 흔한 실수를 추려, 학습자가 빠르게 이해할 수 있는 노트 구성을 생각하겠습니다. ",
    "en": "First, let me identify the core concepts and common pitfalls of this topic that interviews focus on, and plan a concise study note. ",
}

def lesson_prompt(lang, topic, difficulty, related):
    strong = [r for r in (related or []) if float(r.get("score", 0)) >= 0.60][:5]
    rel = "\n".join("- " + str(r.get("question", "")).strip() for r in strong)
    if not rel:
        rel = ("(관련 질문 없음)" if lang != "en" else "(none)")
    if lang == "en":
        return f"""You are a developer learning coach. Write a CONCISE study note on the topic below at '{difficulty}' level, for interview preparation.

Rules:
- Cover ONLY: core concepts, must-know key points, and common mistakes. This is a summary note, NOT a full tutorial.
- Focus ONLY on the [Topic] below. The related interview questions are just a hint for which concepts interviewers emphasize; do NOT cover anything unrelated to the [Topic] (e.g., other data structures or technologies).
- Do NOT invent fake APIs, version numbers, or statistics. If unsure, stay general and correct.
- Write in English, 2-4 short paragraphs (do not overuse markdown headers or long bullet dumps). About 250-350 words.

[Topic] {topic}
[Related interview questions]
{rel}"""
    return f"""당신은 개발자 학습 코치입니다. 아래 주제에 대해 난이도 '{difficulty}' 수준으로, 면접 준비용 '핵심 학습 노트'를 간결하게 작성하세요.

규칙:
- 다루는 범위: 핵심 개념 / 꼭 알아야 할 요점 / 흔히 하는 실수. 이것은 '요약 노트'이지 완전한 강의가 아닙니다.
- 오직 아래 [주제]에만 집중하세요. '관련 면접 질문'은 그 주제에서 어떤 개념이 면접에 자주 나오는지 참고하는 용도이며, [주제]와 직접 관련 없는 질문(다른 자료구조·다른 기술 등)은 절대 다루지 마세요.
- 사실이 아닌 API·버전·수치를 지어내지 마세요. 확실하지 않으면 일반적이고 정확하게 쓰세요.
- 한국어로, 2~4개의 짧은 단락으로 정리하세요(마크다운 헤더 남발 금지, 긴 불릿 나열 금지). 약 400~600자.

[주제] {topic}
[관련 면접 질문]
{rel}"""

def _lesson_related(topic, lang, k=6):
    """RAG: 토픽 관련 면접 질문 검색(근거+UI 표시용). 실패해도 빈 리스트로 진행."""
    try:
        if lang == "en" and "index_en" in M and "records_en" in M:
            index, records = M["index_en"], M["records_en"]
        else:
            index, records = M["index"], M["records"]
        scores, ids = index.search(embed_query(topic), k)
        return [{"question": records[i]["question"], "score": float(s)}
                for s, i in zip(scores[0], ids[0]) if i >= 0]
    except Exception as e:
        print(f">>> [lesson] RAG retrieval skipped: {e}", flush=True)
        return []

_LESSON_SPECIAL = ("[|endofturn|]", "[|assistant|]", "[|system|]", "[|user|]", "[|endoftext|]")
def _lesson_strip_special(s):
    for _t in _LESSON_SPECIAL:
        s = s.replace(_t, "")
    return s

def _lesson_body_pieces(chunks):
    """청크 스트림에서 </thought> 이후 '본문'만 조각으로 내보냄(추론은 숨김)."""
    passed = False
    buf = ""
    for chunk in chunks:
        if not chunk:
            continue
        if passed:
            piece = _lesson_strip_special(chunk)
            if piece:
                yield piece
        else:
            buf += chunk
            if "</thought>" in buf:
                passed = True
                after = _lesson_strip_special(buf.split("</thought>", 1)[1]).lstrip("\n")
                if after:
                    yield after

def _lesson_tee(it, sink):
    for c in it:
        sink.append(c)
        yield c

def _lesson_stream_gen(prompt, prefill, lang, related):
    from transformers import TextIteratorStreamer
    msgs = [{"role": "user", "content": prompt}]
    text = M["llm_tok"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill
    enc = M["llm_tok"](text, return_tensors="pt", add_special_tokens=False).to(M["llm"].device)
    plen = enc["input_ids"].shape[1]
    streamer = TextIteratorStreamer(M["llm_tok"], skip_prompt=True, skip_special_tokens=False)
    holder = {}
    def _gen():
        try:
            with torch.no_grad():
                with M["llm"].disable_adapter():     # 개념 설명은 base 모델
                    holder["out"] = M["llm"].generate(**enc, max_new_tokens=2048,
                                                       do_sample=False, streamer=streamer)
        except Exception as e:
            holder["err"] = e
    GEN_LOCK.acquire()
    t = threading.Thread(target=_gen, daemon=True)
    t0 = time.time()
    t.start()
    full = []
    try:
        for piece in _lesson_body_pieces(_lesson_tee(streamer, full)):
            yield "data: " + json.dumps({"type": "token", "text": piece}, ensure_ascii=False) + "\n\n"
        t.join()
        if "err" in holder:
            raise holder["err"]
        body = _strip_thought("".join(full))
        out = holder.get("out")
        if out is not None:
            gl = int(out.shape[1] - plen); dt = time.time() - t0
            print(f">>> [lesson-stream] {gl}tok / {dt:.1f}s = {gl/max(dt,1e-9):.1f} tok/s", flush=True)
        yield "data: " + json.dumps({"type": "done", "ok": True, "lesson": body, "related": related},
                                     ensure_ascii=False) + "\n\n"
    except Exception as e:
        yield "data: " + json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"},
                                    ensure_ascii=False) + "\n\n"
    finally:
        if t.is_alive():
            t.join()
        try:
            GEN_LOCK.release()
        except RuntimeError:
            pass


class LessonReq(BaseModel):
    topic: str
    difficulty: str = "중"     # 하 / 중 / 상
    lang: str = "ko"

@app.post("/education/lesson")
def education_lesson(req: LessonReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr:
        return nr
    err = vlen("topic", req.topic, lang)
    if err:
        return {"ok": False, "error": err}
    related = _lesson_related(req.topic, lang)
    prompt = lesson_prompt(lang, req.topic, req.difficulty, related)
    gen = run_llm(prompt, LESSON_PREFILL[lang], use_adapter=False, max_new_tokens=2048, do_sample=False)
    body = _strip_thought(gen)
    if body and len(body) >= 20:
        return {"ok": True, "topic": req.topic, "difficulty": req.difficulty,
                "lesson": body, "related": related}
    return {"ok": False, "error": "학습 노트 생성 실패", "raw": gen[-1000:]}

@app.post("/education/lesson/stream")
def education_lesson_stream(req: LessonReq):
    lang = norm_lang(req.lang)
    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    def _one(obj):
        def _g():
            yield "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
        return _g()
    nr = not_ready(lang)
    if nr:
        return StreamingResponse(_one({"type": "error", "error": nr["error"]}),
                                 media_type="text/event-stream", headers=sse_headers)
    err = vlen("topic", req.topic, lang)
    if err:
        return StreamingResponse(_one({"type": "error", "error": err}),
                                 media_type="text/event-stream", headers=sse_headers)
    related = _lesson_related(req.topic, lang)
    prompt = lesson_prompt(lang, req.topic, req.difficulty, related)
    return StreamingResponse(_lesson_stream_gen(prompt, LESSON_PREFILL[lang], lang, related),
                             media_type="text/event-stream", headers=sse_headers)


# ============================================================
#  챗봇 /chat — 하이브리드 RAG (FAQ 강매칭=캔드 / 부분매칭=LLM근거생성 / 폴백)
# ============================================================
FAQ_STRONG     = 0.80   # 이 이상: 캔드 답 그대로 (빠름·정확)
FAQ_WEAK       = 0.55   # 이 이상~STRONG 미만: top-3 FAQ 근거로 LLM 생성
INTERVIEW_HINT = 0.60   # FAQ 약하고 면접질문 이 이상이면 면접 유도

CHAT_LLM_PREFILL = {
    "ko": "사용자의 질문 의도를 먼저 파악하고, 제가 아는 내용을 바탕으로 도움이 되게 바로 답하겠습니다. ",
    "en": "Let me first understand what the user is asking and answer helpfully with what I know. ",
}

def _chat_llm_prompt(lang, msg, faqs):
    ctx = "\n".join(f"- (Q: {f.get('question','')}) {f.get('answer','')}" for f in faqs)
    if lang == "en":
        return (
            "You are DevReady's in-app assistant. Using the reference information below, "
            "answer the user's question naturally, as if you know it yourself.\n"
            "Rules:\n"
            "- Do not mention sources or use words like 'FAQ', 'the provided information', "
            "or 'reference'. Answer in your own voice.\n"
            "- Do not narrate your process or talk to yourself (e.g., 'let me check'). "
            "State only the conclusion.\n"
            "- Do not invent pricing, policies, or features not present in the reference. "
            "If you can't answer confidently, briefly apologize and point to where in DevReady "
            "they can find it (without saying 'FAQ').\n"
            "- For account- or transaction-specific matters (refunds, changing payment methods, "
            "personal billing/subscription records, legal interpretation of the terms), do not "
            "answer directly; direct them to My Page > Billing or customer support. General "
            "policy explanations (e.g., access continues until the paid period ends after "
            "cancellation) are fine to answer.\n"
            "- 2-4 sentences, conversational, no markdown.\n\n"
            f"[Reference]\n{ctx}\n\n[User question]\n{msg}"
        )
    return (
        "당신은 DevReady 서비스의 안내 챗봇입니다. 아래 참고 정보를 바탕으로, 사용자의 질문에 "
        "당신이 직접 아는 것처럼 자연스럽게 답하세요.\n"
        "규칙:\n"
        "- 'FAQ', '자료', '제공된 내용', '출처' 같은 표현을 쓰지 말고 출처를 언급하지 마세요. "
        "당신의 답으로 바로 말하세요.\n"
        "- 답을 찾는 과정이나 스스로에게 하는 말(예: '확인해보겠습니다', '살펴보겠습니다')을 쓰지 "
        "말고 결론만 답하세요.\n"
        "- 참고 정보에 없는 요금·정책·기능을 지어내지 마세요. 확실히 답하기 어려우면 짧게 사과하고, "
        "'FAQ' 언급 없이 관련 기능을 어디서 볼 수 있는지 자연스럽게 안내하세요.\n"
        "- 환불, 결제 수단 변경, 내 결제·구독 내역, 약관의 법적 해석처럼 개인 계정·거래에 관한 "
        "구체적 처리는 직접 답하지 말고 마이페이지 > 결제 정보나 고객센터로 안내하세요. 다만 "
        "'해지해도 남은 결제 기간까지는 이용할 수 있다'와 같은 일반적인 정책 설명은 답해도 됩니다.\n"
        "- 2~4문장, 자연스러운 대화체, 마크다운 없이 답하세요.\n\n"
        f"[참고 정보]\n{ctx}\n\n[사용자 질문]\n{msg}"
    )

@app.post("/chat")
def chat(req: ChatReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr:
        return nr
    msg = (req.message or "").strip()
    if not msg:
        return {"ok": False, "error": "메시지가 비어 있습니다. / Empty message."}
    if len(msg) > 1000:
        msg = msg[:1000]
    qv = embed_query(msg)

    # ① FAQ 검색 (top-3 — 근거 후보 확보)
    faq_hits = []
    if "faq_index" in M and M["faq_index"].ntotal > 0:
        k = min(3, M["faq_index"].ntotal)
        s, i = M["faq_index"].search(qv, k)
        for rank in range(k):
            faq_hits.append((float(s[0][rank]), int(i[0][rank])))
    top_score = faq_hits[0][0] if faq_hits else 0.0

    # ② 강매칭(>=0.80) → 캔드 답 그대로 (빠름)
    if faq_hits and top_score >= FAQ_STRONG:
        rec = M["faq_records"][faq_hits[0][1]]
        return {
            "ok": True, "source": "faq", "score": round(top_score, 4),
            "category": rec.get("category", ""),
            "answer": rec["answer"],
            "matched_question": rec["question"],
        }

    # ②' 부분매칭(0.55~0.80) → top-3 FAQ 근거로 LLM 생성
    if faq_hits and top_score >= FAQ_WEAK:
        ctx_recs = [M["faq_records"][idx] for _, idx in faq_hits]
        prompt = _chat_llm_prompt(lang, msg, ctx_recs)
        try:
            gen = run_llm(prompt, CHAT_LLM_PREFILL[lang], use_adapter=False,
                          max_new_tokens=1536, do_sample=False)
            answer = _strip_thought(gen).strip() if "</thought>" in gen else ""
        except Exception as e:
            print(f">>> [chat] LLM 생성 실패: {e}", flush=True)
            answer = ""
        if answer:
            top_rec = ctx_recs[0]
            return {
                "ok": True, "source": "faq_llm", "score": round(top_score, 4),
                "category": top_rec.get("category", ""),
                "answer": answer,
                "matched_question": top_rec.get("question", ""),
            }
        top_rec = ctx_recs[0]
        return {
            "ok": True, "source": "faq", "score": round(top_score, 4),
            "category": top_rec.get("category", ""),
            "answer": top_rec["answer"],
            "matched_question": top_rec.get("question", ""),
        }

    # ③ FAQ 약함 → 면접질문 RAG 확인 (기술 질문이면 면접 유도)
    iv_best = None
    if "index" in M and M["index"].ntotal > 0:
        s, i = M["index"].search(qv, 1)
        iv_best = (float(s[0][0]), int(i[0][0]))
    if iv_best and iv_best[0] >= INTERVIEW_HINT:
        rel = ""
        try:
            rel = M["records"][iv_best[1]].get("question", "")
        except Exception:
            rel = ""
        return {
            "ok": True, "source": "interview", "score": round(iv_best[0], 4),
            "answer": ("기술 면접 질문에 가까운 내용이네요. 모의 면접에서 직접 연습하면서 "
                       "AI 채점과 피드백을 받아보시는 걸 추천드려요. 면접 페이지에서 시작할 수 있습니다."),
            "related_question": rel,
        }

    # ④ 둘 다 실패 → 폴백
    return {
        "ok": True, "source": "none",
        "answer": ("죄송해요, 그 질문은 제가 정확히 답하기 어려워요. "
                   "DevReady 서비스 이용 방법(회원가입, 이력서, 모의 면접, 학습, 결제 등)에 대해 "
                   "물어봐 주시면 안내해 드릴게요."),
    }


# ============================================================
#  챗봇 /chat/stream — SSE 스트리밍 (부분매칭 LLM만 실시간 타이핑)
# ============================================================
def _chat_tee(it, sink):
    for c in it:
        sink.append(c)
        yield c

_CHAT_SPECIAL = re.compile(r"\[\|[^|]*\|\]|<\|[^|]*\|>")

def _chat_clean(p):
    return _CHAT_SPECIAL.sub("", p)

def _chat_body_pieces(it):
    # </thought> 이전(사고)은 버퍼링, 이후(실답변)만 흘림 + 특수토큰([|...|]/<|...|>) 제거 + 빈조각 skip
    seen = False
    buf = ""
    for c in it:
        if seen:
            c2 = _chat_clean(c)
            if c2:
                yield c2
            continue
        buf += c
        idx = buf.find("</thought>")
        if idx != -1:
            after = _chat_clean(buf[idx + len("</thought>"):])
            seen = True
            if after:
                yield after

def _chat_stream_gen(msg, faqs, lang, score, top_rec):
    from transformers import TextIteratorStreamer
    prompt = _chat_llm_prompt(lang, msg, faqs)
    prefill = CHAT_LLM_PREFILL[lang]
    msgs = [{"role": "user", "content": prompt}]
    text = M["llm_tok"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill
    enc = M["llm_tok"](text, return_tensors="pt", add_special_tokens=False).to(M["llm"].device)
    plen = enc["input_ids"].shape[1]
    streamer = TextIteratorStreamer(M["llm_tok"], skip_prompt=True, skip_special_tokens=False)
    holder = {}
    def _gen():
        try:
            with torch.no_grad():
                with M["llm"].disable_adapter():     # 챗봇 답변은 base 모델
                    holder["out"] = M["llm"].generate(**enc, max_new_tokens=1536,
                                                       do_sample=False, streamer=streamer)
        except Exception as e:
            holder["err"] = e
    GEN_LOCK.acquire()
    t = threading.Thread(target=_gen, daemon=True)
    t0 = time.time()
    t.start()
    full = []
    try:
        for piece in _chat_body_pieces(_chat_tee(streamer, full)):
            yield "data: " + json.dumps({"type": "token", "text": piece}, ensure_ascii=False) + "\n\n"
        t.join()
        if "err" in holder:
            raise holder["err"]
        _raw = "".join(full)
        answer = _strip_thought(_raw).strip() if "</thought>" in _raw else ""
        out = holder.get("out")
        if out is not None:
            gl = int(out.shape[1] - plen); dt = time.time() - t0
            print(f">>> [chat-stream] {gl}tok / {dt:.1f}s = {gl/max(dt,1e-9):.1f} tok/s", flush=True)
        if answer:
            src = "faq_llm"
        else:
            answer = top_rec["answer"]   # 생성 실패 → 최상위 FAQ 캔드 폴백
            src = "faq"
        yield "data: " + json.dumps({"type": "done", "ok": True, "source": src,
                                     "score": round(score, 4),
                                     "category": top_rec.get("category", ""),
                                     "answer": answer,
                                     "matched_question": top_rec.get("question", "")},
                                    ensure_ascii=False) + "\n\n"
    except Exception as e:
        yield "data: " + json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"},
                                    ensure_ascii=False) + "\n\n"
    finally:
        if t.is_alive():
            t.join()
        try:
            GEN_LOCK.release()
        except RuntimeError:
            pass

@app.post("/chat/stream")
def chat_stream(req: ChatReq):
    lang = norm_lang(req.lang)
    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    def _one(obj):
        def _g():
            yield "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
        return _g()
    nr = not_ready(lang)
    if nr:
        return StreamingResponse(_one({"type": "error", "error": nr["error"]}),
                                 media_type="text/event-stream", headers=sse_headers)
    msg = (req.message or "").strip()
    if not msg:
        return StreamingResponse(_one({"type": "error", "error": "메시지가 비어 있습니다. / Empty message."}),
                                 media_type="text/event-stream", headers=sse_headers)
    if len(msg) > 1000:
        msg = msg[:1000]
    qv = embed_query(msg)

    # ① FAQ 검색 (top-3)
    faq_hits = []
    if "faq_index" in M and M["faq_index"].ntotal > 0:
        k = min(3, M["faq_index"].ntotal)
        s, i = M["faq_index"].search(qv, k)
        for rank in range(k):
            faq_hits.append((float(s[0][rank]), int(i[0][rank])))
    top_score = faq_hits[0][0] if faq_hits else 0.0

    # ② 강매칭(>=0.80) → done 즉시 (캔드, 생성 없음)
    if faq_hits and top_score >= FAQ_STRONG:
        rec = M["faq_records"][faq_hits[0][1]]
        return StreamingResponse(_one({"type": "done", "ok": True, "source": "faq",
                                       "score": round(top_score, 4),
                                       "category": rec.get("category", ""),
                                       "answer": rec["answer"],
                                       "matched_question": rec["question"]}),
                                 media_type="text/event-stream", headers=sse_headers)

    # ②' 부분매칭(0.55~0.80) → LLM 스트리밍 (token → done)
    if faq_hits and top_score >= FAQ_WEAK:
        ctx_recs = [M["faq_records"][idx] for _, idx in faq_hits]
        return StreamingResponse(_chat_stream_gen(msg, ctx_recs, lang, top_score, ctx_recs[0]),
                                 media_type="text/event-stream", headers=sse_headers)

    # ③ 면접유도 → done 즉시
    iv_best = None
    if "index" in M and M["index"].ntotal > 0:
        s, i = M["index"].search(qv, 1)
        iv_best = (float(s[0][0]), int(i[0][0]))
    if iv_best and iv_best[0] >= INTERVIEW_HINT:
        rel = ""
        try:
            rel = M["records"][iv_best[1]].get("question", "")
        except Exception:
            rel = ""
        return StreamingResponse(_one({"type": "done", "ok": True, "source": "interview",
                                       "score": round(iv_best[0], 4),
                                       "answer": ("기술 면접 질문에 가까운 내용이네요. 모의 면접에서 직접 연습하면서 "
                                                  "AI 채점과 피드백을 받아보시는 걸 추천드려요. 면접 페이지에서 시작할 수 있습니다."),
                                       "related_question": rel}),
                                 media_type="text/event-stream", headers=sse_headers)

    # ④ 폴백 → done 즉시
    return StreamingResponse(_one({"type": "done", "ok": True, "source": "none",
                                   "answer": ("죄송해요, 그 질문은 제가 정확히 답하기 어려워요. "
                                              "DevReady 서비스 이용 방법(회원가입, 이력서, 모의 면접, 학습, 결제 등)에 대해 "
                                              "물어봐 주시면 안내해 드릴게요.")}),
                             media_type="text/event-stream", headers=sse_headers)




# ===== /recommend/explain =====
EXPLAIN_PREFILL = {
    "ko": "먼저 각 대상이 사용자 프로필과 어떻게 맞닿아 있는지 살펴보겠습니다. ",
    "en": "First, let me review how each item matches the user profile. ",
}

class ExplainItem(BaseModel):
    target_id: int
    title: str
    category: str = ""
    difficulty: str = ""
    matched: list = []

class ExplainReq(BaseModel):
    target_type: str = "COURSE"
    profile: dict = {}
    items: list = []
    lang: str = "ko"

@app.post("/recommend/explain")
def recommend_explain(req: ExplainReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr:
        return nr
    if not req.items:
        return {"ok": False, "error": "items is empty"}

    target_type = (req.target_type or "COURSE").upper()
    profile = req.profile or {}
    job_category = profile.get("job_category", "")
    level = profile.get("level", "")
    skills = profile.get("skills", [])
    skills_str = ", ".join(skills) if skills else "-"

    # 아이템 목록 직렬화 (프롬프트용)
    lines = []
    for it in req.items:
        if isinstance(it, dict):
            tid = it.get("target_id", "")
            title = it.get("title", "")
            cat = it.get("category", "")
            diff = it.get("difficulty", "")
            matched = it.get("matched", [])
        else:
            tid = getattr(it, "target_id", "")
            title = getattr(it, "title", "")
            cat = getattr(it, "category", "")
            diff = getattr(it, "difficulty", "")
            matched = getattr(it, "matched", [])
        matched_str = ", ".join(matched) if matched else "-"
        lines.append(f"- id:{tid} | {title} | 직군:{cat} | 난이도:{diff} | 매칭근거:{matched_str}")
    items_block = "\n".join(lines)

    if lang == "en":
        prompt = f"""You are a career advisor. For each item below, write a 1-2 sentence explanation of why it suits this user. Base your explanation strictly on the matched signals provided — do not invent content.

User profile: job_category={job_category}, level={level}, skills=[{skills_str}]
Target type: {target_type}

Items:
{items_block}

Rules:
- One entry per item, keyed by id
- 1-2 sentences each, natural English
- Ground strictly in matched signals
- After your thinking, output ONLY this JSON:
  {{"reasons": [{{"target_id": <id>, "reason": "<1-2 sentences>"}}]}}"""
    else:
        prompt = f"""당신은 커리어 어드바이저입니다. 아래 각 항목이 이 사용자에게 왜 맞는지 1~2문장으로 설명하세요. 반드시 제공된 매칭 근거에만 기반하고, 없는 내용을 지어내지 마세요.

사용자 프로필: 직군={job_category}, 레벨={level}, 보유스킬=[{skills_str}]
추천 대상 유형: {target_type}

대상 목록:
{items_block}

규칙:
- 항목마다 target_id로 키잉
- 각 1~2문장, 자연스러운 한국어
- 매칭 근거에만 근거할 것
- 사고 과정을 마친 뒤, 마지막에 JSON만 출력:
  {{"reasons": [{{"target_id": <id>, "reason": "<1~2문장>"}}]}}"""

    gen = run_llm(prompt, EXPLAIN_PREFILL[lang], use_adapter=False,
                  max_new_tokens=2048, do_sample=True)
    d = parse_json_lenient(gen)
    if d and "reasons" in d and isinstance(d["reasons"], list):
        return {"ok": True, "target_type": target_type, "reasons": d["reasons"]}
    return {"ok": False, "target_type": target_type, "reasons": [],
            "error": "parse_failed", "raw": gen[-1500:]}


# ===== Quiz SSE streaming - /education/quiz/stream (mirrors evaluate/stream) =====
def _quiz_stream_gen(prompt, prefill, lang, n, topic):
    from transformers import TextIteratorStreamer
    msgs = [{"role": "user", "content": prompt}]
    text = M["llm_tok"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill
    enc = M["llm_tok"](text, return_tensors="pt", add_special_tokens=False).to(M["llm"].device)
    plen = enc["input_ids"].shape[1]
    streamer = TextIteratorStreamer(M["llm_tok"], skip_prompt=True, skip_special_tokens=False)
    holder = {}
    def _gen():
        try:
            with torch.no_grad():
                with M["llm"].disable_adapter():  # quiz always uses base model
                    holder["out"] = M["llm"].generate(
                        **enc, max_new_tokens=3072, do_sample=True,
                        temperature=0.8, top_p=0.9, streamer=streamer)
        except Exception as e:
            holder["err"] = e
    GEN_LOCK.acquire()  # same lock as non-stream, serialize GPU
    t = threading.Thread(target=_gen, daemon=True)
    t0 = time.time()
    t.start()
    full = []
    try:
        for chunk in streamer:
            if not chunk:
                continue
            full.append(chunk)
            piece = chunk
            for _tok in ("[|endofturn|]", "[|assistant|]", "[|system|]", "[|user|]", "[|endoftext|]"):
                piece = piece.replace(_tok, "")
            if piece:
                yield "data: " + json.dumps({"type": "token", "text": piece}, ensure_ascii=False) + "\n\n"
        t.join()
        if "err" in holder:
            raise holder["err"]
        gen_text = "".join(full)
        out = holder.get("out")
        if out is not None:
            gl = int(out.shape[1] - plen); dt = time.time() - t0
            print(f">>> [quiz-stream] {gl}tok / {dt:.1f}s = {gl/max(dt,1e-9):.1f} tok/s (cap 3072, adapter=False)", flush=True)
        parsed = parse_json_lenient(gen_text)
        items = _build_quiz_items(parsed, n)
        if items:
            payload = {"type": "done", "ok": True, "topic": topic, "count": len(items), "quiz": items}
        else:
            payload = {"type": "done", "ok": False, "error": MSG[lang]["gen_fail"], "raw": gen_text[-1500:]}
        yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
    except Exception as e:
        yield "data: " + json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False) + "\n\n"
    finally:
        if t.is_alive():
            t.join()  # wait for generate to finish before releasing lock
        try:
            GEN_LOCK.release()
        except RuntimeError:
            pass


@app.post("/education/quiz/stream")
def education_quiz_stream(req: QuizReq):
    lang = norm_lang(req.lang)
    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    def _one(obj):
        def _g():
            yield "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
        return _g()
    nr = not_ready(lang)
    if nr:
        return StreamingResponse(_one({"type": "error", "error": nr["error"]}), media_type="text/event-stream", headers=sse_headers)
    err = vlen("topic", req.topic, lang)
    if err:
        return StreamingResponse(_one({"type": "error", "error": err}), media_type="text/event-stream", headers=sse_headers)
    n = max(1, min(10, req.n))
    prompt = quiz_prompt(lang, req.topic, n, req.difficulty)
    return StreamingResponse(_quiz_stream_gen(prompt, QUIZ_PREFILL[lang], lang, n, req.topic), media_type="text/event-stream", headers=sse_headers)


# ── 신고 AI 1차 판정(A-017) — POST /report/judge (비동기 호출, 120s) ──────────
REPORT_JUDGE_PREFILL = {
    "ko": "먼저 신고 대상 콘텐츠를 커뮤니티 규칙에 비추어 살펴보고, 위반 가능성을 상/중/하로 판단한 뒤 근거를 정리하겠습니다. ",
    "en": "First, I will assess the reported content against community rules and decide the violation likelihood. ",
}
GRADE_NORMALIZE = {  # LLM이 등급을 다른 말로 낼 때 흡수(최종은 {상,중,하}로만)
    "상": "상", "중": "중", "하": "하",
    "높음": "상", "중간": "중", "낮음": "하",
    "high": "상", "medium": "중", "low": "하",
}

class ReportJudgeReq(BaseModel):
    targetType: str = None
    content: str = None
    lang: str = "ko"

@app.post("/report/judge")
def report_judge(req: ReportJudgeReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr:                                  # 미로딩 → ok:false → 소비자 NULL(미판정) 저장
        return nr
    content = (req.content or "").strip()
    if not content:
        return {"ok": False, "error": MSG[lang]["gen_fail"]}
    snippet = content[:2000]                # 긴 글은 잘라 넣되 거부하지 않음(생성시간 상한)
    ttype = "댓글" if str(req.targetType).upper() == "COMMENT" else "게시글"
    prompt = (
        f"당신은 커뮤니티 신고 검토 보조 AI입니다. 아래 {ttype} 내용이 커뮤니티 규칙을 "
        f"위반할 가능성을 판정하세요.\n"
        f"위반 예: 욕설/모욕, 스팸/광고, 음란물, 개인정보 노출, 혐오/차별.\n\n"
        f"[{ttype} 내용]\n{snippet}\n\n"
        f"위반 가능성을 '상'(높음)/'중'(중간)/'하'(낮음) 중 하나로 정하고, 한국어로 한 문장 "
        f"사유를 쓰세요. JSON만 출력하세요.\n형식: {{\"judgment\": \"상\", \"reason\": \"...\"}}"
    )
    try:
        gen = run_llm(prompt, REPORT_JUDGE_PREFILL[lang], use_adapter=False,
                      max_new_tokens=512, reason_budget=400)
        data = parse_json_lenient(gen)
        if not isinstance(data, dict):
            return {"ok": False, "error": MSG[lang]["gen_fail"], "raw": gen[-1500:]}
        raw_j = str(data.get("judgment", "")).strip()
        judgment = GRADE_NORMALIZE.get(raw_j) or GRADE_NORMALIZE.get(raw_j.lower())
        if judgment not in ("상", "중", "하"):          # clamp 실패 → NULL(미판정)
            return {"ok": False, "error": MSG[lang]["gen_fail"], "raw": gen[-1500:]}
        reason = str(data.get("reason", "")).strip()[:500]
        return {"ok": True, "judgment": judgment, "reason": reason}
    except Exception:
        return {"ok": False, "error": MSG[lang]["gen_fail"]}


# ── 콘텐츠 유해도 2차 필터(A-008) — POST /content/filter (매 작성, 8s 핫패스) ──
CONTENT_FILTER_PREFILL = {
    "ko": "먼저 콘텐츠의 유해성 정도를 커뮤니티 기준으로 살펴보고, 유해/경미/정상 중 하나로 분류한 뒤 근거를 정리하겠습니다. ",
    "en": "First, I will assess the content's harmfulness and classify it. ",
}
# LLM은 분류만(의미). 점수는 결정론적 매핑 → 항상 DECIMAL(5,2)·CHECK(0~100) 안, 유해(90)만 ≥75 차단.
CONTENT_LEVEL_SCORE = {"유해": 90.00, "경미": 45.00, "정상": 5.00}
CONTENT_LEVEL_NORMALIZE = {
    "유해": "유해", "경미": "경미", "정상": "정상",
    "harmful": "유해", "toxic": "유해", "mild": "경미", "clean": "정상", "safe": "정상",
}

class ContentFilterReq(BaseModel):
    targetType: str = None
    content: str = None
    lang: str = "ko"

@app.post("/content/filter")
def content_filter(req: ContentFilterReq):
    lang = norm_lang(req.lang)
    nr = not_ready(lang)
    if nr:                                  # 미로딩 → ok:false → 소비자 fail-open(글쓰기 통과)
        return nr
    content = (req.content or "").strip()
    if not content:
        return {"ok": False, "error": MSG[lang]["gen_fail"]}
    snippet = content[:2000]
    prompt = (
        f"당신은 커뮤니티 콘텐츠 검열 보조 AI입니다. 아래 내용의 유해성을 판단하세요.\n"
        f"유해 예: 심한 욕설/모욕, 혐오/차별, 음란물, 명백한 스팸/광고.\n"
        f"명백히 유해하면 '유해', 애매하면 '경미', 문제없으면 '정상'으로 분류하세요.\n\n"
        f"[내용]\n{snippet}\n\n"
        f"한국어로 한 문장 사유를 쓰세요. JSON만 출력하세요.\n"
        f"형식: {{\"level\": \"정상\", \"reason\": \"...\"}}"
    )
    try:
        gen = run_llm(prompt, CONTENT_FILTER_PREFILL[lang], use_adapter=False,
                      max_new_tokens=256, reason_budget=200)     # 8s 핫패스 → 예산 축소
        data = parse_json_lenient(gen)
        if not isinstance(data, dict):
            return {"ok": False, "error": MSG[lang]["gen_fail"], "raw": gen[-1500:]}
        raw_l = str(data.get("level", "")).strip()
        level = CONTENT_LEVEL_NORMALIZE.get(raw_l) or CONTENT_LEVEL_NORMALIZE.get(raw_l.lower())
        if level not in CONTENT_LEVEL_SCORE:
            return {"ok": False, "error": MSG[lang]["gen_fail"], "raw": gen[-1500:]}
        score = CONTENT_LEVEL_SCORE[level]              # 결정론적, 항상 유효
        reason = str(data.get("reason", "")).strip()[:500]
        return {"ok": True, "score": score, "reason": reason}
    except Exception:
        return {"ok": False, "error": MSG[lang]["gen_fail"]}
