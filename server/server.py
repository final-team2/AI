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
from fastapi.responses import JSONResponse
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
TRAIN     = "/workspace/interview_ai/train.jsonl"

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
    "ko": "먼저 지원자의 답변에서 더 깊이 확인할 부분을 살펴보겠습니다. ",
    "en": "First, let me look at what in the candidate's answer is worth probing further. ",
}
RP_PREFILL = {
    "ko": "먼저 지원자의 면접 결과 전체를 살펴보겠습니다. ",
    "en": "First, let me review the candidate's overall interview results. ",
}

# ---- 영어 채점 rubric (한국어는 train.jsonl에서 M['rubric']로 로드) ----
RUBRIC_EN = """You are a web-developer hiring interviewer. Evaluate the [Candidate Answer] to the [Interview Question].

First decide whether this is a 'technical question' or a 'behavioral / experience / motivation question' (collaboration, conflict, motivation, team fit, strengths/weaknesses). For behavioral questions do NOT demand technical knowledge or say "a code/technical approach is needed"; assess whether the attitude, experience, and reasoning are sound and convincing.

Score each of the four axes as an INTEGER from 0 to 100. This is a 0-100 scale, NOT a 0-10 scale. A strong answer is about 85 (NOT 8); an average answer is about 55 (NOT 5); a weak answer is about 25. Be discriminating: vague or hedging answers ("I'm not sure, maybe...") should score low on specificity and technical_accuracy.
- technical_accuracy: correctness and validity of the content
- specificity: concreteness and depth
- logic: logical structure
- communication: clarity of delivery
Bands: excellent 80-95 / good 60-75 / fair 40-55 / weak 15-35 / very weak 0-10.

Think concisely in English. Do NOT write any JSON inside your thinking. After thinking, output ONLY one JSON object (all content in English), exactly in this shape:
{"scores":{"technical_accuracy":85,"specificity":70,"logic":80,"communication":75},"strengths":["..."],"improvements":["..."],"feedback":"..."}

Worked example (for format and scale calibration only):
[Interview Question] Explain what an index is in a database.
[Candidate Answer] An index is like a book's table of contents; it lets the database find rows without scanning the whole table, speeding up reads but slightly slowing writes.
{"scores":{"technical_accuracy":82,"specificity":68,"logic":80,"communication":85},"strengths":["Clear analogy","Notes the read/write trade-off"],"improvements":["Could mention B-tree structure or which columns to index"],"feedback":"Accurate and well-communicated; add concrete detail on index internals to score higher."}"""

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

SCORE_KEYS = ["technical_accuracy", "specificity", "logic", "communication"]
LABEL = {
    "ko": {"technical_accuracy": "기술 정확도", "specificity": "답변 구체성", "logic": "논리성", "communication": "의사소통"},
    "en": {"technical_accuracy": "Technical accuracy", "specificity": "Specificity", "logic": "Logic", "communication": "Communication"},
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

def run_llm(prompt, prefill, use_adapter=True, max_new_tokens=2048):
    msgs = [{"role": "user", "content": prompt}]
    text = M["llm_tok"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill
    enc = M["llm_tok"](text, return_tensors="pt", add_special_tokens=False).to(M["llm"].device)
    plen = enc["input_ids"].shape[1]
    with torch.no_grad():
        if use_adapter:
            o = M["llm"].generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            with M["llm"].disable_adapter():
                o = M["llm"].generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
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

def fu_prompt(lang, intro, question, answer):
    if lang == "en":
        return f"""{intro} Below are an interview question and the candidate's answer. Create ONE follow-up question that probes the candidate's answer more deeply.

Rules:
- Point at something vague or worth clarifying in the answer
- Ground it in what the candidate actually said (do not invent content)
- One clear sentence, in English
- After your thinking, output ONLY this JSON: {{"followup": "follow-up question"}}

[Interview Question]
{question}

[Candidate Answer]
{answer}"""
    return f"""{intro} 아래는 면접 질문과 지원자의 답변입니다. 지원자의 답변을 더 깊이 파고드는 후속(꼬리) 질문 1개를 만드세요.

규칙:
- 답변에서 모호하거나 더 구체적으로 확인할 수 있는 지점을 짚을 것
- 지원자가 실제로 한 말에 근거할 것 (없는 내용을 지어내지 말 것)
- 한 문장으로 명확하게, 한국어로 작성
- 사고 과정을 마친 뒤, 마지막에 JSON만 출력: {{"followup": "꼬리질문"}}

[면접 질문]
{question}

[지원자 답변]
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
    use_adapter = (lang == "ko")     # 한국어=형식 학습된 어댑터 / 영어=base + 강화 rubric
    gen = run_llm(prompt, PREFILL[lang], use_adapter=use_adapter, max_new_tokens=4096)
    ev = parse_json_lenient(gen)
    if ev and isinstance(ev.get("scores"), dict):
        ev["scores"] = clamp_scores(ev["scores"])
        ev["overall"] = fix_overall(ev["scores"])
        ev["display_scores"] = to_question_scores(ev["scores"])
        return {"ok": True, "evaluation": ev}
    return {"ok": False, "error": MSG[lang]["eval_fail"], "raw": gen[-1500:]}

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
    gen = run_llm(prompt, GEN_PREFILL[lang], use_adapter=False, max_new_tokens=2048)
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
    prompt = fu_prompt(lang, intro, req.question, req.answer)
    gen = run_llm(prompt, FU_PREFILL[lang], use_adapter=False, max_new_tokens=1536)
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
    gen = run_llm(prompt, QUIZ_PREFILL[lang], use_adapter=False, max_new_tokens=3072)
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
    gen = run_llm(prompt, CL_PREFILL[lang], use_adapter=False, max_new_tokens=2560)
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
    gen = run_llm(prompt, POLISH_PREFILL[lang], use_adapter=False, max_new_tokens=1024)
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
    gen = run_llm(prompt, POSTING_GEN_PREFILL[lang], use_adapter=False, max_new_tokens=2560)
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
