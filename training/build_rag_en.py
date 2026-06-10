"""
build_rag_en.py — 영어 RAG 질문 은행 구축 (큐레이션 방식)

- 큐레이션된 영어 IT 면접 질문(역량 카테고리 태깅) -> bge-m3 임베딩 -> FAISS IndexFlatIP
- server.py 의 embed_query 와 동일한 임베딩 레시피(CLS 풀링 + L2 정규화)를 사용해야
  서버의 /interview/question?lang=en 검색이 일관되게 동작함.
- 출력: rag/ict_questions_en.index , rag/ict_questions_en.json (list of {"question","category","topic"})
- bge-m3 만 로드(EXAONE 불필요) -> 서버 켜둔 채 실행해도 VRAM 여유. 실행 후 서버 재시작 필요.

카테고리(사진 역량 프레임): technical, analytical, communication, collaboration, learning_agility
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModel

RAG_DIR   = "/workspace/interview_ai/rag"
EMB_MODEL = "BAAI/bge-m3"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
OUT_INDEX = os.path.join(RAG_DIR, "ict_questions_en.index")
OUT_JSON  = os.path.join(RAG_DIR, "ict_questions_en.json")

# =====================================================================================
# 큐레이션 질문 세트 — 줄만 추가하면 은행이 커집니다. (category, topic) 는 보조 메타.
# =====================================================================================
QUESTIONS = [
    # ---------- Algorithms & Problem Solving (technical / analytical) ----------
    ("analytical", "algorithms", "Explain the difference between O(n) and O(log n) time complexity, with an example of each."),
    ("technical",  "algorithms", "When would you choose a hash map over a balanced binary search tree?"),
    ("technical",  "algorithms", "Describe how you would detect a cycle in a linked list."),
    ("analytical", "algorithms", "What is the trade-off between time and space complexity, and when have you optimized one at the cost of the other?"),
    ("technical",  "algorithms", "Explain the difference between depth-first and breadth-first search and when each is preferable."),
    ("technical",  "algorithms", "How would you find the k largest elements in a very large array efficiently?"),
    ("analytical", "algorithms", "What is dynamic programming, and how do you recognize a problem that can use it?"),
    ("analytical", "algorithms", "Describe a problem you solved by improving a brute-force solution into a more efficient one."),
    ("technical",  "algorithms", "Explain how a hash collision is handled and why it matters for performance."),
    ("technical",  "algorithms", "What is the difference between a stack and a queue, and give a real use case for each?"),
    ("technical",  "algorithms", "How would you reverse a linked list in place?"),
    ("technical",  "algorithms", "Explain the concept of recursion and a case where an iterative solution is better."),
    ("analytical", "algorithms", "What sorting algorithm would you use for nearly-sorted data, and why?"),
    ("analytical", "algorithms", "How do you approach a coding problem you have never seen before?"),
    ("technical",  "algorithms", "Describe the difference between an array and a linked list in terms of access and insertion cost."),
    ("analytical", "algorithms", "What is amortized time complexity, and where have you seen it apply?"),
    ("technical",  "algorithms", "How would you detect and handle duplicate entries in a large dataset?"),
    ("analytical", "algorithms", "Explain the difference between greedy and dynamic-programming approaches with an example."),
    ("technical",  "algorithms", "What is a binary heap and what operations is it good for?"),
    ("analytical", "algorithms", "How would you design an algorithm to find the shortest path in a weighted graph?"),
    ("analytical", "algorithms", "When you optimize code, how do you decide what to optimize first?"),
    ("analytical", "algorithms", "Explain Big-O, Big-Theta, and Big-Omega and how they differ."),
    ("technical",  "algorithms", "How would you check whether two strings are anagrams efficiently?"),
    ("analytical", "algorithms", "Describe a time you had to reason through a tricky edge case in your logic."),
    ("analytical", "algorithms", "What does it mean for a sorting algorithm to be stable, and when does stability matter?"),

    # ---------- Databases (technical) ----------
    ("technical", "databases", "What is a database index, and how does it speed up reads while affecting writes?"),
    ("technical", "databases", "Explain the difference between SQL and NoSQL databases and when you would choose each."),
    ("technical", "databases", "What are ACID properties, and why do they matter for transactions?"),
    ("technical", "databases", "Describe database normalization and a case where you might denormalize."),
    ("technical", "databases", "What is the difference between an inner join and a left join?"),
    ("technical", "databases", "How would you diagnose and fix a slow SQL query?"),
    ("technical", "databases", "Explain the difference between a primary key and a foreign key."),
    ("technical", "databases", "What is a deadlock in a database, and how can it be prevented?"),
    ("technical", "databases", "When would you use a composite index, and how does column order matter?"),
    ("technical", "databases", "Explain the difference between optimistic and pessimistic locking."),
    ("technical", "databases", "What is database sharding, and what problems does it introduce?"),
    ("technical", "databases", "How do transactions handle concurrent access to the same row?"),
    ("technical", "databases", "What is the N+1 query problem, and how do you avoid it?"),
    ("technical", "databases", "Explain the difference between a clustered and a non-clustered index."),
    ("technical", "databases", "How would you design a schema for a simple e-commerce order system?"),
    ("technical", "databases", "What is eventual consistency, and when is it acceptable?"),
    ("technical", "databases", "How do you decide between storing data as JSON versus normalized tables?"),
    ("technical", "databases", "What strategies do you use to back up and restore a production database safely?"),
    ("technical", "databases", "Explain what a transaction isolation level is and name one trade-off it controls."),
    ("technical", "databases", "How would you migrate a large table schema with minimal downtime?"),

    # ---------- Networking & Operating Systems (technical) ----------
    ("technical", "networking_os", "Explain the difference between a process and a thread."),
    ("technical", "networking_os", "What is the difference between TCP and UDP, and when would you use each?"),
    ("technical", "networking_os", "Describe what happens, step by step, when you type a URL into a browser and press enter."),
    ("technical", "networking_os", "What is a race condition, and how do you prevent it?"),
    ("technical", "networking_os", "Explain the difference between concurrency and parallelism."),
    ("technical", "networking_os", "What is a deadlock, and what conditions are required for one to occur?"),
    ("technical", "networking_os", "Describe the difference between a mutex and a semaphore."),
    ("technical", "networking_os", "What is the role of DNS, and what happens when a lookup fails?"),
    ("technical", "networking_os", "Explain the difference between HTTP and HTTPS."),
    ("technical", "networking_os", "What is the difference between stack memory and heap memory?"),
    ("technical", "networking_os", "How does an operating system decide which process to run next?"),
    ("technical", "networking_os", "What is a context switch, and why does it have a cost?"),
    ("technical", "networking_os", "Explain the difference between blocking and non-blocking I/O."),
    ("technical", "networking_os", "What do HTTP status codes mean, and when do 4xx versus 5xx apply?"),
    ("technical", "networking_os", "What is the benefit of a thread pool over creating threads on demand?"),
    ("technical", "networking_os", "Describe how a TCP three-way handshake works."),
    ("technical", "networking_os", "What is virtual memory, and why is it useful?"),
    ("technical", "networking_os", "Explain the difference between synchronous and asynchronous communication."),
    ("technical", "networking_os", "What is the difference between a port and an IP address?"),
    ("analytical", "networking_os", "How would you debug a service that intermittently becomes unresponsive?"),

    # ---------- Web / Backend / System Design (technical) ----------
    ("technical",  "web_backend", "What are the core principles of a RESTful API?"),
    ("technical",  "web_backend", "Explain the difference between authentication and authorization."),
    ("analytical", "web_backend", "How would you design a URL-shortening service?"),
    ("technical",  "web_backend", "What is caching, and what are the risks of serving stale data?"),
    ("technical",  "web_backend", "Explain the difference between horizontal and vertical scaling."),
    ("technical",  "web_backend", "What is a load balancer, and what strategies can it use to distribute traffic?"),
    ("technical",  "web_backend", "How do you implement rate limiting in an API?"),
    ("technical",  "web_backend", "What is the difference between a monolith and a microservices architecture?"),
    ("analytical", "web_backend", "How would you design an API so it stays backward compatible as it evolves?"),
    ("technical",  "web_backend", "What is idempotency, and why does it matter for API design?"),
    ("technical",  "web_backend", "Explain the difference between PUT and PATCH in HTTP."),
    ("technical",  "web_backend", "How would you handle very large file uploads in a web service?"),
    ("technical",  "web_backend", "What is a message queue, and when would you introduce one?"),
    ("technical",  "web_backend", "How do you secure secrets and credentials in a backend application?"),
    ("technical",  "web_backend", "What is the purpose of a reverse proxy?"),
    ("analytical", "web_backend", "How would you approach designing a system to handle a sudden traffic spike?"),
    ("technical",  "web_backend", "What is the difference between server-side and client-side rendering?"),
    ("analytical", "web_backend", "How do you ensure data consistency across multiple services?"),
    ("technical",  "web_backend", "What is a webhook, and how is it different from polling?"),
    ("analytical", "web_backend", "How would you design pagination for an API returning millions of records?"),

    # ---------- Security & Code Quality (technical) ----------
    ("technical", "security_quality", "What is SQL injection, and how do you prevent it?"),
    ("technical", "security_quality", "Explain the difference between hashing and encryption."),
    ("technical", "security_quality", "What is cross-site scripting (XSS), and how do you mitigate it?"),
    ("technical", "security_quality", "How do you safely store user passwords?"),
    ("technical", "security_quality", "What is the principle of least privilege, and how do you apply it?"),
    ("collaboration", "security_quality", "How do you approach reviewing another developer's code?"),
    ("technical", "security_quality", "What makes a good unit test, and what should it avoid?"),
    ("technical", "security_quality", "What is the difference between unit, integration, and end-to-end tests?"),
    ("technical", "security_quality", "How do you decide what to log and what not to log?"),
    ("technical", "security_quality", "What is a CSRF attack, and how is it prevented?"),
    ("technical", "security_quality", "How do you handle error cases gracefully in production code?"),
    ("analytical", "security_quality", "What is technical debt, and how do you decide when to pay it down?"),
    ("technical", "security_quality", "How do you keep your code maintainable for the next developer?"),
    ("collaboration", "security_quality", "What is the purpose of code review beyond catching bugs?"),
    ("technical", "security_quality", "How do you validate and sanitize untrusted user input?"),

    # ---------- Communication & Collaboration ----------
    ("collaboration", "behavioral", "Tell me about a time you disagreed with a teammate and how you resolved it."),
    ("communication", "behavioral", "How do you explain a complex technical concept to a non-technical stakeholder?"),
    ("communication", "behavioral", "Describe a situation where you had to give difficult feedback to a colleague."),
    ("collaboration", "behavioral", "How do you handle a situation where requirements are unclear or keep changing?"),
    ("collaboration", "behavioral", "Tell me about a time you collaborated with someone whose working style differed from yours."),
    ("collaboration", "behavioral", "How do you make sure everyone on a team is aligned on a decision?"),
    ("collaboration", "behavioral", "Describe a conflict within your team and how it was resolved."),
    ("communication", "behavioral", "How do you handle receiving critical feedback on your work?"),
    ("collaboration", "behavioral", "Tell me about a time you helped a struggling teammate."),
    ("communication", "behavioral", "How do you communicate progress and blockers to your team?"),
    ("communication", "behavioral", "Describe a time you had to persuade others to adopt your approach."),
    ("communication", "behavioral", "How do you ensure clarity when writing documentation or technical messages?"),
    ("communication", "behavioral", "Tell me about a time a miscommunication caused a problem and what you learned."),
    ("collaboration", "behavioral", "How do you balance speaking up with listening in team discussions?"),
    ("communication", "behavioral", "Describe how you adapt your communication style for different audiences."),
    ("collaboration", "behavioral", "How do you work effectively with a difficult or unresponsive team member?"),
    ("collaboration", "behavioral", "Tell me about a time you took the initiative to improve how your team works together."),

    # ---------- Learning Agility & Experience / Projects ----------
    ("learning_agility", "behavioral", "Tell me about a time you had to learn a new technology quickly for a project."),
    ("learning_agility", "behavioral", "How do you stay current with new tools and developments in your field?"),
    ("learning_agility", "behavioral", "Describe your most challenging project and the role you played in it."),
    ("learning_agility", "behavioral", "Tell me about a time you failed and what you learned from it."),
    ("learning_agility", "behavioral", "How do you approach a task that requires skills you do not yet have?"),
    ("learning_agility", "behavioral", "Describe a project where you contributed beyond your assigned role."),
    ("learning_agility", "behavioral", "What is something technical you taught yourself recently, and how did you go about it?"),
    ("learning_agility", "behavioral", "Tell me about a project you are most proud of and why."),
    ("learning_agility", "behavioral", "How do you handle situations where you are out of your depth?"),
    ("learning_agility", "behavioral", "Describe a time you had to adapt quickly to an unexpected change."),
    ("learning_agility", "behavioral", "What do you do when you stay stuck on a problem for a long time?"),
    ("learning_agility", "behavioral", "Tell me about a time you received constructive criticism and acted on it."),
    ("learning_agility", "behavioral", "How do you measure your own growth as a developer?"),
    ("learning_agility", "behavioral", "Describe a habit or routine you use to keep learning new skills."),
    ("learning_agility", "behavioral", "Tell me about a project that did not go as planned and how you responded."),
    ("learning_agility", "behavioral", "How do you decide which new skills are worth investing your time in?"),
    ("learning_agility", "behavioral", "Describe a time you stepped outside your comfort zone professionally."),
]


def embed_texts(tok, model, texts, batch=32, max_len=256):
    """server.py embed_query 와 동일: CLS 풀링 + L2 정규화 -> float32 행렬."""
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(DEVICE)
            v = model(**enc).last_hidden_state[:, 0]
            v = F.normalize(v, p=2, dim=1)
            vecs.append(v.float().cpu().numpy().astype("float32"))
    return np.vstack(vecs)


def main():
    os.makedirs(RAG_DIR, exist_ok=True)

    # 1) 정확 중복 제거
    seen, records = set(), []
    for cat, topic, q in QUESTIONS:
        qn = " ".join(q.split())
        key = qn.lower()
        if qn and key not in seen:
            seen.add(key)
            records.append({"question": qn, "category": cat, "topic": topic})
    print(f">>> 질문 {len(records)}개 (정확 중복 제거 후)", flush=True)

    # 2) bge-m3 로드 (server 와 동일 레시피)
    print(">>> bge-m3 로딩...", flush=True)
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    model = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE).half().eval()

    # 3) 임베딩
    mat = embed_texts(tok, model, [r["question"] for r in records])
    dim = int(mat.shape[1])
    print(f">>> 임베딩 완료: shape={mat.shape}, dim={dim}", flush=True)

    # 3.5) 유사 중복 점검(코사인 0.95 이상 쌍 경고만 — 한국어 은행의 near-dup 문제 예방)
    sim = mat @ mat.T
    np.fill_diagonal(sim, 0.0)
    dup_pairs = np.argwhere(sim > 0.95)
    dup_pairs = [(i, j) for i, j in dup_pairs if i < j]
    if dup_pairs:
        print(f">>> [경고] 코사인 0.95 초과 유사쌍 {len(dup_pairs)}개:", flush=True)
        for i, j in dup_pairs[:10]:
            print(f"    {sim[i, j]:.3f}  #{i} <-> #{j}", flush=True)
    else:
        print(">>> 유사 중복(>0.95) 없음 — 깨끗", flush=True)

    # 4) FAISS IndexFlatIP (정규화 벡터 -> 내적=코사인, server 와 동일 동작)
    index = faiss.IndexFlatIP(dim)
    index.add(mat)
    faiss.write_index(index, OUT_INDEX)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f">>> 저장 완료: {OUT_INDEX} (ntotal={index.ntotal})", flush=True)
    print(f">>> 저장 완료: {OUT_JSON} ({len(records)} records)", flush=True)

    # 카테고리 분포
    from collections import Counter
    dist = Counter(r["category"] for r in records)
    print(f">>> 카테고리 분포: {dict(dist)}", flush=True)

    # 5) 검색 스모크 테스트
    for tq in ["database indexing performance", "resolving a conflict with a teammate",
               "learning a new framework quickly", "designing a scalable API"]:
        s, ids = index.search(embed_texts(tok, model, [tq]), 3)
        print(f"\n[{tq}]", flush=True)
        for sc, i in zip(s[0], ids[0]):
            print(f"  {sc:.3f}  [{records[i]['category']}] {records[i]['question']}", flush=True)


if __name__ == "__main__":
    main()
