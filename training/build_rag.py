#!/usr/bin/env python3
"""
RAG 질문 은행 구축: AI Hub 면접 질문 -> bge-m3 임베딩 -> FAISS 인덱스.

실무 수준 특징:
- argparse: 직군/경로/강제재구축을 인자로 (타 직군 재사용)
- 기존 인덱스가 있으면 건너뜀(--force로만 재구축) — 5,684개 인덱스 보호
- --force 재구축 시 기존 인덱스·json을 타임스탬프 백업 후 교체
- 경로/파일/모델 사전 검증 + 단계별 진단 + 검색 테스트

기본 동작(인자 없음)은 기존 ict 인덱스 구축과 동일:
  ckmk_*_ict_*.json -> rag/ict_questions.index + ict_questions.json

사용 예:
  python build_rag.py                  # ICT (이미 있으면 skip)
  python build_rag.py --force          # ICT 강제 재구축(기존은 백업)
  python build_rag.py --occupation sm  # 다른 직군
"""
import os, sys, json, glob, time, shutil, argparse, traceback
import numpy as np
import torch
import torch.nn.functional as F

DATA_DIR_DEFAULT = "/workspace/interview_ai/data"
OUT_DIR_DEFAULT  = "/workspace/interview_ai/rag"
EMB_MODEL = "BAAI/bge-m3"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--occupation", default="ict", help="직군 코드 (ict/sm/ps/bm/mm/ard/rnd)")
    p.add_argument("--data-dir", default=DATA_DIR_DEFAULT)
    p.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--force", action="store_true", help="기존 인덱스가 있어도 재구축(백업 후)")
    return p.parse_args()

def main():
    args = parse_args()
    DATA_DIR, OUT_DIR, OCC = args.data_dir, args.out_dir, args.occupation
    os.makedirs(OUT_DIR, exist_ok=True)
    idx_path  = os.path.join(OUT_DIR, f"{OCC}_questions.index")
    json_path = os.path.join(OUT_DIR, f"{OCC}_questions.json")

    print("=== 0. 환경 ===")
    print("torch", torch.__version__, "| CUDA", torch.cuda.is_available())
    try:
        import faiss
        print("faiss OK")
    except Exception as e:
        print(f"[실패:faiss import] {e}"); sys.exit(1)
    from transformers import AutoTokenizer, AutoModel

    # --- 기존 인덱스 보호: 있으면 skip (--force로만 재구축) ---
    if os.path.exists(idx_path) and os.path.exists(json_path) and not args.force:
        try:
            n = faiss.read_index(idx_path).ntotal
        except Exception:
            n = "?"
        print(f">>> 인덱스가 이미 존재합니다: {idx_path} ({n}개)")
        print(">>> 재구축하려면 --force 를 붙이세요. (기본은 보호를 위해 건너뜀)")
        sys.exit(0)

    # --- 경로/파일 사전 검증 (가정 금지) ---
    if not os.path.isdir(DATA_DIR):
        print(f"[실패] 데이터 폴더가 없습니다: {DATA_DIR}"); sys.exit(1)
    print(f"\n=== 1. {OCC} 질문 추출 ===")
    files = glob.glob(os.path.join(DATA_DIR, "**", f"ckmk_*_{OCC}_*.json"), recursive=True)
    print(f"{OCC} 파일:", len(files))
    if not files:
        print(f"[실패] 패턴에 맞는 파일 0개 (occupation={OCC}) — 경로/직군코드 확인"); sys.exit(1)
    seen = set(); records = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))["dataSet"]
        except Exception:
            continue
        q = (((d.get("question") or {}).get("raw") or {}).get("text") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        info = d.get("info") or {}
        a = d.get("answer") or {}
        records.append({
            "question": q,
            "experience": info.get("experience", ""),
            "gender": info.get("gender", ""),
            "ref_summary": (a.get("summary") or {}).get("text", ""),
        })
    print(f"유니크 질문: {len(records)}개")
    if not records:
        print("[실패] 질문 0개 — 경로/패턴 확인"); sys.exit(1)

    # --- 2. 임베딩 모델(bge-m3) ---
    print("\n=== 2. bge-m3 로드 (첫 실행 ~2GB 다운로드) ===")
    try:
        tok = AutoTokenizer.from_pretrained(EMB_MODEL)
        emb_model = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE).half().eval()
        print("  -> 성공. GPU:", f"{torch.cuda.memory_allocated()/1024**3:.2f} GB")
    except Exception as e:
        print(f"[실패:임베딩모델] {type(e).__name__}: {e}"); traceback.print_exc(); sys.exit(1)

    def embed(texts, bs=64, max_len=256):
        out = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                v = emb_model(**enc).last_hidden_state[:, 0]   # bge-m3: CLS 풀링
                v = F.normalize(v, p=2, dim=1)                 # L2 정규화 -> 내적=코사인
            out.append(v.float().cpu().numpy())
            if (i // bs) % 10 == 0:
                print(f"  임베딩 {min(i+bs, len(texts))}/{len(texts)}")
        return np.vstack(out).astype("float32")

    # --- 3. --force 백업 후 임베딩 + FAISS 인덱스 ---
    if args.force and (os.path.exists(idx_path) or os.path.exists(json_path)):
        ts = time.strftime("%Y%m%d_%H%M%S")
        for p in (idx_path, json_path):
            if os.path.exists(p):
                shutil.copy2(p, f"{p}.{ts}.bak")
        print(f">>> --force: 기존 인덱스/json 백업 완료 (*.{ts}.bak)")

    print("\n=== 3. 임베딩 + 인덱스 구축 ===")
    try:
        questions = [r["question"] for r in records]
        vecs = embed(questions)
        print("임베딩 shape:", vecs.shape)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        faiss.write_index(index, idx_path)
        json.dump(records, open(json_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        size_mb = os.path.getsize(idx_path) / 1024**2
        print(f"  -> 저장: {idx_path} ({index.ntotal}개, {size_mb:.1f} MB)")
    except Exception as e:
        print(f"[실패:인덱스] {type(e).__name__}: {e}"); traceback.print_exc(); sys.exit(1)

    # --- 4. 검색 테스트 ---
    print("\n=== 4. 검색 테스트 ===")
    for tq in ["REST API와 HTTP 메서드에 대해 설명해주세요",
               "데이터베이스 인덱스란 무엇인가요",
               "객체지향 프로그래밍의 특징은"]:
        qv = embed([tq])
        scores, ids = index.search(qv, 3)
        print(f"\n[쿼리] {tq}")
        for rank, (s, idx) in enumerate(zip(scores[0], ids[0]), 1):
            print(f"  {rank}. (유사도 {s:.3f}) {records[idx]['question'][:70]}")

    print(f"\n[완료] {OCC} RAG 질문 은행 구축 완료 ({index.ntotal}개)")

if __name__ == "__main__":
    main()
