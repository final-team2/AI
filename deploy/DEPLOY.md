# DEPLOY — DevReady AI 띄우기

AI 관리자가 없어도 팀원이 직접 API를 띄울 수 있게 정리한 가이드. 두 가지 방법이 있다.

- **RunPod** — 실제 데모·통합용 (안정적, 권장)
- **Google Colab + Drive** — 전체 서버를 무료로 기동 (세션·터널이 매번 바뀜, 개발·테스트용)

> 두 방법 모두 **추론만** 한다. 학습 데이터(AI Hub)는 필요 없고, 모델 어댑터는 Hugging Face에서 내려받는다. (재학습은 관리자 로컬에서만)

---

## A. RunPod (권장)

1. **Pod 생성** — GPU: RTX 4090(또는 24GB급). 템플릿: PyTorch / CUDA 12.x.
2. **네트워크 볼륨 연결** — 코드·모델·RAG 인덱스가 있는 100GB 볼륨을 `/workspace`에 마운트. (볼륨이 만들어진 **같은 데이터센터**에서 Pod를 생성해야 연결 가능)
3. **환경변수 + 기동**
   ```bash
   export HF_HOME=/workspace/interview_ai/hf_cache
   bash /workspace/interview_ai/start.sh
   ```
   - `start.sh`가 venv 활성화 → uvicorn(포트 8000) 기동 → `/health` 폴링까지 처리.
   - HF 토큰은 볼륨에 보존되어 재로그인 불필요.
4. **외부 주소 확인** — 출력 끝의 `https://<id>-8000.proxy.runpod.net`. **이 주소는 Pod를 켤 때마다 바뀐다.** 백엔드 담당자에게 새 주소를 공유.
5. **인증** — 모든 호출에 `X-API-Key` 헤더 필요(키는 볼륨의 `.api_key`). `/health`·`/docs`는 키 없이 접근 가능.

### Pod Stop / 재개 주의
- Stop 시 GPU 과금만 멈추고 볼륨(스토리지) 과금은 계속.
- on-demand는 재개 시 GPU가 0개로 잡힐 수 있음 → 그 Pod는 terminate 후 **같은 데이터센터**에 새 Pod 생성 + 볼륨 재연결(데이터는 그대로).

---

## B. Google Colab + Google Drive — 전체 서버 (무료)

`server.py`를 통째로 띄워 **모든 엔드포인트**를 제공한다. 코드·RAG 인덱스·`train.jsonl`은 공유 **Drive 폴더**에서, 베이스 모델과 **LoRA 어댑터는 HuggingFace**에서 받는다. 팀원이 각자 자기 Colab에서 띄울 수 있다.

> **사전 조건 (관리자가 한 번만):**
> 1. Drive 폴더 `AI-캡스톤(비상업용)`을 팀원에게 **공유**. 폴더 최상위에 `server.py · mapping.py · voice.py · stt.py · train.jsonl · run_in_colab.ipynb`, 그리고 `rag/`(ict_questions.index/.json [+ _en])이 있어야 한다.
> 2. **HF Read 토큰**(이 어댑터 레포 범위) 발급 → 팀원에게 전달. (어댑터 레포가 비공개라 토큰 필수)

**팀원이 각자 (한 번씩):**
1. 자기 드라이브 → **공유 문서함(Shared with me)** → `AI-캡스톤(비상업용)` 폴더 **우클릭 → "내 드라이브에 바로가기 추가"**. (팀 폴더 안에 있어도 그 **AI 폴더 자체**를 바로가기 추가하면 됨 — Colab은 "공유 문서함"은 못 보고 "내 드라이브"만 마운트하기 때문)
2. 그 폴더 안 **`run_in_colab.ipynb`**를 Colab으로 연다.
3. Colab 왼쪽 **🔑 Secrets** → 이름 `HF_TOKEN`, 값 = 받은 토큰 → **Notebook access ON**.
4. **런타임 → GPU(T4)** → **모두 실행(Run all)**.
5. 7~9분(첫 모델 다운로드) 후 마지막 셀이 `https://<무작위>.trycloudflare.com` 공개 URL을 출력. `/health`가 `ready: true`면 라이브, 전체 명세는 `/docs`.

> cell 1이 Drive 폴더 위치를 자동탐색하므로 경로 수정은 필요 없다.

한계: Colab은 일정 시간 후 세션이 끊기고 cloudflared 무료 터널 URL도 매번 바뀐다. **개발·테스트용**으로 쓰고, 안정적인 데모/통합은 RunPod를 권장.

### B-2. 경량 데모 (채점만)
RAG/Drive 없이 채점 핵심만 빠르게 보려면 `deploy/colab_serve.ipynb` — HF에서 어댑터만 받아 `/evaluate`·`/health`만 띄우는 경량 버전(역시 `HF_TOKEN` 필요). 전체 엔드포인트가 필요하면 위 A 또는 B를 쓴다.

---

## API 호출

연동 계약(요청/응답 JSON, 인증 헤더, 권장 타임아웃)은 [`AI_연동가이드.md`](AI_연동가이드.md) 참고. 핵심만:

```bash
curl -X POST "<BASE_URL>/interview/evaluate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <키>" \
  -d '{"question":"REST API란?","answer":"...","lang":"ko"}'
```

- 추론 모델이라 응답에 **20~60초**가 걸린다 → 백엔드 타임아웃을 90~120초로, 프론트에 로딩 UI 필수.
- 응답은 `{"ok": true/false, ...}` 형태. `ok`를 먼저 확인하고, Pod가 꺼졌을 때의 폴백을 둘 것.
