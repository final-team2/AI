# DEPLOY — DevReady AI 띄우기

AI 관리자가 없어도 팀원이 직접 API를 띄울 수 있게 정리한 가이드. 두 가지 방법이 있다.

- **RunPod** — 실제 데모·통합용 (안정적, 권장)
- **Google Colab** — 빠른 임시 테스트용 (무료, 세션·터널이 불안정)

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

## B. Google Colab (임시 테스트)

> ⚠️ **사전 조건:** 학습한 LoRA 어댑터가 Hugging Face에 올라가 있어야 한다. 노트북은 HF에서 어댑터를 받아 추론한다. (현재 HF 레포 루트에는 구버전 어댑터가 있으니, 최신 어댑터 경로를 노트북 설정 셀에서 맞출 것)

1. `deploy/colab_serve.ipynb`를 Colab에서 연다. (런타임 → 런타임 유형 변경 → **GPU(T4)** 선택)
2. 상단 **설정 셀**에 ngrok authtoken과 어댑터 HF 경로를 입력.
3. **런타임 → 모두 실행(Run all).** 모델 로드(수 분) 후 마지막 셀이 공개 URL을 출력.
4. 그 URL의 `/health`가 열리면 라이브. `/evaluate`로 채점 호출.

한계: Colab은 일정 시간 후 세션이 끊기고, ngrok 무료 터널도 불안정하다. **빠른 확인용**으로만 쓰고, 실제 데모/통합은 RunPod를 쓴다.

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
