# DevReady AI — AI 면접 평가 엔진

> 자체 호스팅·파인튜닝한 단일 LLM(**EXAONE-Deep-7.8B**)으로 구동되는 한/영 AI 모의면접 평가 엔진.
> 취업 준비 플랫폼 **DevReady**의 AI 부문(FastAPI 백엔드). 답변 채점 · 질문 생성 · 꼬리질문 · 종합 리포트 · 음성 STT · 퀴즈/이력서/공고 생성을 **하나의 모델**로 처리한다.

**작성:** Seongchae ([@jsc1209](https://github.com/jsc1209)) · AI 코어 단독 설계·구현
**모델(어댑터):** [huggingface.co/seongchaeae/capstone-interview-ai-lora](https://huggingface.co/seongchaeae/capstone-interview-ai-lora)

---

## 한눈에

- **외부 LLM API 의존 0.** 면접 평가부터 퀴즈·자기소개서·채용공고 생성까지 전부 직접 튜닝·서빙한 EXAONE 한 모델로 동작 (블랙박스 상용 API 호출이 아님).
- **EXAONE-Deep-7.8B를 4bit(NF4)로 로드** → VRAM 약 6GB, RTX 4090 한 장에서 서빙.
- **RAG(지식) + QLoRA(출력 형식·행동 교정) + FastAPI(서빙)** 의 책임 분리.
- **한국어 / 영어 이중언어.** 같은 엔드포인트에 `lang` 하나로 전환.

## 아키텍처

```
React (프론트) ──(JWT)──▶ Spring Boot ──(서버-투-서버)──▶ FastAPI (이 repo, AI)
                              │                              ├─ EXAONE-Deep-7.8B (4bit) + QLoRA 어댑터
                              └─ JWT 검증 · DB 저장            ├─ RAG: bge-m3 + faiss (질문 은행)
                                                             └─ faster-whisper STT
```

- **RAG = 지식·그라운딩**, **QLoRA = 채점 형식·행동 교정**, **FastAPI = 서빙 계층** 으로 역할을 나눴다.
- 표정 분석(face-api.js)은 브라우저에서 수행하고 **집계 점수만** 백엔드로 보낸다(영상은 브라우저 밖으로 나가지 않음 — 프라이버시).

## 엔지니어링 하이라이트

이 프로젝트에서 실제로 부딪히고 해결한 것들 (자세한 수치·과정은 [`docs/DEVLOG.md`](docs/DEVLOG.md)):

1. **한국어 추론 강제 (prefill).** EXAONE-Deep은 `<thought>` 추론을 영어로 하려는 성향이 강해 한국어 채점의 약 80%가 영어 사고로 샜다. 채팅 템플릿 뒤에 한국어 문장을 prefill로 붙여 추론 언어를 한국어로 고정.
2. **completion-only 손실 버그 근본 수정.** TRL `DataCollatorForCompletionOnlyLM`이 응답 마커를 못 찾아 라벨을 전부 마스킹 → `loss 0.0`(학습 0)으로 죽어 있었다. TRL을 걷어내고 **프롬프트는 `-100`, 완성부(추론+JSON)만 직접 마스킹**하도록 재구현해 정상 손실 곡선(0.41→0.24)을 회복.
3. **자기증류 + 품질 게이트.** base 모델로 채점 데이터를 생성(self-distillation)하되, 핵심 3축(구체성·논리·전달) 0점 불가 / 인성문항 기술점수 N/A 허용 / 피드백·강점·사고 길이 검사로 거른다 (1000건 생성 → 933건 통과).
4. **"과적합본이 base보다 나쁠 수 있다"는 발견.** `train_loss`만 보면 못 잡는다. 검증 손실 최저(epoch1)를 자동 채택(`load_best_model_at_end` + early stopping)했고, 과적합된 epoch3는 JSON 포맷 준수율이 62%로 **base(75%)보다도 나빴다.**
5. **안티 할루시네이션.** 공고 분석·문장 다듬기는 "원문·사실만", 자소서·공고 생성은 "임의 수치 금지". 모델이 내는 집계값(종합점수, 정답 인덱스 등)은 신뢰하지 않고 **Python에서 재계산.**

## API 엔드포인트 (요약)

| 분류 | 메서드·경로 | 설명 |
| --- | --- | --- |
| 면접 | `POST /interview/evaluate` | 답변 채점 (4축 + 종합 + 강점/개선/피드백) |
| 면접 | `POST /interview/question` | 토픽 기반 질문 검색 (RAG) |
| 면접 | `POST /interview/followup` | 이전 답변 기반 꼬리질문 |
| 면접 | `POST /interview/generate` | 이력서·공고 기반 맞춤 질문 |
| 면접 | `POST /interview/report` | 세션 종합 리포트 |
| 면접 | `POST /interview/expression` | 표정 집계 점수 수신 |
| 면접 | `POST /interview/stt` | 음성→텍스트 + 전달력 지표 |
| 학습 | `POST /education/quiz` | 주제 기반 객관식 퀴즈 |
| 이력서 | `POST /resume/cover-letter`, `/resume/polish` | 자소서 생성 / 문장 다듬기 |
| 공고 | `POST /posting/generate`, `/posting/analyze` | 공고 생성 / 구조화 분석 |
| 시스템 | `GET /health` | 상태·로딩 확인 |

연동 계약(요청/응답 JSON, 인증, 권장 타임아웃)은 [`deploy/AI_연동가이드.md`](deploy/AI_연동가이드.md).

## 기술 스택 (검증·고정)

- **모델:** `LGAI-EXAONE/EXAONE-Deep-7.8B` (revision 고정 `e3f42b18f6b1`, 4bit NF4)
- **런타임:** PyTorch 2.4.1+cu124 · Transformers 4.48.3 · tokenizers 0.21.4 · CUDA 12.x
- **양자화/학습:** bitsandbytes 0.45.5 · accelerate 1.2.1 · peft · safetensors
- **RAG:** bge-m3 임베딩 + faiss · **STT:** faster-whisper large-v3 · **서빙:** FastAPI
- 전체 핀: [`requirements_lock.txt`](requirements_lock.txt)

## 실행

- **RunPod (권장 · 실제 데모):** RTX 4090 + 100GB 볼륨 → [`deploy/DEPLOY.md`](deploy/DEPLOY.md)
- **Colab (임시 테스트):** [`deploy/colab_serve.ipynb`](deploy/colab_serve.ipynb) — HF에서 어댑터만 받아 추론, "Run all"로 API 기동

## 저장소 구조

```
.
├── server/        # FastAPI 서빙 (server.py, start.sh, stt/voice/mapping)
├── training/      # 자기증류 · 품질게이트 · QLoRA 학습 · base vs LoRA 비교
├── deploy/        # 배포 가이드 · Colab 노트북 · API 연동 계약
├── docs/          # DEVLOG (개발 일지)
└── data/          # 형식 예시 샘플만 (AI Hub 원본·학습데이터는 비공개)
```

## 데이터 · 라이선스

- 학습은 **AI Hub "채용 면접 인터뷰" 데이터**의 라벨링 데이터를 사용했으며, **재배포가 금지**되어 원본·파생 학습 데이터(`*.jsonl`)와 RAG 인덱스는 이 저장소에 포함하지 않는다. (`data/sft_data.sample.jsonl`은 형식 설명용으로 직접 작성한 가짜 예시)
- 베이스 모델 **EXAONE-Deep-7.8B는 비상업(NC) 라이선스**이며, 그 위에서 학습한 **LoRA 어댑터도 동일하게 비상업 용도로 제한**된다. 자세한 내용은 [`LICENSE`](LICENSE).
