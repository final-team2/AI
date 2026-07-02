# AI 모의면접 답변 평가 시스템 — 개발 일지

> 국비지원 캡스톤 프로젝트 · AI 코어 담당
> Base: EXAONE Deep 7.8B (4bit NF4) + RAG + QLoRA + FastAPI

## 1. 프로젝트 개요
AI 웹개발자 모의면접 Q&A 평가 시스템. 서비스직 교육평가 · 대화형 면접 평가 · 음성 상황인식 AI 세 주제를 하나의 채점기로 통합한다. 팀 프로젝트이며 본인은 AI 코어 전체(모델·RAG·학습·서빙)를 담당하고, React 프론트엔드는 팀원이 API로 사용한다.

## 2. 아키텍처
역할을 셋으로 분리했다.
- **RAG** — 지식·질문 은행. 실제 면접 질문을 검색해 와서 모델이 질문을 지어내지 않게 한다.
- **QLoRA** — 채점의 형식·행동 학습. 점수+피드백을 항상 일관된 JSON으로 출력하게 한다.
- **FastAPI** — React가 호출하는 서빙 계층.

base 모델은 한국어 추론 모델인 EXAONE Deep 7.8B를 4bit NF4로 로드한다. 부가 기능으로 브라우저 표정 분석(MediaPipe/face-api.js), 음성(Whisper STT + TTS)이 로드맵에 있다.

## 3. 기술 스택 (검증·고정)
- torch 2.4.1+cu124, transformers 4.48.3, tokenizers 0.21.4, accelerate 1.2.1, bitsandbytes 0.45.5
- peft 0.13.2, trl 0.12.0
- 임베딩 BAAI/bge-m3, 벡터검색 faiss
- **모델 revision은 e3f42b18f6b1로 고정** (main 커밋은 transformers v5 전제라 이 스택을 깨뜨림)
- 환경: RunPod RTX 4090, 100GB 볼륨

## 4. 진행 현황
1. **환경·모델** — RunPod에 EXAONE Deep 7.8B 4bit 로드(VRAM 약 5GB). 의존성·revision 고정.
2. **데이터** — AI Hub "채용 면접 인터뷰"(라벨링 68,078건)에서 ICT 직군 추출. 원본에 점수 라벨 없음 → 채점은 학습이 아닌 모델 추론으로 방향 결정.
3. **RAG** — bge-m3 임베딩 + faiss 인덱스, 중복 제거한 ICT 질문 5,684개.
4. **학습 데이터** — base 모델 self-distillation으로 (질문+답변 → 점수+피드백 JSON) 200건 생성, 3중 게이트로 필터링.
5. **QLoRA** — r16/alpha32, 3 epoch, loss 2.92 → 0.83.
6. **서빙** — server.py에 질문 생성 / 답변 평가 두 엔드포인트. greedy + 한국어 prefill, GPU 6.13GB, 응답 50~60초.
7. **배포** — HF repo(capstone-interview-ai-lora)에 어댑터+모델카드 업로드, 다운로드로 검증.

## 5. 트러블슈팅 로그
진행하며 실제로 막혔던 지점과 해결책.

### 5-1. 모델 로드 실패 (transformers 버전 충돌)
- 증상: 최신 코드로 EXAONE 로드 시 에러.
- 원인: 모델 main 커밋이 transformers v5를 전제 → 4.48.3 스택과 불일치.
- 해결: 모델 revision을 초기 커밋 e3f42b18f6b1로 고정, 의존성 버전도 lock.

### 5-2. 추론(사고)이 영어로 나옴 (약 80% 탈락)
- 증상: EXAONE이 사고 과정을 영어로 생성해 한국어 채점에 부적합.
- 원인: 추론 모델의 강한 영어 사고 prior. 200샘플 LoRA로도 뒤집지 못함.
- 해결: chat template로 얻은 텍스트(끝이 `<thought>` 줄바꿈)에 한국어 문장을 prefill로 덧붙여 재토크나이즈 후 생성하면 사고가 한국어로 이어진다. 한국어 판정은 prefill을 제외한 생성분으로 한다.

### 5-3. 어댑터 저장 에러 (safetensors)
- 증상: 학습 체크포인트 저장 시 SafetensorError: invalid shape.
- 원인: EXAONE의 공유(tied) 텐서(embed ↔ lm_head).
- 해결: save_safetensors=False로 .bin 저장(공유 텐서 문제 우회).

### 5-4. 평가 생성 시 NaN (추론 안정성)
- 증상: 서빙 중 probability tensor contains inf/nan.
- 원인: 4bit+LoRA 샘플링에서 큰 logit/temperature 조합의 오버플로.
- 해결: do_sample=False(greedy)로 전환. 확률 텐서를 거치지 않아 안정적이고 채점 재현성도 확보.

### 5-5. completion-only 재학습 실패 → 롤백
- 증상: 프롬프트를 마스킹하고 응답만 학습하는 재학습이 두 번 다 train_loss 0.0(빈 학습).
- 원인: response_template 토큰이 chat template의 실제 토큰화와 불일치 → 전체가 마스킹됨.
- 해결: 2회 실패 후 검증된 full-text 손실 v1 어댑터로 롤백. ("두 번 실패하면 롤백" 원칙 적용)

### 5-6. AI Hub 다운로드 실패 (해외 IP)
- 증상: RunPod에서 aihubshell 실행 시 HTTP 502.
- 원인: 해외 IP geo-restriction.
- 해결: 한국 PC에서 다운로드 후 SFTP로 pod에 전송.

### 5-7. Windows tar 한글 경로 깨짐
- 증상: 데이터를 묶을 때 한글 폴더명 인코딩이 깨짐.
- 해결: 필요한 라벨링데이터만 골라 tar로 묶어 전송(원본 약 162GB 중 약 82MB).

## 6. 포트폴리오 포인트
- 의존성 호환·revision 고정으로 재현성 확보.
- 한국어 추론 강제(prefill) — 검색해도 안 나오는 자체 해법.
- 양자화+LoRA 추론 안정화(greedy).
- 자기증류 데이터 품질 장치(overall 재계산, 3중 게이트).
- 질문 유형 자동 판별(기술 vs 인성) 후 항목별 채점.

## 7. 남은 작업
- 음성(STT/TTS), 표정 분석.
- 이력서·공고 기반 면접 문제 자동 생성.
- 프론트엔드 통합·데모.

## 개발 로그
### 2026-06-06
- HF repo `capstone-interview-ai-lora` 생성, 어댑터(adapter_model.bin 168MB) + 모델카드 업로드.
- 다운로드로 검증(어댑터 묶음 온전, AI Hub 원본 답변 미포함 확인).
- 본 개발 일지(DEVLOG.md) 작성 시작.

### 2026-06-07
- (오늘 한 작업/오류/해결을 여기에)

### 2026-06-07
- 자동 질문 생성 기능 추가: 이력서+채용공고 → EXAONE base(어댑터 off) + 한국어 prefill + greedy → 맞춤 면접 질문 N개 JSON. server.py에 POST /interview/generate 엔드포인트로 통합. 가상 이력서/공고 테스트에서 한국어 사고·JSON 5개·맞춤형 질문 모두 확인.
- 질문 생성에는 평가용 LoRA 어댑터를 disable_adapter()로 끄고 base 모델 사용(어댑터는 점수+피드백 형식에 특화돼 있어 질문 생성엔 base가 자연스러움).
- 트러블슈팅(포트 충돌): RunPod 재시작 시 Jupyter Lab이 8888 포트를 선점 → uvicorn이 그 포트에 bind 실패(Errno 98, address already in use)해 startup 직후 종료. 모델 로드 자체는 정상이었고 포트만 문제. 해결: 서버를 8000 포트로 운영(jupyter는 8888 유지), React 팀에도 8000 안내.

### 2026-06-07 (추가)
- 꼬리질문 기능: 질문+답변 → 답변을 파고드는 후속 질문 1개 JSON. POST /interview/followup (base 모델 + 한국어 prefill). 답변의 특정 지점("보안 그룹 설정 변경")을 정확히 짚는 것 확인.
- 면접관 페르소나: PERSONAS 5종(default/senior_tech/culture_fit/pressure/mentor). GET /interview/personas로 목록 제공, /generate·/followup에 persona 키로 선택해 면접관 스타일을 프롬프트 intro로 주입.
- server.py 정돈: append로 세 번 늘어난 코드를 한 파일로 정리(엔드포인트 6개: health/personas/question/evaluate/generate/followup). 기존 채점 회귀 없음 확인. 운영 포트는 8000(jupyter가 8888 점유).

### 2026-06-07 (server.py 실무 수준 전면 개편)
- 엔드포인트 동작은 유지하되 운영 견고성 9종을 추가하고 server.py를 한 파일로 재정리(331줄). 배포 전 샌드박스에서 문법·유틸 유닛테스트(6/6)·base64 왕복검증을 거쳐 한글 깨짐 없이 교체.
- (1) 요청 로깅 미들웨어: 엔드포인트별 [메서드] 경로 -> 상태 (소요시간)을 server.log에 기록(지연 추적).
- (2) /health 강화: ready + rag_questions(5684) + adapter_loaded(true) + gpu_memory_gb(6.13).
- (3) 입력 검증: 빈 값·길이 상한(answer 6000 / resume·job_posting 12000 등) 초과 시 즉시 {ok:false, error}. 빈 answer 테스트로 확인.
- (4) 점수 0~100 클램핑: 모델이 범위 밖 값을 내도 강제 보정.
- (5) JSON 파싱 강건화: </thought> 이후에서 코드펜스·트레일링 콤마 보정 후 1회 재시도.
- (6) 응답 형식 일관화: 전 엔드포인트 {ok, ...}. 검증 200 / 형식오류 422 / 예외 500(전역 핸들러) 모두 본문 형태 동일. /interview/question에 ok:true 필드 추가(프론트 공지 필요).
- (7) /report 한글 라벨: LLM 입력 점수 키를 "기술 정확도/답변 구체성/논리성/의사소통"으로 변환해 직역("특이성") 방지. weaknesses가 실제 답변 근거로만 출력됨 확인.
- (8) 프롬프트 오타 수정(마지막에/보완점).
- (9) 공통 헬퍼 run_llm으로 evaluate/generate/followup/report 생성 로직 중복 제거.
- 검증: 4개 테스트(health·입력검증·평가 회귀·리포트) 모두 통과. 평가 회귀 정상(기술질문 95점 JSON), 리포트 technical_accuracy=90(인성 0 제외)·overall 82.
- 운영 포트 8000 유지. 롤백 백업: server_pre_prod.py.

### 2026-06-07 (실무 수준 하드닝: 서버 + 파이프라인 전체)
텍스트 AI 코어 전반을 "재현 가능한 실무 수준"으로 정비. 모든 변경은 배포 전 샌드박스에서 문법·핵심로직 검증을 거쳤고, 기존 산출물(RAG 인덱스 5,684 / sft_data·train 200 / LoRA 어댑터)은 재실행 없이 그대로 보존. 모든 원본은 *_pre_prod 백업.

[server.py] 운영 견고성 9종: 요청 로깅 미들웨어, /health 강화(rag_questions·adapter_loaded·gpu_memory), 입력 길이 검증, 점수 0~100 클램핑, JSON 파싱 강건화(+1회 재시도), 전 엔드포인트 {ok,...} 일관화+전역 예외 핸들러, /report 한글 점수라벨로 직역("특이성") 방지, 프롬프트 오타 수정, 공통 헬퍼 run_llm. 포트 8000 운영. /interview/question에 ok:true 추가(프론트 공지 필요). 검증: health·입력검증·평가회귀(기술질문 95점)·리포트(technical_accuracy 90·인성 0 제외, overall 82) 통과.

[gen_sft_data.py] argparse(직군/목표/출력/--fresh)로 타 직군·언어 재사용 가능화. --fresh 시 기존 파일 타임스탬프 자동 백업(과거 rm로 인한 데이터 유실을 구조적으로 차단). 중단점 재개(생성된 질문 skip), 매 건 flush 저장, Ctrl+C 안전 종료+재개 안내, 경로/파일 사전 검증, 주기적 통계. 생성 파라미터·품질 게이트는 검증된 그대로.

[train_qlora.py] 체크포인트 재개(--resume): epoch별 최신 checkpoint에서 이어서 학습(Pod 중단·OOM 복구). 하이퍼파라미터 인자화, training_summary.json(설정·train_loss·소요시간) 기록, 학습 중 운영 금지사항 출력, 경로/데이터 검증. (옵션) --val-ratio 검증 분할+eval_loss. 인자 없는 기본 동작은 검증된 full-text SFT 설정과 동일(save_safetensors=False .bin 유지).

[build_rag.py] 직군 인자화. 기존 인덱스 있으면 skip(--force로만 재구축)로 인덱스 보호; --force 시 백업 후 교체. CLS 풀링·L2 정규화·IndexFlatIP 그대로.

[build_sft.py] 입출력 인자화, 기존 train.jsonl 백업 후 재생성, 손상 레코드 skip+집계+overall 분포 통계. overall 재계산(technical=0→3축 평균)과 messages/completion 포맷 그대로.

[start.sh] 포트 8888->8000. 준비 감지를 로그 문자열 대신 /health의 ready:true 폴링으로 변경. HF 캐시 경로 고정(재시작 후 재다운로드 방지), Jupyter 종료 제거(8888 충돌 없음->노트북 보호), server.py 존재 확인. 실제 실행으로 준비완료+health(ready:true, 5684) 확인.

원칙: 검증된 동작·포맷은 변경하지 않고 그 위에 (1) 산출물 보호/백업 (2) 중단점 재개 (3) 경로·입력 사전 검증 (4) 인자화로 재사용성 (5) 통계·요약 로깅 만 추가.

### 2026-06-07 (실무 수준 하드닝: 서버 + 파이프라인 전체)
텍스트 AI 코어 전반을 "재현 가능한 실무 수준"으로 정비. 모든 변경은 배포 전 샌드박스에서 문법·핵심로직 검증을 거쳤고, 기존 산출물(RAG 인덱스 5,684 / sft_data·train 200 / LoRA 어댑터)은 재실행 없이 그대로 보존. 모든 원본은 *_pre_prod 백업.

[server.py] 운영 견고성 9종: 요청 로깅 미들웨어, /health 강화(rag_questions·adapter_loaded·gpu_memory), 입력 길이 검증, 점수 0~100 클램핑, JSON 파싱 강건화(+1회 재시도), 전 엔드포인트 {ok,...} 일관화+전역 예외 핸들러, /report 한글 점수라벨로 직역("특이성") 방지, 프롬프트 오타 수정, 공통 헬퍼 run_llm. 포트 8000 운영. /interview/question에 ok:true 추가(프론트 공지 필요). 검증: health·입력검증·평가회귀(기술질문 95점)·리포트(technical_accuracy 90·인성 0 제외, overall 82) 통과.

[gen_sft_data.py] argparse(직군/목표/출력/--fresh)로 타 직군·언어 재사용 가능화. --fresh 시 기존 파일 타임스탬프 자동 백업(과거 rm로 인한 데이터 유실을 구조적으로 차단). 중단점 재개(생성된 질문 skip), 매 건 flush 저장, Ctrl+C 안전 종료+재개 안내, 경로/파일 사전 검증, 주기적 통계. 생성 파라미터·품질 게이트는 검증된 그대로.

[train_qlora.py] 체크포인트 재개(--resume): epoch별 최신 checkpoint에서 이어서 학습(Pod 중단·OOM 복구). 하이퍼파라미터 인자화, training_summary.json(설정·train_loss·소요시간) 기록, 학습 중 운영 금지사항 출력, 경로/데이터 검증. (옵션) --val-ratio 검증 분할+eval_loss. 인자 없는 기본 동작은 검증된 full-text SFT 설정과 동일(save_safetensors=False .bin 유지).

[build_rag.py] 직군 인자화. 기존 인덱스 있으면 skip(--force로만 재구축)로 인덱스 보호; --force 시 백업 후 교체. CLS 풀링·L2 정규화·IndexFlatIP 그대로.

[build_sft.py] 입출력 인자화, 기존 train.jsonl 백업 후 재생성, 손상 레코드 skip+집계+overall 분포 통계. overall 재계산(technical=0→3축 평균)과 messages/completion 포맷 그대로.

[start.sh] 포트 8888->8000. 준비 감지를 로그 문자열 대신 /health의 ready:true 폴링으로 변경. HF 캐시 경로 고정(재시작 후 재다운로드 방지), Jupyter 종료 제거(8888 충돌 없음->노트북 보호), server.py 존재 확인. 실제 실행으로 준비완료+health(ready:true, 5684) 확인.

원칙: 검증된 동작·포맷은 변경하지 않고 그 위에 (1) 산출물 보호/백업 (2) 중단점 재개 (3) 경로·입력 사전 검증 (4) 인자화로 재사용성 (5) 통계·요약 로깅 만 추가.

### 2026-06-07 (음성 STT 파이프라인 추가)
면접 답변을 음성으로 받는 STT 백엔드 구축. 고정 스택을 건드리지 않고(설치 전후 torch/transformers/tokenizers/bitsandbytes/numpy 동일 확인) 음성 전용 패키지만 추가: faster-whisper 1.2.1 + ctranslate2 4.8.0(torch 비의존 GPU 런타임) + av 17.1.0(ffmpeg 내장) + onnxruntime + python-multipart.

- STT 모듈 stt.py: faster-whisper large-v3, GPU/float16, 지연 로딩(스레드 안전), vad_filter=True로 무음/비음성 환각 감소. transcribe_bytes() -> {text, language, duration, elapsed, segments}.
- server.py에 POST /interview/stt 엔드포인트 append(기존 코드 무손상): multipart 업로드 -> 변환 텍스트 {ok, text, ...}. 25MB 상한, {ok,...} 일관.
- server.py 시작 시 whisper 백그라운드 웜 스타트(데몬 스레드) 추가 -> 첫 요청도 즉시.
- 검증: 스모크 테스트(large-v3 로드 7.5s, 변환 1.0s) + /stt 콜드 6.9s / 웜 0.22s(3초 오디오 약 14배속). VRAM 서버+whisper 10.2/23.5GB.
- React 연동: 브라우저 MediaRecorder -> /stt -> 텍스트 -> /evaluate(기존 채점 엔진 재사용); TTS는 브라우저 speechSynthesis(ko-KR). webm/mp4 변환 불필요(av 디코딩). CORS 허용.
- 신규 패키지는 requirements_lock.txt에 고정.


---

## [2026-06-08] 이중언어(한/영) 지원 + JSON 파서 하드닝

### 추가 / 변경
- **영어 평가·생성·꼬리질문·리포트 지원**(한글 수준 동등). 언어는 요청의 `lang`("ko"/"en")으로 분기하며 기본값은 "ko" → 기존 한글 동작은 바이트 단위로 동일하게 보존.
- 핵심 설계: 영어는 **별도 어댑터 불필요**. EXAONE이 영어 추론을 네이티브로 수행하므로, 한글에서 쓰던 "한국어 prefill 강제" 트릭 없이 **base 모델 + 영어 루브릭(RUBRIC_EN)** 만으로 처리. 한글은 기존대로 어댑터 경로 사용.
- **RUBRIC_EN 캘리브레이션**: 0~100 스케일을 명시("85 not 8")하고 DB 인덱스 worked-example을 넣어, 강한 답변이 0~10이 아니라 80~85대로 채점되도록 고정.
- **`parse_json_lenient` 3단 강화**: strict `json.loads` → 트레일링 콤마 보정 → **json-repair 폴백**. base 모델이 가끔 내는 깨진 JSON(키 따옴표 누락 등)을 복구. 한·영 공통 적용.
- 페르소나·운영 메시지·라벨·프롬프트를 전부 ko/en 이중 키로 구성. `/health`에 `languages`·`en_question_bank` 추가, `/interview/personas?lang=`, `/interview/stt`에 `lang` Form 필드 추가.
- lifespan에서 `rag/ict_questions_en.index`/`.json`가 있으면 자동 로드(없으면 `en_question_bank:false`, `/question?lang=en`은 안내 메시지 반환).

### 검증 결과
- 한국어 평가(어댑터): 95 / 90 / 85 / 88, overall 90 — 회귀 정상(어댑터 경로 불변 확인).
- 영어 평가(base): 85 / 80 / 80 / 80, overall 81 — **스케일 정확(0~100)**, JSON 클린 파싱(json-repair 미발동).
- 영어 질문 생성: 이력서/공고 맞춤 4문항(기술 3 + 행동 1), 영어 한 문장씩.
- RAG 질문 은행: 5개 역량 카테고리(기술·분석·소통·러닝어질리티·협업) 전부 커버 확인. 인성/협업 계열이 오히려 강하게 매칭(협업 토픽 코사인 0.75).

### 의존성
- `json-repair==0.60.1` 추가(순수 파이썬, 고정 스택 불변). `requirements_lock.txt` 반영.

### 운영 교훈
- 수십~수백 줄 규모의 터미널 붙여넣기는 MobaXterm에서 줄 누락/경계 뭉개짐이 반복 발생(예: 77줄 → 68줄). **파일은 SFTP로 전송**하고 sha256로 검증한 뒤 교체하는 방식이 안전. server.py 교체도 base64 붙여넣기 대신 SFTP + sha256 게이트로 처리해 성공.


---

## [2026-06-08] 영어 RAG 질문 은행 구축 + base vs QLoRA 비교 실험

### 영어 RAG 질문 은행 (완성)
- **큐레이션 방식** 채택(EXAONE 생성 대신): ① 품질·다양성 보장 ② 유사 중복 없음 ③ GPU·모델 로딩 부담 없음(bge-m3 임베딩만) ④ AI Hub 파생이 아니라 라이선스 깨끗 ⑤ 역량 태깅 가능. AI '생성' 서사는 `/generate`(라이브 기능)가 담당하므로 RAG는 큐레이션 지식 저장소로 분리.
- `build_rag_en.py`: 큐레이션 영어 IT 면접 질문 **134개**(역량 5축 × 도메인 7버킷 — 알고리즘·DB·네트워크/OS·웹백엔드·보안·행동) → bge-m3(server와 동일한 CLS 풀링 + L2 정규화 레시피) → FAISS `IndexFlatIP` → `rag/ict_questions_en.index` / `.json`.
- 카테고리 분포: technical 78 / analytical 20 / learning_agility 17 / collaboration 11 / communication 8. **정확·유사(코사인>0.95) 중복 0개.**
- 서버 재시작 후 `/health` `en_question_bank: true` 확인. `/question?lang=en` 검색 품질 양호(DB 토픽 0.67, 협업 토픽 0.70 등 정확 매칭).
- 한국어 은행(5,684, AI Hub 전사)의 **유사 중복 문제(거의 같은 질문 3개)** 가 큐레이션 영어 은행에는 없음 — 구조적 개선.

### base vs QLoRA 파인튜닝 비교 실험 (발표용)
- `compare_base_vs_lora.py`: 동일 (질문, 답변) **8쌍**을 어댑터 ON/OFF · 동일 프롬프트·루브릭 · greedy 디코딩으로 평가하고 출력 일관성 지표를 비교. `compare_results.md` / `.json` 저장.
- **요약(N=8):** JSON 파싱 LoRA 8/8 vs base 6/8 · 스키마 4축 8/8 vs 6/8 · feedback 8/8 vs 6/8 · `</thought>` 7/8 vs 5/8 · 점수 0~100 범위 8/8 vs 6/8 · 한국어 추론 8/8 vs 8/8.
- **실질 격차는 표보다 큼:** base의 "통과" 중 2건은 모든 점수가 0점인 깡통 출력(JSON 틀만 복사)이라, base가 실제로 쓸 수 없는 출력을 낸 게 **4/8**(파싱 실패 2 + 깡통 2). LoRA는 0/8 실패.
- base는 `overall` 필드를 **0/8**(LoRA 8/8)로 출력 계약 미준수. 인성 문항에 `technical_accuracy=40`을 부여해 루브릭을 위반(LoRA는 0으로 정상 처리).
- base가 정상 출력한 케이스의 점수는 LoRA와 근접(예: 90/70/85/85 vs 90/65/85/88) → **"파인튜닝은 점수 크기가 아니라 깨지던 포맷·행동을 안정화한다"** 는 QLoRA 설계 목표를 실증.
- **정직한 한계:** 한국어 추론 유지율은 동률(8/8) — prefill 트릭이 base 경로에도 적용되므로 이 부분은 어댑터의 공헌이 아님. 발표 시 과장하지 않음.


---

## [2026-06-08] AI 후처리 4종 — 점수 매핑 · 표정/음성 수신 · 리포트 통합

프론트(React) 스키마에 맞춰 백엔드 출력을 정리하고, 비언어 신호(표정·음성)를 평가 리포트에 연결했다. 모든 변경은 **덧붙이기**(기존 응답 키·한국어 동작 불변)이며, 단계마다 백업(`server_pre_*.py`) + 멱등 패치 스크립트 + `py_compile`로 처리했다.

### 1) 루브릭 매핑 (`mapping.py`, 순수 함수·GPU 불필요)
평가 4축(`technical_accuracy`·`specificity`·`logic`·`communication`)을 프론트가 쓰는 점수 체계로 변환.
- **문항별 3점** `to_question_scores`: `logic←logic`, `clarity←communication`, `depth←round((specificity+technical_accuracy)/2)` → `/interview/evaluate` 응답에 `display_scores` 추가.
- **종합 5점** `to_report_scores`: `tech·comm·problem·attitude·logic` (+ `overall`=5축 평균, `grade`) → `/interview/report` 응답에 `categories`·`grade`·`overall_categories` 추가. `problem=(logic+specificity)/2`, `attitude`=비언어 신호 있으면 그것·없으면 `(comm+logic)/2` 휴리스틱.

### 2) 표정 점수 수신 `POST /interview/expression`
브라우저 face-api.js가 계산한 **집계 점수만** 수신(영상·프레임은 브라우저 밖으로 나가지 않음 — 프라이버시). 4지표 `confidence·composure·attention·expressiveness`를 0~100 클램프하고, **클라이언트 overall은 신뢰하지 않고** `0.3·자신감 + 0.3·안정 + 0.3·주의 + 0.1·표현력`으로 백엔드 재계산(평가 overall 가드레일과 동일 원칙). 라벨(안정적/보통/개선필요) + "전달력 보조 지표이며 합격 예측 아님" 안내 반환.

### 3) 음성 전달력 지표 (`voice.py`, 순수 함수 → `/stt` 확장)
STT 결과(`text`·`duration`·`segments`)만으로 계산해 **추가 추론 없음**. `stt.py`는 단어 타임스탬프를 주지 않지만 `vad_filter=True`로 무음이 제거되므로 **문장(segment) 간 갭 = 실제 휴지**로 추정. 지표: `wpm`(분당 어절)·`filler_count/ratio`·`pause_count`·`long_pause_count`·`longest_pause`·`speaking_ratio`·`delivery_score`(0~100). `delivery_score = 0.4·말속도 + 0.3·유창성(필러) + 0.3·흐름(발화비율·긴휴지)`. 적정 말속도 한국어 110–180 어절/분 / 영어 120–160 단어/분. `/interview/stt` 응답에 `voice` 블록만 덧붙임(기존 키 불변).

### 4) 리포트 통합 `/interview/report`
`ReportReq`에 `voice`·`expression`(선택) 필드 추가. `mapping.py`는 그대로 두고 `/report`에서 입력만 변환 — expression은 `overall` 제외 후 4지표, voice는 `delivery_score`를 mapping이 읽는 `clarity_score`로 매핑 — 해서 `to_report_scores(..., voice=, expression=)`로 전달. 결과적으로 **표정+음성이 비언어 종합으로 `attitude`(태도·열정) 축**을 채운다. 신호가 없으면 기존 텍스트 휴리스틱으로 동작(하위호환).

### 검증 (라이브 + 샌드박스)
- 매핑: 4축 `{80,75,80,85}` → `display {logic80, clarity85, depth78}`. `/evaluate`·`/report` 라이브에서 `display_scores`·`categories(5)`·`grade` 확인.
- 표정: 입력 `overall=99` 무시하고 **71로 재계산** 확인. 이상치(130/-5/문자) 클램프, 빈 입력 0점.
- 음성: 5케이스 단위검증 — 정상 `wpm73/delivery74`, 필러과다 `ratio0.5/유창성0`, 느리고 긴 휴지 `longest5.0/delivery37`, 영어 `wpm127/delivery97`, 빈 입력 0.
- 통합: `/report`에 expression+voice 주면 `attitude` 77(휴리스틱)→**69**(`72·80·65·55·74` 평균), `overall_categories` 74→72. 신호 빼면 기존대로.

### 설계 원칙
- 표정·음성은 **합격 예측이 아닌 전달력 보조 지표**로 명시(디스클레이머). 원본 영상·오디오는 백엔드로 전송하지 않음(표정은 점수만, 음성은 STT 후 텍스트·지표만).
- `mapping.py`·`voice.py`는 GPU 불필요 순수 함수 — 샌드박스 단위검증 후 배포. 패치는 앵커 문자열 기반 + 멱등 + 백업 + 문법검사.
- 향후 옵션: 음성 전달력을 `communication` 축에 분리 반영(현재는 표정+음성을 `attitude` 비언어 종합으로 합침).

## 2026-06-08 · 그룹 B — Claude API 대체 (base EXAONE 단독, 추가 학습 없음)

프론트의 Claude API 의존 기능 5종을 base EXAONE(use_adapter=False)으로 대체.

### 추가 엔드포인트
- `POST /education/quiz` — 주제 → 객관식 퀴즈 (QuizReq{topic, n, difficulty, lang})
- `POST /resume/cover-letter` — 입력 정보 → 자기소개서 초안
- `POST /resume/polish` — 이력서 문장 다듬기 (사실 유지)
- `POST /posting/generate` — 직무·기술 → 채용공고 초안 (구조화 JSON)
- `POST /posting/analyze` — 공고 원문 → requirements/preferred/keywords 추출

### 설계 원칙
1. 기능별 전용 prefill: 면접 질문용 GEN_PREFILL을 재사용하면 퀴즈가 "면접 질문 모드"로 새는 문제 발생 → 기능마다 prefill 분리(QUIZ/CL/POLISH/POSTING_GEN/POSTING_ANALYZE).
2. 출력 후처리: 퀴즈는 모델이 정답을 "텍스트"로만 내고 보기 셔플 후 Python이 정답 인덱스 계산. 산문은 _strip_thought로 </thought> 이후 본문만 추출 + 특수토큰([|endofturn|] 등) 제거 + 앞/뒤 메타(제목·분량표기·구분선) 제거. JSON은 parse_json_lenient + 정규화(_as_str_list: 중복 제거·줄 분리).
3. 안티 할루시네이션: 분석/다듬기는 "원문·사실만, 항목 추가 금지". 생성(자소서·공고)은 제공 정보 기반 + 임의 수치 금지(연봉 등은 "협의" 같은 일반 표현).

### 트러블슈팅 교훈
- 추출 작업 스키마는 한국어 섹션과 1:1로 둘 것. required_skills+qualifications+preferred 3분류는 모델이 '우대사항'을 qualifications에 잘못 넣음 → requirements(자격요건)+preferred(우대사항) 2분류로 정리하니 정확히 분리됨.
- 자소서 수치 날조(예: '30% 개선', '3배 증가')는 7.8B 생성모델에서 프롬프트만으로 0% 제거 불가. 규칙 강화로 감소했으나 잔존 → "초안 → 본인 검토" 전제로 운영.
- prefill에 규칙 문구를 넣으면 그 문장이 본문으로 베껴 나옴 → prefill은 중립적으로 유지하고 규칙은 프롬프트 본문에만.
- 모델이 내는 집계값(overall, 정답 index 등)은 신뢰하지 말고 Python에서 재계산.

상태: 5종 모두 RunPod 라이브 검증 완료.

## [2026-06-08] QLoRA 재학습 (v3) — completion-only 손실 버그 근본 수정 + 데이터 933 + 어댑터 교체

기존 어댑터(`lora_adapter`)가 사실상 빈 어댑터였음을 발견하고, 원인을 잡아 재학습한 뒤 서버에 교체 연결했다.

### 핵심 버그: loss 0.0 (TRL 응답마커 매칭 실패)
- 기존 `train_qlora.py`가 TRL `DataCollatorForCompletionOnlyLM`을 썼는데, 응답 마커 `[|assistant|]`(토큰 `[3]`)를 실제 시퀀스에서 못 찾아 **라벨 전부 `-100` 마스킹 → loss 0.0**(학습 0). 즉 기존 `lora_adapter`는 학습이 안 된 빈 어댑터.
- (덱의 옛 수치 `loss 2.92→0.83`은 v1 런 값, prod 어댑터는 loss 0.0이라 숫자 불일치 — 이번에 정직한 값으로 교체)
- **해결:** TRL 제거, 라벨 직접 마스킹. `sft_data_clean.jsonl`을 직접 읽어 프롬프트(RUBRIC+Q+A+생성프롬프트+PREFILL)는 `-100`, 완성부(추론+`</thought>`+JSON+EOT)만 학습 → 경계 모호성 0, TRL 버전 비의존.
- 검증: 첫 샘플의 `완성부 앞`이 루브릭이 아니라 추론 텍스트로 시작 → 마스킹 정상. loss가 0.0이 아니라 0.41→0.24로 감소.

### 데이터 확장 (자기증류)
- ICT 200 → **생성 1000 → 품질 게이트 통과 933 (93% 유지)**. 재시작 넘어 자동 이어받기(done 질문 스킵) 동작 확인.
- 품질 게이트(`quality.py`, gen/filter 공용): 핵심 3축(`specificity`·`logic`·`communication`)은 0 불가(모든 답변 적용), `technical_accuracy`는 **인성문항 N/A(0) 허용**. + 피드백 15자↑·한국어, 강점/개선 비-플레이스홀더, 사고 150자↑, 중복질문 제거.
- 게이트 설계 교훈: 처음엔 "0점 하나라도 컷"으로 34%만 통과(탈락 306/473이 전부 tech=0). 분석하니 전부 인성문항의 tech N/A → 핵심 3축만 0 검사로 완화하니 94% 유지. 인성문항 tech=0은 조건부 행동이라 기술문항 채점엔 영향 없음.

### 학습 결과 (`lora_adapter_v3`)
- 933 클린(길이초과 일부 제외 → 학습 837 / 검증 93), r16/α32, all-linear, 4bit base. train_loss 0.41→**0.2377**, eval best **0.3068**(epoch1).
- **best 자동 선택:** `load_best_model_at_end=True` + `EarlyStoppingCallback(patience=2)`. eval_loss가 0.307(ep1)→0.314(ep2)→0.345(ep3)로 2연속 악화 → epoch3에서 조기종료, **epoch1을 최종 어댑터로 자동 저장**(수동 checkpoint 선택 불필요). `save_total_limit=2`.
- 발견: 데이터 2배(447→933)에도 과적합 무릎은 여전히 epoch1. eval_loss는 447 때(0.315)보다 미세하게 낮음(0.307).

### base vs 어댑터 비교 (`compare_base_vs_lora.py`, 8문항·greedy·어댑터 ON/OFF만 차이)
- **v3 7/8 vs base 6/8** — JSON 파싱·스키마 7/8 vs 6/8, `</thought>` 완료 **7/8 vs 5/8**, feedback 7/8 vs 6/8. base가 `[0,0,0,0]`으로 뭉개던 채점을 어댑터는 차등 처리.
- **정직한 한계:** 한국어 추론 유지는 8/8 동률 — prefill 트릭이 base 경로에도 적용되므로 이 부분은 어댑터 공헌이 아님. 발표 시 과장 금지.
- v3 vs best(447·epoch1)는 박빙(7/8 vs 8/8, 서로 다른 1문항에서 삐끗 = N=8 노이즈). **v3 채택 이유**: 데이터 2배로 일반화 기대 + v3 실패방식(JSON 없음 → 서버 `parse_json_tail`이 잡음)이 best 실패방식(0~10 스케일이 조용히 파싱됨)보다 안전.
- 과적합본(epoch3)은 포맷 62%로 **base(75%)보다도 나쁨** → train_loss만으론 과적합 못 잡고, eval+포맷 지표로 판정해야 함을 실증.

### 서버 연결 + 인증
- `server.py` 어댑터 경로 상수를 `lora_adapter_v3`로 교체(1줄, 백업 `server_pre_v3adapter.py`). 한국어 `/evaluate`만 어댑터 사용(`use_adapter = lang=="ko"`), 나머지 전 엔드포인트는 base. 라이브 확인: 4축 85/65/70/68, overall 72(=평균 재계산), 한국어 feedback + `display_scores` 정상.
- **API 키 인증 추가:** `X-API-Key` 미들웨어(`server.py`) + `start.sh`가 `.api_key`를 환경변수로 주입. `/health`·`/docs`·`/openapi.json`은 면제, 나머지는 키 필수. 환경변수 없으면 자동 비활성(하위호환). 공개 proxy URL 보호용.

상태: v3 학습·비교·서버 연결·인증 모두 RunPod 라이브 검증 완료.

### 2026-06-11 — 추론 성능 개선 (스트리밍 · 직렬화 락 · 토큰 로깅)
- 문제: `/interview/evaluate` 지연 ~1분/호출, 동시·중단 요청 시 CUDA device-side assert 크래시.
- 진단: 지연은 생성 토큰 수에 비례(EXAONE-Deep 추론 체인). `max_new_tokens` 상한엔 거의 안 닿음 → 생성 바운드. 크래시는 동시/중단 요청이 CUDA 컨텍스트를 오염시킨 것(클린 재시작으로 복구).
- 조치(품질 중립 3가지):
  1. `threading.Lock`으로 `generate` 직렬화 → 동시요청 크래시 방지(단일 워커).
  2. 생성 토큰·시간 로깅 추가 (`>>> [gen] Ntok / Ss = tok/s`).
  3. SSE 스트리밍 엔드포인트 `/interview/evaluate/stream` 추가(`TextIteratorStreamer` + 백그라운드 스레드, 같은 락). 기존 `/interview/evaluate`는 변경 없음.
- 결과: 스트리밍으로 체감 지연 개선, 동시요청 안전. 라이브 검증 통과(스트리밍/비스트리밍 점수 동일).

### 2026-06-11 — 평가·지표: base vs LoRA v3 비교 하네스
- 평가셋: 직접 작성한 채점용 Q&A 18건(질문 6 × 상/중/하 3단계, 기술 4 + 인성 2). 포트폴리오 공개 안전(AI Hub 원본 미사용).
- 하네스(`eval_compare.py`): base(어댑터 OFF) vs LoRA v3(ON)를 동일 조건(`MAXTOK=2048`, greedy)으로 채점 비교. 재개 가능(케이스별 즉시 저장).
- 결과(n=18):
  - 파싱 성공률: base 78% → LoRA 83%
  - 한국어 사고율: base 100% / LoRA 100%
  - 변별력(상/중/하 평균 overall): base 85 / 63 / 22, LoRA 84 / 66 / 26
- 관찰: 어댑터가 출력 형식 안정성을 소폭 끌어올림. 변별력은 양쪽 모두 양호 — 베이스 EXAONE-Deep이 이미 강해 LoRA의 이득이 압도적이진 않음. 지연은 LoRA가 평균적으로 더 길게 생성(일부 2048 캡 도달)하기 때문.

## 2026-06-12 — 추론 속도/안정성: 예산 강제(budget forcing) + 생성형 샘플링

### 배경
- 일부 평가 케이스에서 추론이 수렴하지 못하고 상한까지 생성됨. 프로덕션(상한 4096)에서 ~130초 소요 + Cloudflare 524(원본 ~100초 타임아웃) 발생, 점수도 미산출.

### 조기 종료(early-stop) 검토 → 데이터 기반 기각
- 진단(자체 스크립트, n=5): 깨끗한 케이스는 JSON 완성 직후 EOS로 종료(완성후토큰=1) → 자를 토큰 없음. 실패 케이스는 상한(2048)까지 가도 valid JSON 자체를 못 만듦(EOS=False) → "JSON 완성 시 종료" 트리거가 걸리지 않음.
- 결론: 이 실패 양상엔 조기 종료가 무효. 미적용.

### budget forcing 도입 (채택)
- 추론 단계를 REASON_BUDGET(1600토큰, 측정된 정상 최대 1156 위)으로 제한:
  - 예산 내 자연 종료 → 그대로 (깨끗한 케이스 영향 0)
  - 예산 소진 + `</thought>` 있음 → 남은 원래 예산으로 답변 마저 생성
  - 예산 소진 + `</thought>` 없음(폭주) → `\n</thought>\n` 강제 주입 후 ANSWER_BUDGET(768)로 답변 생성, `[FORCED]` 로깅
- 검증(하네스, baseline 2048 vs budget 1600+400):
  - 깨끗한 케이스(t1/t4): 변화 없음(파싱 O, 강제 -)
  - 실패 케이스(t3/b2-avg/b2-weak): 파싱 X→O 회복, 토큰 2048→~1720-1751, 시간 ~68s→~57-59s
  - 효과: 프로덕션 4096 폭주(~130초) → ~2000토큰(~57초) 단축 + 524 회피 + 점수 산출
- 트레이드오프: 강제로 끊긴 케이스는 점수 품질이 다소 낮을 수 있으나, 무효(524/실패)보다 유효 점수 우선. 원래 실패하던 ~17%만 영향.

### 생성형 엔드포인트 선택적 샘플링
- run_llm에 do_sample/temperature(0.8)/top_p(0.9) 옵션 추가.
- 생성형 6개(질문 생성·꼬리질문·퀴즈·자기소개서·이력서 첨삭·공고 생성)에만 do_sample=True → 호출마다 다양한 결과.
- 평가·리포트·공고분석은 그리디(do_sample=False) 유지 → 점수·분석 일관성(결정론).

### 적용/운영 메모
- server.py: 상수(BUDGET_FORCE/REASON_BUDGET/ANSWER_BUDGET) + run_llm 본문 교체 + 생성형 호출부 6곳 do_sample=True. 백업(server.py.bak) + py_compile 검증, 멱등 패치. 새 의존성 없음.
- Colab 배포는 Drive의 server.py를 복사해 실행 → server.py 변경 시 Drive 재업로드 필요.

---

### 2026-06-13 — /interview/followup 꼬리질문 적응형 강화

**목적**: 모의면접에서 첫 질문 이후의 꼬리질문을 답변 내용에 따라 유동적으로 생성. 특히 답변이 모호할 때 구체화를 직접 유도하도록 개선.

**방법** (`server.py`, 멱등 패치 + 백업 + py_compile):
- `fu_prompt`에 2단계 구조 도입 — [1단계] 답변을 진단(길이·구체성·근거·깊이), [2단계] 진단에 따라 분기: 짧음/두루뭉실/근거부족 → 구체적 사례·상황·수치·이유를 요구 / 구체적·충실 → 선택 이유·대안 비교·트레이드오프·한계로 심화.
- 기술 용어(Redux, React Query 등) 원문(영문) 유지 규칙 추가 → 음차 옦류 방지.
- `FollowupReq`에 `history: list = []` 필드 추가(하위호환). 제공 시 이전 문답을 프롬프트에 주입해 멀티턴 중복 회피·심화.
- `FU_PREFILL`(사고 프리필)을 답변 진단 지향 문장으로 교체.
- followup은 base 모델(use_adapter=False) + do_sample=True 유지.

**결과** (동일 질문 "React 상태 관리 어떻게?", 강화 전 vs 후):
- 모호한 답 케이스: 강화 전 = 단순 기술질문(useReducer vs useState 선호?) → 강화 후 = 선택 이유 + 대안(Redux/Context API) 고려 여부 + 구체적 사례 요구.
- 구체적 답 케이스: 강화 전 용어 음차 오류("레드소스 툴킷") → 강화 후 "Redux Toolkit" 정상 표기 + 선택 이유·기존 방법 차이 심화.

**검증**: stub로 4개 치환(FU_PREFILL/fu_prompt/FollowupReq/호출부) 각 1회 + py_compile + 멱등성 + 실제 프롬프트 생성(한·영, history 유무) 확인 후 서버 적용.

---

### 2026-06-13 — 자소서/이력서 분석 엔드포인트 진단: 빠진 고리 및 한계

**현황 평가**
- ✅ `/interview/generate` (이력서+공고 기반 맞춤 질문 생성): 완성. GenerateReq로 resume/job_posting 필수 입력, gen_prompt에서 "이력서의 경험·기술과 공고 요구사항을 연결한 맞춤형 질문" 명시.
- ✅ `/resume/cover-letter` (자소서 생성): 완성. CoverLetterReq로 직무·회사·경력·스킬·프로젝트 등 입력, 자소서 문체 생성.
- ✅ `/resume/polish` (이력서 문장 첨삭): 완성. 한 항목(불릿/문장)을 명확·임팩트 있게 수정.
- ❌ `/resume/analyze` (자소서/이력서 통짜 분석·진단): 빠짐.

**빠진 고리 — `/resume/analyze` 미구현**
기획서 슬라이드 6의 ③맞춤 피드백(약점 진단·보완 추천)은 자소서/이력서를 다음처럼 분석·진단하는 것을 의도:
- 강점 & 약점 항목별 도출
- 구체성 평가 (구체적 사례·수치·기술 유무)
- STAR 구조 적합도 진단 (상황-태스크-액션-결과 명확도)
- 공고 키워드 정합도 (지원 직무와의 관련도)
- 개선 제안 + 점수(예: 전체/구체성/STAR/정합도 각 점수)
- 출력: JSON {strengths, weaknesses, analysis, improvement_suggestions, scores}

현재 능력: 이력서로 질문을 뽑고(generate), 문장을 고치지만(polish), 문서 전체를 진단하지 않음. 5축 채점은 면접 답변용.

**한계 (정직하게)**
1. **검증 불가** — "분석"은 LLM의 읽기·추론이지 사실 검증이 아닙니다. 좋은 진단을 뽑지만, 자소서가 빈약하거나 거짓 정보를 담으면 일반적 피드백만 나오거나 없는 내용을 추정할 수 있습니다. 프롬프트로 "문서에 명시된 것만 기반으로"를 강제해 완화 가능하나 100% 방지 불가.

2. **주관성** — 자소서를 "잘 썼나" 판단은 구조·구체성·STAR·키워드 정합성 같은 객관 항목도 잘 평가하지만, 본질적으로 주관적입니다. 데모·포트폴리오로는 "AI가 이렇게 피드백한다"는 설득력 충분하나, 실제 채용 합격 판정을 대체할 수 없습니다. 보조 도구 포지션.

3. **개인정보** — 자소서/이력서엔 이름·전화·이메일·프로젝트 세부 등 PII(개인식별정보)가 있습니다. 서버 전송·로깅·보관 시 암호화/마스킹은 팀 백엔드(Spring)에서 담당해야 합니다. AI 서버는 진단만 하고 문서는 보관 금지 권장.

**다음 단계**
- `/resume/analyze` 엔드포인트 구현 (시간 있으면 추가)
- 또는 ②④⑤ 독립 데모 UI 완성 후 나중에 추가
- DEVLOG에 구현 시 방법/결과 기록


---

## 2026-06-13 — /resume/analyze: 자소서/이력서 분석 (하이브리드 채점)

### 목적
자소서/이력서를 입력받아 구체성·STAR·직무적합도·종합 점수와 강점/약점/개선제안/총평을
반환하는 분석 엔드포인트. (기획보고서 로드맵 ③ 맞춤 피드백)

### 문제 — base 모델 단독 채점 실패
- LLM에 점수와 피드백을 한 번에 생성시키면 프롬프트 예시 점수(6/5/4)에 앵커링하여,
  모호한 이력서와 구체적인 이력서가 거의 동일한 점수를 받음(차등 실패).
- 채점 기준 명시·"예시 복사 금지" 지시를 추가해도 효과 미미.
- 수치가 풍부한 이력서에 "정량적 결과 부족"이라는, 문서와 모순되는 환각 약점 생성.

### 해결 — 하이브리드 + 환각 필터 (v5)
- 점수 = 파이썬 휴리스틱(결정론적):
  - specificity: 숫자/%/기술용어/성과어 밀도 -> clamp(2 + raw*0.45)
  - star: STAR 4요소(상황/과제/액션/결과) 키워드 존재 개수 -> {0:1,1:3,2:5,3:7,4:8},
    3요소 이상 + 숫자 존재 시 +1
  - job_fit: 공고 토큰과의 겹침 비율 -> clamp(2 + overlap*10), 공고 없으면 0
  - overall: 가중합(공고 有 0.40/0.35/0.25, 無 0.55/0.45)
- 질적 피드백 = LLM(base 모델, greedy):
  - 계산된 점수를 프롬프트에 주입 -> 점수와 모순 방지
  - "문서에 없는 내용 지어내지 말 것" 근거 강제
  - 구체성 >= 8이면 "지표 부족 류 약점 금지, 깊이·직무연관성·명확성에서 찾아라" 가드 삽입
  - 후처리 필터: 구체성 >= 8일 때 "지표/수치/정량 + 부족/부재" 패턴 약점을 결정론적 제거(안전망)

### 결과
- 모호한 이력서: overall 2 / specificity 2 / star 3
- 구체적 이력서: overall 9 / specificity 10 / star 8
- 점수 명확 차등 달성 + 환각 약점 제거.
  (구체 이력서 약점이 "수치 부족" -> "장기적 영향력/기술 과정 설명 부족"으로 전환,
   프롬프트 가드가 작동하여 필터는 미발동.)

### 한계
- 휴리스틱은 표면 신호(숫자·키워드 카운트) 기반이며 의미 이해가 아님.
- 질적 피드백은 여전히 LLM 생성이라 일반론 가능성 존재 -> 점수가 신뢰 가능한 핵심 신호.
- 자소서는 PII 포함 -> 백엔듗(Spring)에서 암호화/마스킹, AI 서버는 문서 미저장.


## 2026-06-17 — 적응형 레벨테스트(Adaptive Level Test)
- 추가: POST /leveltest/next(객관식 적응 루프, 즉시채점→문항마다 난이도 실시간 조정),
  POST /leveltest/session-next(서술형 세션 5축 점수→다음 난이도+로드맵).
- 설계(하이브리드): 난이도 숫자/문항선택=규칙 기반(결정론적·무상태)로 'LLM이 난이도를 직접 정해
  점수와 어긋나는' 환각을 원천 차단. 추천 텍스트=base EXAONE 생성, 계산된 결과 수치만 근거로 강제,
  미준비/실패 시 규칙 기반 폴백. 상태(answers)는 요청에 실려오고 정답은 서버 은행에만 보관(qid만 교환)
  → 재시작·동시요청 안전, DB 불필요.
- 구현: 내장 객관식 은행 30문항(fe/be/cs × 하/중/상). 난이도 규칙=시작'중', 정답+1/오답-1(1~3 클램프).
  프로필 role로 토픽 필터, 레벨=정답률+도달난이도→입문/초급/중급/고급.
- 검증: 순수 엔진 시뮬레이션(EXAONE 불필요)에서 강한 사용자 중→상 수렴(고급)/약한 사용자 중→하(입문)/
  혼합 중↔상 적응, 토픽 필터·결정론·중복방지·폴백 통과. 통합 py_compile OK.


## 2026-06-17 — 교육 'AI 학습 노트' 엔드포인트 (/education/lesson)
- 커리큘럼 노드(토픽)별로 EXAONE이 핵심 학습 노트를 실시간 생성. POST /education/lesson(비스트리밍) + POST /education/lesson/stream(SSE, <thought> 추론 숨기고 본문만 스트리밍).
- RAG(FAISS/bge-m3)로 토픽 관련 면접 질문을 검색해 근거로 주입(면접 연관 개념 중심). 범위는 핵심 개념/요점/흔한 실수 요약으로 한정, base 모델(disable_adapter) 사용.
- 주제 이탈 가드: 관련 질문을 유사도 0.60 미만 컷 + 상위 5개만 프롬프트 주입. 7.8B 모델이 음성 지시("무시하라")를 잘 안 따르는 문제를, tangential 질문을 애초에 프롬프트에서 빼는 방식으로 해결(예: '스택과 큐' 노트에 그래프 섹션이 끼던 문제 제거 확인).
- 스트리밍은 기존 evaluate/stream 패턴 재사용(TextIteratorStreamer + GEN_LOCK).

## 2026-06-27 — 면접 답변 STAR 채점 (LLM 평가, 1~5 앵커 척도)

### 목적
모의면접 답변 평가(evaluate)에 STAR 구조 점수를 추가한다. 기존 5축(기술정확도·구체성·논리·의사소통)에 더해, 답변이 STAR(Situation/Task/Action/Result)를 얼마나 갖췄는지를 항목별로 채점한다. 프론트는 라이브 리포트와 저장된 세션 상세에서 STAR 레이더를 표시한다(DB 컬럼 situation/task/action/result_score는 이미 존재).

### 배경 — 기존 휴리스틱의 한계
프론트에 STAR 산출 함수(analyzeSTAR)가 있었으나 키워드 카운트 방식이었다: 카테고리별 키워드 포함 개수 × 30 + 길이 보너스. 짧거나 지정 키워드가 없는 답변은 전부 0점이 나와, 라이브 리포트에서 STAR가 0으로만 표시됐다. 배선·저장·조회는 정상이었고 산출 로직만 문제. resume/analyze의 STAR도 키워드 카운트지만, 면접 답변은 같은 모델이 5축을 이미 LLM으로 채점하므로 STAR도 같은 호출에서 LLM이 판단하게 하는 것이 자연스럽다.

### 문제 — 0~100 척도에서 무한루프
처음엔 5축과 동일하게 STAR를 0~100 정수로 채점하도록 루브릭에 추가했다. 비스트리밍은 정상 작동(경험질문 S85/T80/A80/R85, 기술질문 0/0/0/0으로 정확)했으나, 앱이 쓰는 스트리밍(SSE)에서 같은 입력이 가끔 무한루프에 빠졌다. 토큰 덤프를 보면 모델이 점수를 다 정해놓고도("기술 0, 구체성 4...") JSON으로 전환하지 못하고 "하지만 문제에서... 평균을... 하지만..."을 17회 반복하다 cap에 걸려 파싱 실패(ok:false). 같은 답변이 어떤 호출은 성공, 어떤 호출은 실패하는 불안정성이라 데모에 치명적.

### 시행착오 — 디코딩 파라미터는 실패
- repetition_penalty=1.3: 단어 회피 폭주 유발(한국어 채점인데 "Foreign Affairs Diplomacy..." 영어 단어를 끝없이 나열). 1.15로 낮추니 이번엔 </think> 종료 토큰을 무한 반복. 값을 어떻게 조정해도 새로운 종류의 폭주가 생겨 폐기. 모든 토큰에 페널티를 거는 방식이 종료·정상 생성까지 망가뜨림.
- no_repeat_ngram_size=4: 루프는 잡혔으나(10/10 성공) 채점이 왜곡됨. 점수 매기기는 본질적으로 숫자·항목명이 반복되는데, n-gram 반복 금지가 이를 방해해 "이틀 앞당겼다"는 정량 결과가 있는 답변의 result가 D로, 행동이 분명한 답변의 action이 N/A로 나옴. 루프와 채점 정확도가 상충.

### 해결 — 1~5 앵커 척도 재설계 (벤치마킹 반영)
실제 기업(Google·Amazon·McKinsey 등)이 STAR를 1~5(또는 1~4) 척도에 행동 기준(behavioral anchor)을 붙여 채점한다는 점에 착안. 0~100은 사람도 LLM도 변별 못 하고 모델이 "30? 35?" 망설이다 루프에 빠지지만, 5단계는 "이건 일반적이니 3"으로 즉시 결정된다.
- 척도 변경: STAR를 항목별 1~5 정수로. 각 점수에 앵커 명시(예: action 5=본인 행동을 단계적으로/3="우리"가 많거나 추상적/1=가설형 "~할 것이다"). 벤치마킹 기준 반영("우리" 남발 감점, 가설형 강등, 정량 결과 없으면 result 낮게).
- 루프 차단(루브릭): "점수를 정한 뒤 같은 고민을 반복하지 말고 곧바로 JSON 출력" + "1~5는 5축(0~100)과 무관한 별도 척도" 명시. 디코딩이 아닌 지시로 루프 해결 → 채점 왜곡 없음.
- 후함 보정: 초기 루브릭이 모든 항목 A를 줘서(평범한 답변이 100점) 우수 답변과 역전됨. "평범한 답변은 대부분 2~3점, 5는 모범적일 때만, 4는 명확히 우수할 때만, 모든 항목에 높은 점수 금지" 분포 명시로 변별력 회복.

### 등급 환산 (clamp_star)
LLM은 1~5만 출력하고, 코드가 점수·등급으로 환산(모델 부담 최소, 일관성 보장).
- 개별: 1~5 → ×20 → 0~100(프론트 e.star.S 호환) + 등급 A~F(5=A/4=B/3=C/2=D/1=F). 미평가(0)는 N/A.
- 종합: 4항목 평균 → 0~100 + 세분화 등급(A+/A/A-/B+/B/B-/C+/C/C-/D/F). 종합에만 +/- 부여(개별은 1~5라 +/- 불가, 평균은 연속값이라 가능).
- 미평가 항목은 종합 평균에서 제외.

### 응답 구조
evaluate 응답의 evaluation에 4개 필드 추가(비스트리밍·스트리밍 양쪽):
- star: {S,T,A,R} 0~100 (기존 프론트 호환)
- star_grade: {S,T,A,R} A~F
- star_overall: 0~100 종합 점수
- star_overall_grade: A+/B- 등 세분화 등급

### 결과
스트리밍 반복 테스트(답변별 5회) 전부 ok:true, 루프 박멸. 변별 정상:
- 우수(갈등해결, 정량결과 有): 종합 90 A+ (R=A로 정량결과 정확 반영)
- 평범(팀장 역할 서술): 종합 60 B-
- 기술(인덱스 설명, STAR 무관): 낮게
역전 해소(우수 > 평범), 매 호출 일관. 5축 채점은 기존대로 정상 동반.

### 한계
- 1~5는 5단계라 개별 항목의 미세 변별은 제한적(종합 등급에서 +/-로 보완).
- 순수 기술질문의 STAR는 보조 지표(주 평가는 5축). 기술질문에 STAR가 다소 높게 나올 수 있으나 실사용 영향은 작음.
- cap은 evaluate만 2048(STAR 추가로 사고가 길어짐), followup/resume은 1536 유지.

### 부수 작업
- T3 라우팅: 챗봇의 면접 유도 임계값(INTERVIEW_HINT) 0.55→0.60 상향("오늘 날씨" 류 무관 질문이 면접으로 잘못 분류되던 문제 완화).


## 2026-07-01 — 챗봇 /chat 하이브리드 RAG + SSE 스트리밍 + 사고누출 방어 + FAQ 확충

기존 /chat은 FAQ 검색(bge-m3 FAISS, 60개)으로 캔드/고정문구만 반환하고 LLM 생성이 없었다. 검색은 주제를 맞히지만 캔드 답이 질문의 구체 의도를 못 맞추는 경계 질문(환불·순서·카메라 등)에 엉뚱한 답이 나갔다. 이를 3구간 하이브리드로 개편하고, 스트리밍·사고누출 방어·FAQ 확충까지 진행.

### 3구간 라우팅 (실측 임계값)
- curl 실측 점수 분포: 강매칭 0.94~0.99(회원가입 0.99 / 비번 0.95 / 면접점수 0.98 / 이력서 0.98), 경계 0.62~0.77(환불 0.62 / 순서 0.73 / 카메라 0.77).
- 경계 구간은 "검색은 주제 맞음 + 캔드 답은 의도 못 맞춤"이라 LLM이 가장 값어치 있는 지점.
- 채택: >=0.80 캔드 즉답(source=faq) / 0.55~0.80 top-3 FAQ 근거 LLM 생성(source=faq_llm, use_adapter=False·근거강제 프롬프트·생성 실패 시 최상위 FAQ 폴백) / <0.55 면접유도(interview)->폴백(none).
- 임계값 근거: 0.62 캔드 유지였으면 "느슨한 매칭"(예: 환불 0.62가 결제방법 캔드로 나감)이 계속 엉뚱하게 나감 -> 캔드 상한을 0.80으로 올려 경계를 LLM 구간으로 넘김.

### SSE 스트리밍 (/chat/stream)
- 기존 lesson/evaluate stream 패턴 재사용: TextIteratorStreamer(skip_prompt·skip_special_tokens=False) + Thread + GEN_LOCK, data: {json} 형식(token/done/error).
- faq_llm 구간만 token 스트리밍, 캔드·폴백은 생성이 없어 done 이벤트 하나로 즉답.
- 사고 숨김: </thought> 이전 토큰은 버퍼링, 이후 실답변만 흘림(자체 _chat_body_pieces).
- done 페이로드 = 비스트리밍 /chat과 동일(source·score·category·answer·matched_question) -> 프론트가 스트리밍/비스트리밍을 동일 처리.

### 사고누출 버그 & 해결 (기저 버그)
- 증상: 스트리밍 답변에 모델의 사고 과정("사용자는 물었습니다... 어떻게 답할까요?...")이 통째로 노출. 같은 질문이 어떤 호출은 정상, 어떤 호출은 누출되는 랜덤성이라 데모에 치명적.
- 원인: do_sample=True + max_new_tokens=512에서 사고가 예산 내 안 끝나 </thought> 미출력 -> _strip_thought는 "</thought> 없으면 전체 반환" 설계(다른 엔드포인트용)라 챗봇에선 사고 전체가 답변으로 튐. 스트리밍은 </thought>를 못 만나 버퍼에 갇혀 done에 통짜로 나옴.
- 해결 3종(_strip_thought 공용 함수는 미수정, 챗봇 경로만 국소 수정): (1) 예산 512->1536(사고 종료 공간 확보) (2) do_sample->greedy(사고 결정적 종료·답변 일관성, lesson과 동일) (3) </thought> 없으면 answer 버리고 캔드 폴백(누출 원천 차단).
- 검증: 같은 질문 4회 반복 글자까지 동일 + 사고 문구 0. 폴백 로직 시뮬레이션(정상/사고만/빈값)도 통과.

### 특수토큰 필터
- 스트림 token에 [|endofturn|] 등 노출 -> _chat_body_pieces 흘리기 직전 정규식 [|...|]/<|...|> 제거 + 빈조각 skip. done은 이미 _strip_thought 처리라 무변.

### FAQ 확충 (60->68)
- 실측 갭 8개를 요구사항정의서 근거로 신규 작성(학습·면접 순서 / 카메라 필수 / 구독·결제 / 표정분석 프라이버시 / 꼬리질문 / 음성 답변 / 면접 설정 / 환불). question 필드가 사용자 말투에 가깝게.
- 환불은 팀에서 정책 미정 -> 조건("7일 100%" 등) 지어내지 않고 안내형(결제 정보 메뉴 확인·챗봇 문의)으로만.
- build_faq_rag.py --force로 bge-m3 재빌드(CLS 풀링 + L2 정규화, server.py embed_query와 동일 스케일), 옛 인덱스 .bak 백업. faq.jsonl은 append.
- 재테스트: 순서 0.73->0.96, 카메라 0.77->1.00, 환불 0.62->0.94. 셋 다 강매칭 캔드로 전환, 질문에 맞는 답.

### 원칙 / 남은 것
- 하이브리드 원칙 유지: 숫자·캔드=규칙, 서술만 LLM. 근거강제 프롬프트로 없는 서비스 정보 지어내기 차단.
- 후속(팀 연동): FAQ 소스를 파일->DB(chatbot_faq)로, Spring /api/chat/stream SSE 프록시. (일지엔 FAQ 답변 원문·AI Hub 원본 미포함.)

## 2026-07-02 — 추천 매칭 설명 /recommend/explain (교육·공고 공용)

### 목적
- E-013(이력서+레벨테스트 기반 추천)의 "왜 이게 당신에게 맞는지" 서술 생성. 랭킹 점수는 Spring이 결정론 휴리스틱으로 산출하고, 설명 문장만 이 엔드포인트가 담당.

### 설계 — base + prefill, 배치 1회 호출
- followup 계열과 동일: base 모델(use_adapter=False) + 한국어 prefill(EXPLAIN_PREFILL) + parse_json_lenient.
- items 전체를 한 번의 generate로 처리 → target_id별 reason을 배열로 반환. 후보 N개를 N번 호출하지 않아 trycloudflare ~100초 타임아웃 회피.
- 응답: {ok, target_type, reasons:[{target_id, reason}]}. target_id 키잉이라 Spring이 순서 무관하게 매핑.

### target_type 공용 (교육·공고)
- 요청 target_type(COURSE/POSTING)을 응답에 그대로 되돌림 → 공고 담당 팀원이 POSTING으로 같은 엔드포인트 재사용. AI 설명 로직을 교육·공고에 하나로 통일.

### 환각 방지 / 폴백
- 프롬프트가 제공된 matched(걸린 매칭 근거)에만 기반하도록 강제("없는 내용 지어내지 말 것"). 프로필 원문이 아니라 구조화 신호(job_category/level/skills)만 전달.
- 파싱 실패 시 ok:false 반환 → Spring 측 규칙 문장 폴백과 짝. AI 엔드포인트가 없어도 /api/recommend/courses는 완전 동작.

### 원칙 / 검증
- 하이브리드 유지: 점수=Spring 휴리스틱(결정론), 서술만 LLM. 랭킹에 GPU 불필요 → 파드 off 상태에서도 추천 자체는 동작.
- 검증: 파드에서 2항목 배치 호출 → ok:true, target_id 2개 모두 matched 근거 기반 한국어 이유 생성 확인. (일지엔 근거 문장 원문·AI Hub 원본 미포함.)
