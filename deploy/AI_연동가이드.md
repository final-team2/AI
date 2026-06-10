# AI 연동 가이드 (API 계약)

프론트(React)·백엔드(Spring) 팀이 AI 서버(FastAPI)를 호출할 때 필요한 규약.

## 호출 구조

```
React ──(JWT)──▶ Spring Boot ──(서버-투-서버 · X-API-Key)──▶ FastAPI (AI)
```

- 브라우저(React)가 AI 서버를 **직접 부르지 않는다.** Spring이 JWT로 사용자 인증을 끝낸 뒤, 서버-투-서버로 AI를 호출한다.
- AI 키(`X-API-Key`)는 **백엔드에만** 둔다(프론트로 노출 금지).

## Base URL

- 호스팅 방식에 따라 주소가 다르다:
  - **RunPod**: `https://<id>-8000.proxy.runpod.net`
  - **Colab(임시)**: `https://<무작위>.trycloudflare.com`
- **이 주소는 서버를 다시 띄울 때마다 바뀐다.** 관리자가 새 주소를 공유하므로, 백엔드는 이 값을 **환경변수/설정**으로 빼두고 하드코딩하지 말 것.
- 어느 방식이든 아래 인증·스키마·타임아웃 규약은 동일하다.

## 인증

- `/health`, `/docs`, `/openapi.json` 외 **모든 엔드포인트는 `X-API-Key` 헤더 필수.**
- 헤더: `X-API-Key: <키>` (키는 관리자에게 별도 전달)
- 키 없거나 틀리면 401.

## 권위 있는 스키마 = Swagger

모든 엔드포인트의 **정확한 요청/응답 필드**는 서버에서 자동 생성되는 Swagger 문서가 기준이다:

```
<BASE_URL>/docs          # 브라우저로 열어 확인 (인증 없이 접근 가능)
<BASE_URL>/openapi.json  # 기계가 읽는 스펙
```

아래는 자주 쓰는 것들의 예시이며, 필드가 늘거나 바뀌면 `/docs`가 최신이다.

## 공통 규약

- 응답은 `{"ok": true/false, ...}` 형태. **`ok`를 먼저 확인**하고 false면 `error` 메시지를 처리.
- 추론 모델이라 응답에 **20~60초**가 걸린다 → 백엔드 타임아웃 **90~120초**, 프론트에 로딩 UI 필수.
- Pod가 꺼져 있을 수 있으니, 호출 실패 시의 **폴백 메시지**를 둘 것.
- 점수는 모두 0~100 정수. 종합점수(overall)는 서버가 4축 평균으로 재계산한 값.

## 핵심 엔드포인트 예시

### 답변 채점 — `POST /interview/evaluate`
요청
```json
{ "question": "REST API란 무엇인가요?", "answer": "자원을 URI로...", "lang": "ko" }
```
응답
```json
{
  "ok": true,
  "scores": { "technical_accuracy": 85, "specificity": 70, "logic": 80, "communication": 82 },
  "overall": 79,
  "display_scores": { "logic": 80, "clarity": 82, "depth": 78 },
  "strengths": ["..."],
  "improvements": ["..."],
  "feedback": "..."
}
```

### 퀴즈 생성 — `POST /education/quiz`
요청
```json
{ "topic": "HTTP", "n": 5, "difficulty": "medium", "lang": "ko" }
```
응답은 보기·정답 인덱스가 포함된 문항 배열(정답 인덱스는 서버가 셔플 후 계산). 정확한 형태는 `/docs` 확인.

### 그 외
`/interview/question`(RAG 질문검색) · `/interview/followup`(꼬리질문) · `/interview/generate`(이력서·공고 기반 질문) · `/interview/report`(종합 리포트) · `/interview/expression`(표정 점수 수신) · `/interview/stt`(음성→텍스트+전달력) · `/resume/cover-letter` · `/resume/polish` · `/posting/generate` · `/posting/analyze` — 요청/응답 필드는 `/docs` 참조.

## Spring 호출 예시 (WebClient)

```java
WebClient ai = WebClient.builder()
    .baseUrl(aiBaseUrl)                       // 환경변수 (재시작마다 갱신)
    .defaultHeader("X-API-Key", aiApiKey)     // 백엔드에만 보관
    .build();

Map<String, Object> body = Map.of(
    "question", question, "answer", answer, "lang", "ko");

EvalResponse res = ai.post()
    .uri("/interview/evaluate")
    .bodyValue(body)
    .retrieve()
    .bodyToMono(EvalResponse.class)
    .timeout(Duration.ofSeconds(120))         // 콜드/추론 시간 고려
    .block();
```

## 운영 체크리스트

- [ ] AI Base URL을 설정값으로 분리(재시작마다 갱신)
- [ ] `X-API-Key`는 백엔드 환경변수에만
- [ ] 타임아웃 90~120초, 프론트 로딩 UI
- [ ] `ok==false` / 호출 실패 폴백 처리
- [ ] 연동 전 `<BASE_URL>/health`로 `ready: true` 확인
