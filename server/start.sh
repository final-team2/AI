#!/usr/bin/env bash
# ============================================================
#  Interview AI 서버 시작 스크립트 (실무 수준)
#  Pod를 stop/start 한 뒤 이 한 줄이면 됩니다:
#      bash /workspace/interview_ai/start.sh
#  - 포트 8000 (Jupyter가 8888 사용 -> 충돌 없음, Jupyter는 건드리지 않음)
#  - /health 가 ready:true 를 반환할 때까지 대기 (로그 문자열에 의존하지 않음)
#  - 로딩 중 서버가 죽으면 즉시 로그 출력
# ============================================================
PROJECT_DIR="/workspace/interview_ai"
PORT=8000

cd "$PROJECT_DIR" || { echo "[실패] $PROJECT_DIR 를 찾을 수 없습니다."; exit 1; }

# HF 캐시를 프로젝트 폴더로 고정 (Pod 재시작 후 재다운로드 방지)
export HF_HOME="$PROJECT_DIR/hf_cache"
export API_KEY="$(cat /workspace/interview_ai/.api_key 2>/dev/null)"

# 1) 기존 서버만 정리 (Jupyter는 8888이라 그대로 둔다)
echo ">>> 기존 uvicorn 정리..."
pkill -f "uvicorn server:app" 2>/dev/null
sleep 2

# 2) venv 확인 + 활성화
if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "[실패] venv가 없습니다 (영구화가 안 된 상태)."
    exit 1
fi
source "$PROJECT_DIR/venv/bin/activate"

# 3) server.py 존재 확인
if [ ! -f "$PROJECT_DIR/server.py" ]; then
    echo "[실패] server.py가 없습니다: $PROJECT_DIR/server.py"
    exit 1
fi

# 4) 서버 백그라운드 실행
nohup uvicorn server:app --host 0.0.0.0 --port "$PORT" > server.log 2>&1 &
SERVER_PID=$!
echo ">>> 서버 시작 (PID $SERVER_PID, 포트 $PORT). 모델 로딩 대기 중 (1~2분)..."

# 5) /health 가 ready:true 가 될 때까지 대기 (최대 4분)
for i in $(seq 1 120); do
    # 로딩 중 죽었는지 먼저 감시
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[실패] 서버가 로딩 중 종료됐습니다. 마지막 로그 20줄:"
        tail -n 20 server.log
        exit 1
    fi
    # health 체크 (로딩 완료 전엔 연결 실패 -> 계속 대기)
    if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -qE '"ready": ?true'; then
        echo ""
        echo ">>> [준비 완료]"
        echo -n ">>> 내부 health: "
        curl -s "http://localhost:$PORT/health"; echo
        if [ -n "$RUNPOD_POD_ID" ]; then
            echo ">>> 외부 주소 : https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net/health"
        fi
        exit 0
    fi
    sleep 2
done

echo "[경고] 4분 내 준비되지 않음. 확인: tail -f $PROJECT_DIR/server.log"
exit 1
