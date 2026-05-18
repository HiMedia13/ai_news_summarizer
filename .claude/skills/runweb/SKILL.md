---
name: runweb
description: Flask 뉴스 앱(app.py)을 백그라운드로 띄우고 사용자에게 http://localhost:5000 안내. 검색·요약·브리핑 웹 UI를 빠르게 띄울 때 사용.
---

# /runweb — Flask 뉴스 앱 실행

## 절차

1. **포트 점유 확인.** 5000번 포트가 이미 사용 중인지 점검:
   ```powershell
   netstat -ano | findstr :5000
   ```
   결과가 있으면 사용자에게 "5000 포트가 이미 사용 중입니다 — 기존 프로세스를 종료할까요?" 라고 묻고, 사용자 확인 후 진행.

2. **`.env` 점검.** `OPENAI_API_KEY`가 없으면 앱이 분석 없이 빈 결과만 내므로 미리 알린다:
   ```powershell
   if (-not (Test-Path .env)) { Write-Output ".env 파일 없음 — OPENAI_API_KEY 미설정 상태로 실행" }
   ```

3. **백그라운드 실행.** Bash 도구의 `run_in_background: true`로 `python app.py` 실행. (Flask debug는 기본 off — `FLASK_DEBUG=1`을 설정하려면 사용자에게 확인.) 반환되는 `shell_id`(또는 background id)를 기억해 둘 것.

4. **준비 확인.** Monitor 도구(deferred — 필요 시 `ToolSearch query: "select:Monitor"`로 로드)로 백그라운드 작업의 stdout에서 `Running on http://127.0.0.1:5000` 라인이 나올 때까지 대기. timeout 15초. Monitor 사용이 어려우면 약간 기다린 뒤 `BashOutput`/`KillBash`로 대체.

5. **안내.** 사용자에게 다음을 출력:
   ```
   ✅ Flask 앱 실행 중 — http://localhost:5000 에서 확인
   종료를 원하면 "앱 종료해줘"라고 말하거나 터미널에서 Ctrl+C
   ```

6. 종료를 사용자가 요청하면 `KillBash`(또는 동등한 background kill 도구)로 해당 shell_id를 종료.
