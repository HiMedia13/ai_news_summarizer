---
name: runbot
description: Discord 봇(discord_bot.py)을 백그라운드 실행하고 "로그인 완료" 메시지가 뜰 때까지 대기. /news /headlines 슬래시 커맨드를 디스코드에서 호출할 수 있게 함.
---

# /runbot — Discord 봇 실행

## 절차

1. **토큰 점검.** `DISCORD_BOT_TOKEN`이 `.env`에 있는지 확인:
   ```powershell
   if (-not (Select-String -Path .env -Pattern "^DISCORD_BOT_TOKEN=" -Quiet)) {
     Write-Output "DISCORD_BOT_TOKEN 미설정 — discord_bot.py가 즉시 종료될 수 있음"
   }
   ```
   없으면 사용자에게 발급 URL(`https://discord.com/developers/applications`) 안내 후 중단.

2. **OPENAI_API_KEY 확인.** 없으면 `/news` 명령이 즉시 "OPENAI_API_KEY 미설정" 응답만 함을 미리 알린다.

3. **백그라운드 실행.** Bash 도구의 `run_in_background: true`로:
   ```powershell
   python discord_bot.py
   ```
   반환되는 shell_id를 기억해 둘 것.

4. **로그인 대기.** Monitor 도구(deferred — 필요 시 `ToolSearch query: "select:Monitor"`로 로드)로 다음 라인이 나올 때까지 대기:
   ```
   [discord] <봇이름>#0000 로그인 완료 · 등록된 슬래시 커맨드: ['news', 'headlines']
   ```
   timeout 30초. Monitor 사용이 어려우면 약간 기다린 뒤 `BashOutput`으로 로그를 확인. 안 뜨면 stderr/stdout 전부 보여주고 원인 진단(토큰 무효, 네트워크 등).

5. **안내.** 로그인 완료 시:
   ```
   ✅ 봇 온라인 — Discord에서 /news <검색어> 또는 /headlines 호출
   슬래시 커맨드 캐싱은 최대 1시간이라 처음에는 안 보일 수 있음.
   ```

6. 종료 요청 시 `KillBash`(또는 동등한 background kill 도구)로 shell_id 종료. 봇 프로세스가 죽으면 Discord에서 "애플리케이션이 응답하지 않았어요" 에러가 발생함을 사용자에게 상기.
