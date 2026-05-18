---
name: eval-retrieval
description: evaluate_retrieval.py를 실행해 소스 × threshold별 retrieval precision을 LLM-judge로 측정. docs/llm-evaluation/retrieval-report.md를 갱신. 임계값 튜닝이 필요할 때 사용.
---

# /eval-retrieval — Retrieval 품질 평가

## 절차

1. **사전 점검.** OPENAI_API_KEY가 없으면 스크립트가 즉시 종료. `.env` 확인:
   ```powershell
   if (-not (Select-String -Path .env -Pattern "^OPENAI_API_KEY=" -Quiet)) {
     Write-Output "OPENAI_API_KEY 미설정 — 평가 불가"; exit 1
   }
   ```

2. **사용자 확인.** 평가는 OPENAI API와 LangSmith를 호출하므로 비용 발생을 안내:
   - 쿼리 6개 × 소스 3개 × threshold 4개 = 72회 retrieval + 그 결과를 judge LLM이 채점
   - 대략 gpt-4o-mini 수백 호출 (몇 백 원 수준)
   사용자가 진행 동의하면 계속.

3. **실행.** Bash 도구로 (백그라운드 권장 — 1~2분 소요):
   ```powershell
   python evaluate_retrieval.py
   ```

4. **완료 후 리포트 확인.** `docs/llm-evaluation/retrieval-report.md`를 Read로 열어 사용자에게 핵심만 요약:
   - 최적 (source, threshold) 조합
   - 현재 `app.py`의 `RELEVANCE_THRESHOLD` / `RELEVANCE_THRESHOLD_RSS`와 비교
   - 변경 권장 여부 (변경 가치가 작으면 "지금 값 유지" 라고 알릴 것)

5. **자동 적용 금지.** 평가 결과에 따른 임계값 변경은 사용자 확인 후에만. 단순히 평균 precision이 약간 더 높다고 자동으로 코드 수정하지 말 것 — recall/노이즈 균형은 도메인 판단.
