# AI 뉴스 큐레이션 프로젝트

학습용 Flask + LangGraph + Discord 봇. 한국어 응답이 기본.

## 진입점

- `app.py` 단일 모듈에 retrieval / 분석 / LangGraph `news_agent` / Self-RAG wrapper `reflective_news_agent` / ReAct `react_agent` / Flask 라우트가 모두 들어 있음.
  - `python app.py` → `http://localhost:5000` (Flask)
  - `/` : `reflective_news_agent` — `news_agent` 한 번 실행 후 critic 점수 < 3이면 query rewrite + sources 확장으로 1회 retry
  - `/agent` : ReAct `react_agent` — LLM이 도구를 동적으로 결정 (별도 흐름)
- `discord_bot.py` : `/news`, `/headlines` 슬래시 커맨드. `/news`는 웹과 **동일한** `reflective_news_agent`를 호출(2026-05 이후 통일). `_format_news_result`로 디스코드용 마크다운 변환.
- `evaluate_retrieval.py` : retrieval precision을 LLM-judge로 평가. 결과는 `docs/llm-evaluation/retrieval-report.md`(로컬 전용, `docs/`는 .gitignore).
- `evaluate_agent.py` : ReAct 에이전트의 trace 평가(별도).

## 데이터 흐름 (한 검색 호출)

```
query
  │
  ▼
_rewrite_query (lru_cache, 자연어→키워드 + 의도 보존)
  │
  ├─ search_q  ─→  외부 API (Naver / GeekNews RSS / Naver 크롤)
  │                  │
  │                  ▼ (raw items)
  │
  └─ semantic_q ─→ rank_by_relevance (OpenAI 임베딩 → 코사인 → threshold 컷)
                                              │
                                              ▼
                                       top_k items
```

## Self-RAG 흐름 (reflective_news_agent)

```
reflective_news_agent(query, sources, do_summarize)
  │
  ▼
news_agent.invoke()         ── (1차) fetch → enrich → rank → brief
  │
  ▼
critique_results            ── critic LLM(gpt-4o-mini)이 1~5점 채점
  │
  ├─ score ≥ 3  ────→  (그대로 반환)
  │
  └─ score < 3  ────→  retry 조건 평가
                          ├─ suggest_query / 확장된 sources에 변화 없음 → 그대로 반환
                          └─ 변화 있음 → news_agent.invoke() 재실행
                                            │
                                            ▼ retry 결과가 빈약하면 원본 유지,
                                              아니면 brief에 "보강했습니다" 메모 추가
```

비용: 검색당 critic LLM 1회 추가. retry 발생 시 news_agent 전체 한 번 더(브리핑·분석 포함).
임계값(3)은 보수적 — 실제 retry는 자주 발생하지 않음.

소스별 임계값(`docs/llm-evaluation/retrieval-report.md` 근거):
- `RELEVANCE_THRESHOLD = 0.15` — naver_api, naver_crawl (이미 키워드 매칭된 결과)
- `RELEVANCE_THRESHOLD_RSS = 0.25` — geeknews (RSS 전체 풀에서 의미로만 선별)

## 자주 쓰는 명령

- `python app.py` — 웹 (debug는 `FLASK_DEBUG=1`로만 활성)
- `python discord_bot.py` — 봇 (`.env`의 `DISCORD_BOT_TOKEN` 필수)
- `python evaluate_retrieval.py` — retrieval 평가
- `python evaluate_importance.py` — importance(1~5) 채점 정확도 평가. 골든셋(`importance-golden.json`)과 LLM-judge(gpt-4o)로 `app.analyze` 결과 비교.
- `python -c "import app; import discord_bot; import evaluate_retrieval; import evaluate_importance; print('OK')"` — 변경 후 빠른 검증

또는 동등한 slash command: `/runweb`, `/runbot`, `/check-imports`, `/eval-retrieval` (`.claude/skills/` 참조).

## 외부 API / 비밀

`.env` (gitignored). 필요한 키:
- `OPENAI_API_KEY` — 필수. 없으면 분석/임베딩/브리핑 모두 skip되고 placeholder 응답.
- `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` — 네이버 검색 API 키. 없으면 `fetch_naver_api`가 자동으로 `fetch_naver_crawl`(웹 페이지 파싱)로 폴백.
- `DISCORD_BOT_TOKEN` — Discord 봇 실행 시만.
- `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true` — 옵션. 있으면 OpenAI 클라이언트가 wrap되어 trace 기록.

## 운영 메모

- **메인 브랜치 직접 push가 정책으로 차단**되어 있으므로 사용자 명시 허용이 필요.
- `.claude/`는 `.gitignore`에서 부분 공유: `.claude/skills/`와 `.claude/settings.json`만 추적, `settings.local.json` 등은 로컬 전용.
- 네이버 페이지 크롤(`fetch_naver_crawl`)은 ToS 회색지대 — 학습 용도로만, 셀렉터(`sds-comps-text-type-headline1`)는 네이버 개편 시 깨지므로 그때마다 `_parse_naver_sds`/`_parse_naver_legacy` 갱신.
- Discord 슬래시 커맨드는 등록 후 **최대 1시간 캐싱**. 봇 프로세스가 죽어 있어도 명령어 목록엔 남아 있고, 호출하면 "응답하지 않았어요" 에러가 뜸.

## 평가 정책

- retrieval 임계값을 자동으로 코드에 적용하지 말 것. 평균 precision이 약간 더 높아도 recall 손실이 클 수 있어 사용자 판단 필요.
- importance 채점 프롬프트(`app.analyze`의 system 메시지)와 골든셋(`importance-golden.json`)의 rubric은 **수동으로 동기화**할 것. 골든셋 정답에 맞춰 프롬프트를 자동 튜닝하면 과적합되어 실제 뉴스에서 분포가 무너짐.
- 골든셋은 사용자 판단 정답이므로 자동 라벨링 금지. 모델 출력으로 정답을 덮어쓰면 평가 의미가 사라짐.
- Discord 봇과 웹의 결과 차이는 2026-05 이후 동일 파이프라인으로 통일됨. 단, ReAct(`/agent`)는 별도 흐름이라 차이가 의도된 것.
