# AI 뉴스 큐레이션

학습용 RAG 프로젝트 — **Flask 웹** + **Discord 봇** + **LangGraph 에이전트**.
멀티 소스에서 뉴스를 끌어와 OpenAI 임베딩으로 재랭킹하고, gpt-4o-mini로 분석·브리핑한 뒤,
critic LLM이 결과 품질을 채점해 부족하면 한 번 더 검색하는 **Self-RAG** 패턴을 구현합니다.

```
검색어
  │
  ├─ _rewrite_query  ── (자연어→키워드 + 의미용 원문 분리)
  │
  ├─ 멀티 소스 fetch (병렬, LangGraph)
  │     GeekNews / 네이버 API / 네이버 크롤 / TechCrunch AI / VentureBeat AI
  │
  ├─ rank_by_relevance  ── OpenAI 임베딩 + 코사인 유사도 + threshold
  │     ↑ embed_batch가 SQLite로 임베딩 캐싱
  │
  ├─ enrich (gpt-4o-mini, 기사별 importance 1~5)
  │
  ├─ rank → top 3
  │
  ├─ make_brief (gpt-4o-mini, 한국어 브리핑)
  │
  └─ critique  ── 1~5점 채점, < 3이면 query rewrite + sources 확장 후 1회 retry
```

## 빠른 시작

```bash
pip install -r requirements.txt
cp .env.example .env  # 실제 키 채우기

python app.py          # 웹: http://localhost:5000
python discord_bot.py  # Discord 봇 (선택)
```

`.env` 키:
| 변수 | 필수 | 용도 |
|------|------|------|
| `OPENAI_API_KEY` | 예 | 분석·임베딩·브리핑. 없으면 placeholder 응답 |
| `NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET` | 아니오 | 네이버 검색 API. 없으면 자동으로 웹 크롤로 폴백 |
| `DISCORD_BOT_TOKEN` | 봇 실행 시 | Discord 봇 |
| `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true` | 아니오 | LangSmith trace 기록 |

## 진입점

| 모듈 | 설명 |
|------|------|
| `app.py` | 단일 모듈. fetchers / `rank_by_relevance` / LangGraph `news_agent` / Self-RAG `reflective_news_agent` / ReAct `react_agent` / Flask 라우트 |
| `discord_bot.py` | `/news` (웹과 동일 파이프라인) · `/headlines` (GeekNews 헤드라인) |
| `evaluate_retrieval.py` | 소스 × threshold별 retrieval precision을 LLM-judge로 측정 |
| `evaluate_agent.py` | ReAct trace 평가 |

웹 UI:
- `/` — Self-RAG 파이프라인
- `/agent` — ReAct 에이전트 (LLM이 도구 호출을 스스로 결정)

## 소스 목록

| 소스 | retriever | 비고 |
|------|-----------|------|
| GeekNews | `fetch_geeknews` | RSS, 한국 IT/해커뉴스 큐레이션 |
| 네이버 검색 API | `fetch_naver_api` | 키 없으면 크롤 폴백 |
| 네이버 크롤 | `fetch_naver_crawl` | 검색 페이지 HTML 파싱 (학습 용도) |
| TechCrunch AI | `fetch_techcrunch_ai` | RSS, 영문 AI/스타트업 |
| VentureBeat AI | `fetch_venturebeat_ai` | RSS, 영문 AI 비즈니스·산업 분석 |

소스별 threshold (`docs/llm-evaluation/retrieval-report.md` 근거):
- `RELEVANCE_THRESHOLD = 0.15` — naver 계열 (이미 키워드 매칭된 결과)
- `RELEVANCE_THRESHOLD_RSS = 0.25` — RSS 계열 (전체 풀에서 의미로만 선별)

## 임베딩 캐시

`rank_by_relevance`는 `embed_batch` → SQLite(`.cache/embeddings.sqlite`)를 거칩니다.
같은 텍스트(쿼리든 기사든)가 다시 나오면 OpenAI 호출을 건너뜁니다.
스키마: `(text_hash, model)` PK + float32 BLOB.

경로는 `EMBED_CACHE_PATH` 환경 변수로 변경 가능. 캐시를 비우려면 파일 삭제하면 됨.

## 자주 쓰는 명령

```bash
python app.py                      # 웹 (FLASK_DEBUG=1로 debug 모드)
python discord_bot.py              # 봇
python evaluate_retrieval.py       # retrieval 평가 → docs/llm-evaluation/retrieval-report.md
python -c "import app; import discord_bot; import evaluate_retrieval; print('OK')"  # 빠른 검증
```

Claude Code 환경이면 동등한 slash command: `/runweb`, `/runbot`, `/check-imports`, `/eval-retrieval` (`.claude/skills/` 참조).

## 디렉토리

```
.
├── app.py                  # 메인 모듈 (Flask + LangGraph + Self-RAG + ReAct)
├── discord_bot.py          # Discord 슬래시 커맨드
├── evaluate_retrieval.py   # retrieval 평가
├── evaluate_agent.py       # ReAct trace 평가
├── templates/              # Jinja 템플릿
│   ├── index.html
│   └── agent.html
├── .claude/skills/         # Claude Code 스킬 (공유)
├── .cache/                 # 임베딩 캐시 (gitignored)
└── docs/                   # 평가 노트 (gitignored)
```

## 메모

- 메인 브랜치 직접 push는 정책상 단일 사용자 학습용 — 운영 프로젝트라면 PR 흐름 권장.
- 네이버 페이지 크롤은 ToS 회색지대 — 학습 용도로만. 네이버 개편 시 셀렉터(`sds-comps-text-type-headline1`)는 그때그때 갱신.
- Discord 슬래시 커맨드는 등록 후 **최대 1시간 캐싱**. 봇이 죽어 있어도 명령어 목록엔 남고, 호출하면 "응답하지 않았어요" 에러.
- 자세한 동작·정책은 [`CLAUDE.md`](CLAUDE.md) 참조.
