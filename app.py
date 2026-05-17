"""
AI 뉴스 크롤링 + OpenAI 요약 웹앱
실행: python app.py  →  http://localhost:5000
"""
import json
import operator
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Annotated, TypedDict

import feedparser
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, render_template, request
import markdown as md_lib
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langsmith import traceable
from markupsafe import Markup
from openai import OpenAI

load_dotenv()

app = Flask(__name__)


@app.template_filter("markdown")
def _render_markdown(text: str):
    return Markup(md_lib.markdown(text or "", extensions=["nl2br", "fenced_code", "tables"]))


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    _raw_client = OpenAI(api_key=OPENAI_API_KEY)
    if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        from langsmith.wrappers import wrap_openai
        client = wrap_openai(_raw_client)
    else:
        client = _raw_client
else:
    client = None

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

EMBED_MODEL = "text-embedding-3-small"
# 소스별 임계값 — retrieval-report.md 평가 결과 반영
# naver 계열은 이미 keyword 매칭된 결과라 임계값이 거의 무의미 (0.15가 평균 최적)
# geeknews는 검색 없이 전체 풀에서 의미로만 선별하므로 더 엄격한 컷(0.25) 필요
RELEVANCE_THRESHOLD = 0.15        # 기본값(naver 등 키워드 검색 기반 소스)
RELEVANCE_THRESHOLD_RSS = 0.25    # RSS/비검색 소스(geeknews)


def _cosine(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


@traceable(run_type="chain", name="rank_by_relevance",
           metadata={"model": EMBED_MODEL})
def rank_by_relevance(query: str, items: list[dict], top_k: int = 10,
                      threshold: float | None = None) -> list[dict]:
    """OpenAI 임베딩으로 query와 각 item의 의미적 유사도를 계산해 상위 top_k 반환.
    threshold 미만은 노이즈로 간주해 제외. 임베딩 실패 시 원본 순서로 fallback.
    threshold=None이면 모듈 레벨 상수(RELEVANCE_THRESHOLD)를 동적으로 참조."""
    if threshold is None:
        threshold = RELEVANCE_THRESHOLD
    q = (query or "").strip()
    if not q or q.upper() == "AI" or not items or client is None:
        return items[:top_k]
    texts = []
    for it in items:
        title = it.get("title", "")
        summary = (it.get("summary") or "")[:300]
        texts.append(f"{title}\n{summary}".strip())
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=[q] + texts)
        q_emb = resp.data[0].embedding
        item_embs = [d.embedding for d in resp.data[1:]]
        scored = [(it, _cosine(q_emb, emb)) for it, emb in zip(items, item_embs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [it for it, s in scored if s >= threshold]
        return ranked[:top_k]
    except Exception:
        return items[:top_k]


@traceable(run_type="retriever", name="fetch_geeknews")
def fetch_geeknews(query: str = "", limit: int = 10):
    """GeekNews RSS — 서버측 검색이 없으므로 50건을 받아 임베딩 의미 유사도로 재랭킹.
    query가 비어 있거나 'AI'면 최신 순 그대로."""
    feed = feedparser.parse("https://news.hada.io/rss/news")
    q = (query or "").strip()
    fetch_pool = 50 if q and q.upper() != "AI" else limit
    items = []
    for entry in feed.entries[:fetch_pool]:
        items.append({
            "title": entry.title,
            "link": entry.link,
            "summary": BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()[:500],
            "published": entry.get("published", ""),
            "source": "GeekNews",
        })
    return rank_by_relevance(q, items, top_k=limit, threshold=RELEVANCE_THRESHOLD_RSS)


@traceable(run_type="retriever", name="fetch_naver_api")
def fetch_naver_api(query="AI", limit=10):
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        # API 키가 없으면 크롤링으로 자동 폴백
        return fetch_naver_crawl(query, limit)
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    # 더 넓은 풀(30건) 받아 임베딩 재랭킹 → 상위 limit건만 반환
    fetch_pool = min(30, max(limit, 10))
    params = {"query": query, "display": fetch_pool, "sort": "date"}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    items = []
    for item in r.json().get("items", []):
        items.append({
            "title": BeautifulSoup(item["title"], "html.parser").get_text(),
            "link": item.get("originallink") or item["link"],
            "summary": BeautifulSoup(item["description"], "html.parser").get_text(),
            "published": item.get("pubDate", ""),
            "source": "네이버 검색 API",
        })
    return rank_by_relevance(query, items, top_k=limit)


@traceable(run_type="retriever", name="fetch_naver_crawl")
def fetch_naver_crawl(query="AI", limit=10):
    """네이버 뉴스 검색 결과 페이지 파싱 (ToS 주의 - 개인 학습 용도로만).
    최근 2년치만 가져오도록 날짜 범위(nso)를 URL에 추가."""
    today = date.today()
    two_years_ago = today - timedelta(days=365 * 2)
    ds = two_years_ago.strftime("%Y.%m.%d")
    de = today.strftime("%Y.%m.%d")
    nso = f"so:dd,p:from{two_years_ago.strftime('%Y%m%d')}to{today.strftime('%Y%m%d')}"
    q_encoded = quote(query or "", safe="")
    url = (
        "https://search.naver.com/search.naver?where=news"
        f"&query={q_encoded}&sort=1&pd=3&ds={ds}&de={de}&nso={quote(nso, safe=':,')}"
    )
    try:
        r = requests.get(url, headers=UA, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        items = []

        # 새 구조 (2025~): sds-comps 디자인 시스템 기반
        # 각 기사 헤드라인은 span.sds-comps-text-type-headline1, 부모 <a>가 링크
        # 기사별 컨테이너: 헤드라인 anchor에서 위로 올라가 headline span이 정확히 1개인 가장 작은 div
        headlines = soup.select("span.sds-comps-text-type-headline1")
        seen_links = set()
        for span in headlines:
            a = span.find_parent("a", href=True)
            if not a:
                continue
            href = a.get("href", "")
            if not href or href in seen_links:
                continue
            seen_links.add(href)
            title = span.get_text(strip=True)

            # 기사별 컨테이너 찾기 (헤드라인 1개만 포함하는 최소 ancestor)
            container = a.parent
            for _ in range(6):
                if container is None:
                    break
                if len(container.select("span.sds-comps-text-type-headline1")) == 1:
                    break
                container = container.parent

            summary = ""
            published = ""
            if container is not None:
                # 본문 요약은 body1, 시간/메타는 body2
                body1 = container.select_one("span.sds-comps-text-type-body1")
                if body1:
                    txt = body1.get_text(strip=True)
                    if txt and txt != title:
                        summary = txt
                for b2 in container.select("span.sds-comps-text-type-body2"):
                    txt = b2.get_text(strip=True)
                    if txt and ("전" in txt or txt.endswith(".")):
                        published = txt
                        break

            items.append({
                "title": title,
                "link": href,
                "summary": summary,
                "published": published,
                "source": "네이버 뉴스 크롤링",
            })
            if len(items) >= limit:
                break

        # 구버전 셀렉터 폴백
        if not items:
            nodes = soup.select("ul.list_news li.bx") or soup.select("div.group_news li")
            for li in nodes[:limit]:
                a = li.select_one("a.news_tit") or li.select_one("a.tit")
                desc = li.select_one("div.dsc_wrap") or li.select_one("a.api_txt_lines")
                if not a:
                    continue
                items.append({
                    "title": a.get("title") or a.get_text(strip=True),
                    "link": a.get("href", ""),
                    "summary": desc.get_text(strip=True) if desc else "",
                    "published": "",
                    "source": "네이버 뉴스 크롤링",
                })

        if not items:
            items.append({"title": "(크롤링 결과 없음 - 네이버 페이지 구조 변경 가능성)",
                          "link": "#", "summary": "", "published": "",
                          "source": "네이버 뉴스 크롤링", "ai_summary": ""})
        return items
    except Exception as e:
        return [{"title": f"(크롤링 실패: {e})", "link": "#", "summary": "",
                 "published": "", "source": "네이버 뉴스 크롤링", "ai_summary": ""}]


EMPTY_ANALYSIS = {"summary": "", "importance": 0, "evaluation": ""}


@traceable(
    run_type="chain",
    name="analyze_article",
    metadata={"model": "gpt-4o-mini"},
)
def analyze(item):
    """OpenAI로 요약 + 중요도(1~5) + 한 줄 평가를 한 번에 받음."""
    text = item.get("summary", "")
    title = item.get("title", "")
    if not text or item.get("link") == "#":
        return dict(EMPTY_ANALYSIS)
    if client is None:
        return {"summary": "(OPENAI_API_KEY 미설정 - .env에 키를 추가하세요)",
                "importance": 0, "evaluation": ""}
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": (
                     "당신은 한국어 뉴스 분석 전문가입니다. 주어진 뉴스를 분석해 JSON으로 응답하세요.\n"
                     "필드:\n"
                     '- "summary": 핵심을 2~3문장으로 요약 (추측/부연 금지)\n'
                     '- "importance": 1~5 정수 점수. 1=가십·단순 동향, 3=업계 관심사, 5=산업/사회에 큰 영향\n'
                     '- "evaluation": 그 점수를 매긴 이유를 한 줄(40자 이내)로 설명'
                 )},
                {"role": "user", "content": f"제목: {title}\n\n내용: {text}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=400,
            temperature=0.3,
        )
        data = json.loads(resp.choices[0].message.content)
        try:
            importance = max(0, min(5, int(data.get("importance", 0))))
        except (TypeError, ValueError):
            importance = 0
        return {
            "summary": (data.get("summary") or "").strip(),
            "importance": importance,
            "evaluation": (data.get("evaluation") or "").strip(),
        }
    except Exception as e:
        return {"summary": f"(분석 실패: {e})", "importance": 0, "evaluation": ""}


@traceable(name="brief", run_type="llm")
def make_brief(top_picks):
    if not top_picks or client is None:
        return ""
    bullets = "\n".join(
        f"- ({it['importance']}/5) {it['title']}: {it.get('ai_summary', '')}"
        for it in top_picks
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": (
                     "당신은 한국어 뉴스 큐레이터입니다. 주어진 상위 뉴스들을 종합해 "
                     "오늘의 핵심 흐름을 2~3문장 브리핑으로 작성하세요. "
                     "추측은 금지하고 주어진 정보만 활용하세요."
                 )},
                {"role": "user", "content": bullets},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(브리핑 실패: {e})"


# ---------------------------------------------------------------------------
# LangGraph 에이전트
# ---------------------------------------------------------------------------
class NewsState(TypedDict):
    query: str
    sources: list[str]
    do_summarize: bool
    items: Annotated[list[dict], operator.add]
    analyzed: list[dict]
    top_picks: list[dict]
    brief: str


def route_sources(state: NewsState):
    targets = []
    if "geeknews" in state["sources"]:
        targets.append("fetch_geeknews_node")
    if "naver_api" in state["sources"]:
        targets.append("fetch_naver_api_node")
    if "naver_crawl" in state["sources"]:
        targets.append("fetch_naver_crawl_node")
    return targets or ["enrich_node"]


def fetch_geeknews_node(state: NewsState):
    return {"items": fetch_geeknews(state.get("query", ""))}


def fetch_naver_api_node(state: NewsState):
    return {"items": fetch_naver_api(state["query"])}


def fetch_naver_crawl_node(state: NewsState):
    return {"items": fetch_naver_crawl(state["query"])}


def enrich_node(state: NewsState):
    items = state["items"]
    if not items:
        return {"analyzed": []}
    if not state.get("do_summarize"):
        analyzed = []
        for it in items:
            merged = dict(it)
            merged.setdefault("ai_summary", "")
            merged.setdefault("importance", 0)
            merged.setdefault("evaluation", "")
            analyzed.append(merged)
        return {"analyzed": analyzed}
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(analyze, items))
    analyzed = []
    for it, r in zip(items, results):
        merged = dict(it)
        if "ai_summary" not in merged:
            merged["ai_summary"] = r["summary"]
            merged["importance"] = r["importance"]
            merged["evaluation"] = r["evaluation"]
        merged.setdefault("ai_summary", "")
        merged.setdefault("importance", 0)
        merged.setdefault("evaluation", "")
        analyzed.append(merged)
    return {"analyzed": analyzed}


def rank_node(state: NewsState):
    real = [it for it in state["analyzed"] if it.get("link") and it.get("link") != "#"]
    top = sorted(real, key=lambda x: x.get("importance", 0), reverse=True)[:3]
    top = [t for t in top if t.get("importance", 0) > 0]
    return {"top_picks": top}


def brief_node(state: NewsState):
    if not state.get("do_summarize"):
        return {"brief": ""}
    return {"brief": make_brief(state["top_picks"])}


def _build_graph():
    g = StateGraph(NewsState)
    g.add_node("fetch_geeknews_node", fetch_geeknews_node)
    g.add_node("fetch_naver_api_node", fetch_naver_api_node)
    g.add_node("fetch_naver_crawl_node", fetch_naver_crawl_node)
    g.add_node("enrich_node", enrich_node)
    g.add_node("rank_node", rank_node)
    g.add_node("brief_node", brief_node)
    g.add_conditional_edges(
        START,
        route_sources,
        ["fetch_geeknews_node", "fetch_naver_api_node",
         "fetch_naver_crawl_node", "enrich_node"],
    )
    g.add_edge("fetch_geeknews_node", "enrich_node")
    g.add_edge("fetch_naver_api_node", "enrich_node")
    g.add_edge("fetch_naver_crawl_node", "enrich_node")
    g.add_edge("enrich_node", "rank_node")
    g.add_edge("rank_node", "brief_node")
    g.add_edge("brief_node", END)
    return g.compile()


news_agent = _build_graph()


# ---------------------------------------------------------------------------
# ReAct 에이전트 (LLM이 도구 호출을 스스로 결정)
# ---------------------------------------------------------------------------
@tool
def search_news(source: str = "geeknews", query: str = "AI") -> str:
    """뉴스 검색 도구.
    source 옵션:
      - 'geeknews'    : GeekNews RSS (query로 클라이언트 필터링)
      - 'naver'       : 네이버 검색 API (키 없으면 자동으로 크롤링 폴백)
      - 'naver_crawl' : 네이버 뉴스 검색 페이지 크롤링 (최근 2년 한정)
    검색된 기사들의 번호/제목/요약/출처를 텍스트로 반환."""
    if source == "naver":
        items = fetch_naver_api(query)
    elif source == "naver_crawl":
        items = fetch_naver_crawl(query)
    else:
        items = fetch_geeknews(query)
    if not items:
        return "(검색 결과 없음)"
    lines = []
    for i, it in enumerate(items[:10], 1):
        lines.append(
            f"[{i}] {it.get('title', '')}\n"
            f"    요약: {(it.get('summary') or '')[:200]}\n"
            f"    링크: {it.get('link', '')}"
        )
    return "\n".join(lines)


@tool
def rate_article(title: str, content: str) -> str:
    """단일 기사의 중요도(1~5)와 한 줄 평가를 OpenAI로 산출. content는 200자 이내 핵심만."""
    r = analyze({"title": title, "summary": content, "link": "ok"})
    return (f"중요도: {r['importance']}/5\n"
            f"평가: {r['evaluation']}\n"
            f"요약: {r['summary']}")


@tool
def compose_brief(headlines_with_scores: str) -> str:
    """여러 기사의 핵심을 종합 브리핑으로 작성. 입력은 줄바꿈으로 구분된 '제목 - 요약' 형식."""
    lines = [ln.strip() for ln in headlines_with_scores.split("\n") if ln.strip()]
    items = [{"title": ln, "importance": 3, "ai_summary": ""} for ln in lines]
    return make_brief(items)


_REACT_PROMPT = (
    "당신은 한국어 뉴스 큐레이션 에이전트입니다.\n\n"
    "## 도구\n"
    "- search_news(source, query): source는 'geeknews' | 'naver' | 'naver_crawl'.\n"
    "    * **키워드 검색(예: 'AI 반도체', 'GPT', '메타버스')**: 'geeknews'와 'naver' "
    "**두 소스를 모두 호출**해 IT 해커뉴스와 한국 일반 뉴스를 함께 수집. "
    "'naver' 결과가 부족하면 'naver_crawl'로 보강.\n"
    "    * 사용자가 'GeekNews만'·'네이버만'처럼 **소스를 단일하게 지정**한 경우에만 그 하나만 호출.\n"
    "    * 사용자가 키워드 없이 '오늘 뉴스', 'IT 헤드라인' 같이 요청하면 'geeknews'만으로 충분.\n"
    "- rate_article(title, content): 단일 기사 중요도(1~5) + 한 줄 평가\n"
    "- compose_brief(headlines_with_scores): 여러 기사 종합 브리핑 — "
    "사용자가 '브리핑/종합/요약해서 알려줘'를 요청하면 **반드시 호출**.\n\n"
    "## 사용자 의도 분류 (먼저 판단할 것)\n"
    "**Mode A — 헤드라인 전용**\n"
    "  · '헤드라인만', '빨리 보여줘', '평가/브리핑 필요 없어' 같은 표현이 있을 때.\n"
    "  · 호출 도구: `search_news` **만**. (rate_article, compose_brief 절대 호출 금지)\n"
    "**Mode B — 평가 (단건 또는 N건)**\n"
    "  · '중요도/평가해줘', 'N건 골라' (단, '브리핑'은 명시 안 됨).\n"
    "  · 호출 도구: `search_news` → `rate_article` × N (정확히 N번). compose_brief 호출 금지.\n"
    "**Mode C — 종합 브리핑**\n"
    "  · '브리핑', '종합', '요약해줘' 같은 표현이 있을 때.\n"
    "  · 호출 도구: `search_news` → `rate_article` × N → `compose_brief` **필수**.\n"
    "  · compose_brief 입력은 줄바꿈으로 구분된 '제목 - 한줄요약' 형식.\n\n"
    "## 엄격한 규칙\n"
    "1. N건 요청 시 rate_article은 정확히 N번. 비교용 추가 호출 금지.\n"
    "2. 검색 결과에 없는 내용은 추측 금지. 최종 답변은 항상 한국어.\n"
    "3. 도구 호출 내역이나 내부 동작 설명을 답변에 포함하지 말 것.\n\n"
    "## 최종 응답 형식 (의도별 분기)\n\n"
    "### Mode A — 헤드라인 전용\n"
    "도입부 인사말이나 종합 설명 **절대 금지**. 코드블록(```) 사용 금지.\n"
    "번호 매긴 리스트만 출력하되, **제목 자체를 링크**로 만들 것. 예:\n"
    "  `1. [Git은 괜찮지 않다](https://news.hada.io/topic?id=29561)`\n"
    "  `2. [Steve Jobs의 망명기](https://news.hada.io/topic?id=29559)`\n\n"
    "### Mode B — 평가\n"
    "**반드시 두 부분을 모두 포함할 것. 어느 한 쪽도 생략 금지.**\n"
    "(1) **한 문단 종합 평가** (3~5문장, 절대 생략 금지) — 결과의 흐름을 친근한 톤으로 설명.\n"
    "    시작 예: '오늘 GeekNews에서는 …가 두드러집니다. 특히 …'\n"
    "    1건만 평가하는 경우에도 그 한 기사의 맥락·의미를 3문장 이상으로 풀어 쓸 것.\n"
    "(2) 빈 줄 한 칸 후, **마크다운 목록**. 각 줄은 다음 형식을 그대로 따를 것:\n"
    "    `- [**<기사 제목>**](<실제 URL>) · 중요도 N/5 — <한 줄 평가>`\n"
    "    실제 예: `- [**Claude for Legal**](https://news.hada.io/topic?id=29557) · 중요도 4/5 — 법률 업무에 큰 변화 예상`\n\n"
    "### Mode C — 종합 브리핑\n"
    "**반드시 두 부분을 모두 포함할 것.**\n"
    "(1) **종합 브리핑 단락** — compose_brief 도구의 출력 텍스트를 한 단락으로 그대로 제시.\n"
    "(2) 빈 줄 한 칸 후, Mode B와 **완전히 동일한** 마크다운 목록 형식으로 각 기사 정리."
)

if OPENAI_API_KEY:
    _react_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    react_agent = create_react_agent(
        _react_llm,
        tools=[search_news, rate_article, compose_brief],
        prompt=_REACT_PROMPT,
    )
else:
    react_agent = None


@app.route("/")
def index():
    submitted = request.args.get("submitted") == "1"
    sources = request.args.getlist("source")
    if not submitted and not sources:
        sources = ["geeknews", "naver_api"]
    query = request.args.get("q", "AI")
    do_summarize = request.args.get("summarize") == "1" if submitted else True

    result = news_agent.invoke({
        "query": query,
        "sources": sources,
        "do_summarize": do_summarize,
        "items": [],
        "analyzed": [],
        "top_picks": [],
        "brief": "",
    })

    items = list(result.get("analyzed", []))
    top_picks = result.get("top_picks", [])
    top_links = {t.get("link") for t in top_picks}
    for it in items:
        it["is_top"] = it.get("link") in top_links and it.get("link") != "#"
    items.sort(key=lambda x: (not x.get("is_top", False), -x.get("importance", 0)))

    return render_template(
        "index.html",
        items=items, sources=sources, query=query,
        do_summarize=do_summarize,
        brief=result.get("brief", ""),
        top_picks=top_picks,
    )


def _split_messages(messages):
    """LangGraph agent 메시지를 (final_answer, trace) 로 분리."""
    trace = []
    final_answer = ""
    for m in messages:
        role = getattr(m, "type", "")
        if role == "human":
            continue
        elif role == "ai":
            for tc in getattr(m, "tool_calls", None) or []:
                trace.append({
                    "kind": "tool_call",
                    "tool_name": tc.get("name", ""),
                    "tool_args": json.dumps(tc.get("args", {}),
                                            ensure_ascii=False, indent=2),
                })
            if m.content:
                if final_answer:
                    trace.append({"kind": "assistant_mid", "content": final_answer})
                final_answer = m.content
        elif role == "tool":
            content = m.content if isinstance(m.content, str) else str(m.content)
            if len(content) > 1500:
                content = content[:1500] + "\n...(생략)"
            trace.append({
                "kind": "tool_result",
                "tool_name": getattr(m, "name", ""),
                "content": content,
            })
    return final_answer, trace


@app.route("/agent", methods=["GET"])
def agent_route():
    user_query = request.args.get("q", "")
    if not user_query:
        return render_template("agent.html", final_answer="", trace=[],
                               user_query="", available=react_agent is not None)
    if react_agent is None:
        return render_template(
            "agent.html",
            final_answer="OPENAI_API_KEY 미설정 - .env에 키를 추가하세요.",
            trace=[], user_query=user_query, available=False,
        )
    try:
        result = react_agent.invoke(
            {"messages": [("user", user_query)]},
            config={"recursion_limit": 25},
        )
        final_answer, trace = _split_messages(result["messages"])
    except Exception as e:
        final_answer = f"(에이전트 실행 실패: {e})"
        trace = []
    return render_template("agent.html", final_answer=final_answer, trace=trace,
                           user_query=user_query, available=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
