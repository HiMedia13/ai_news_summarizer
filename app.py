"""
AI 뉴스 크롤링 + OpenAI 요약 웹앱
실행: python app.py  →  http://localhost:5000
"""
import array
import hashlib
import json
import logging
import math
import operator
import os
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
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

# `python app.py` 외에 `from app import app`로 띄우는 경우(테스트·gunicorn 등)에도
# INFO 레벨 로그가 콘솔에 보이도록 모듈 import 시 한 번만 설정. root에 이미 핸들러가
# 있으면 (예: pytest의 caplog) 손대지 않는다.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

log = logging.getLogger("ai_news_summarizer")

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
# 임계값 근거는 docs/llm-evaluation/retrieval-report.md 참조.
RELEVANCE_THRESHOLD = 0.15        # 기본값(naver 등 키워드 검색 기반 소스)
RELEVANCE_THRESHOLD_RSS = 0.25    # RSS/비검색 소스(geeknews)

# ---------------------------------------------------------------------------
# Embedding cache (SQLite) — (text_hash, model) → embedding BLOB.
# 같은 기사/쿼리가 다시 나오면 OpenAI 호출을 건너뛴다.
# ---------------------------------------------------------------------------
EMBED_CACHE_PATH = Path(os.getenv("EMBED_CACHE_PATH", ".cache/embeddings.sqlite"))
_embed_db_lock = threading.Lock()
_embed_db_ready = False


def _embed_db_conn() -> sqlite3.Connection:
    """Lazy init: 디렉토리·테이블·WAL을 한 번만 셋업하고 연결을 돌려준다."""
    global _embed_db_ready
    if not _embed_db_ready:
        with _embed_db_lock:
            if not _embed_db_ready:
                EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(EMBED_CACHE_PATH, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS embedding_cache (
                        text_hash TEXT NOT NULL,
                        model     TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        created_at REAL NOT NULL DEFAULT (unixepoch()),
                        PRIMARY KEY (text_hash, model)
                    )
                """)
                conn.commit()
                conn.close()
                _embed_db_ready = True
    return sqlite3.connect(EMBED_CACHE_PATH, check_same_thread=False)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _emb_to_blob(emb) -> bytes:
    return array.array("f", emb).tobytes()


def _blob_to_emb(blob: bytes) -> list[float]:
    return array.array("f", blob).tolist()


def _load_cached_embeddings(hashes: list[str], model: str) -> dict[str, list[float]]:
    if not hashes:
        return {}
    conn = _embed_db_conn()
    try:
        placeholders = ",".join("?" * len(hashes))
        rows = conn.execute(
            f"SELECT text_hash, embedding FROM embedding_cache "
            f"WHERE model = ? AND text_hash IN ({placeholders})",
            [model, *hashes],
        ).fetchall()
        return {h: _blob_to_emb(b) for h, b in rows}
    finally:
        conn.close()


def _store_cached_embeddings(items: list[tuple[str, list[float]]], model: str) -> None:
    if not items:
        return
    conn = _embed_db_conn()
    try:
        with _embed_db_lock:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (text_hash, model, embedding) "
                "VALUES (?, ?, ?)",
                [(h, model, _emb_to_blob(e)) for h, e in items],
            )
            conn.commit()
    finally:
        conn.close()


@traceable(run_type="embedding", name="embed_batch",
           metadata={"model": EMBED_MODEL})
def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """텍스트 리스트 → 임베딩 리스트. 캐시에 있는 건 재사용, 없는 것만 OpenAI 호출."""
    if not texts or client is None:
        return []
    hashes = [_hash_text(t) for t in texts]
    cached = _load_cached_embeddings(list(set(hashes)), model)
    missing_idx = [i for i, h in enumerate(hashes) if h not in cached]
    if missing_idx:
        missing_texts = [texts[i] for i in missing_idx]
        resp = client.embeddings.create(model=model, input=missing_texts)
        new_embs = [d.embedding for d in resp.data]
        to_store = []
        for idx, emb in zip(missing_idx, new_embs):
            cached[hashes[idx]] = emb
            to_store.append((hashes[idx], emb))
        _store_cached_embeddings(to_store, model)
    return [cached[h] for h in hashes]

# 웹 기본 소스 — discord_bot.py도 이 상수를 import해 사용
DEFAULT_SOURCES = ["geeknews", "naver_api"]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


_NATURAL_HINT_RE = re.compile(
    r"(어떤|어떻|어때|어디|언제|어느|왜|뭐|무엇|있나|있어|있을까|"
    r"되나|되어|될까|해줘|알려|보여|평가|분석|설명|요약|정리)"
)


def _looks_natural(query: str) -> bool:
    """질문이 자연어 문장인지 휴리스틱으로 판단.
    - '?' 포함, 의문어/요청동사 포함, 또는 15자 초과일 때 True."""
    q = (query or "").strip()
    if not q:
        return False
    if "?" in q or "？" in q:
        return True
    if _NATURAL_HINT_RE.search(q):
        return True
    return len(q) > 15


@lru_cache(maxsize=128)
@traceable(run_type="chain", name="rewrite_query_for_search")
def _rewrite_query(query: str) -> tuple[str, str]:
    """자연어 질문이면 LLM으로 검색 키워드를 추출.

    Returns: (search_query, semantic_query)
      - search_query: 외부 검색 API/필터에 던질 키워드형 문자열
      - semantic_query: 의미 재랭킹용 — 자연어 의도 그대로 보존
    짧은 키워드 input은 LLM 호출을 생략해 비용/지연을 절약.
    lru_cache로 메모이즈 — LangGraph가 같은 query에 대해 fetch_*를 병렬로 호출해도
    LLM 호출은 한 번만 발생."""
    q = (query or "").strip()
    if not q or not _looks_natural(q) or client is None:
        return q, q
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": (
                     "당신은 한국어 검색 키워드 추출기입니다. 사용자의 자연어 질문에서 "
                     "뉴스 검색 API에 던질 핵심 명사·고유명사만 공백으로 구분해 출력하세요. "
                     "조사·의문어·동사 어미는 모두 제거. 다른 텍스트 없이 키워드만.\n"
                     "예시:\n"
                     "  Q: 최근 AI 반도체 시장에서 엔비디아 위치는 어때?\n"
                     "  A: AI 반도체 엔비디아\n"
                     "  Q: Claude의 코딩 능력은 어떻게 발전했나?\n"
                     "  A: Claude 코딩 능력"
                 )},
                {"role": "user", "content": q},
            ],
            max_tokens=40,
            temperature=0,
        )
        kw = (resp.choices[0].message.content or "").strip().strip('"\'')
        return (kw if kw else q), q
    except Exception:
        return q, q


def _resolve_queries(query: str, intent: str) -> tuple[str, str]:
    """fetch_* 함수들이 공통으로 쓰는 (search_query, semantic_query) 결정 로직.
    intent가 명시되면 검색은 query, 재랭킹은 intent. 없으면 _rewrite_query로 자동 분리."""
    if intent:
        return (query or "").strip(), intent.strip()
    return _rewrite_query(query)


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
    if not q or q.upper() == "AI" or not items or client is None or len(items) <= 1:
        return items[:top_k]
    texts = []
    for it in items:
        title = it.get("title", "")
        summary = (it.get("summary") or "")[:300]
        texts.append(f"{title}\n{summary}".strip())
    try:
        embs = embed_batch([q] + texts)
        if not embs:
            return items[:top_k]
        q_emb, item_embs = embs[0], embs[1:]
        scored = [(it, _cosine(q_emb, emb)) for it, emb in zip(items, item_embs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [it for it, s in scored if s >= threshold]
        return ranked[:top_k]
    except Exception:
        return items[:top_k]


def _fetch_rss(rss_url: str, source_label: str,
               query: str, limit: int,
               *, intent: str = "",
               threshold: float | None = None) -> list[dict]:
    """RSS 피드용 공용 retriever. 서버측 검색이 없으므로 최신 50건을 받아
    임베딩 의미 유사도로 재랭킹. query가 비어 있거나 'AI'면 최신 순 그대로."""
    search_q, semantic_q = _resolve_queries(query, intent)
    feed = feedparser.parse(rss_url)
    fetch_pool = 50 if search_q and search_q.upper() != "AI" else limit
    items = []
    for entry in feed.entries[:fetch_pool]:
        items.append({
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "summary": BeautifulSoup(
                entry.get("summary", ""), "html.parser",
            ).get_text()[:500],
            "published": entry.get("published", ""),
            "source": source_label,
        })
    t = threshold if threshold is not None else RELEVANCE_THRESHOLD_RSS
    return rank_by_relevance(semantic_q, items, top_k=limit, threshold=t)


@traceable(run_type="retriever", name="fetch_geeknews")
def fetch_geeknews(query: str = "", limit: int = 10, *,
                   intent: str = "", threshold: float | None = None):
    """GeekNews RSS (한국 IT/해커뉴스 큐레이션)."""
    return _fetch_rss(
        "https://news.hada.io/rss/news", "GeekNews",
        query, limit, intent=intent, threshold=threshold,
    )


@traceable(run_type="retriever", name="fetch_techcrunch_ai")
def fetch_techcrunch_ai(query: str = "", limit: int = 10, *,
                        intent: str = "", threshold: float | None = None):
    """TechCrunch AI 카테고리 RSS (영문 AI/스타트업 산업 뉴스)."""
    return _fetch_rss(
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "TechCrunch AI",
        query, limit, intent=intent, threshold=threshold,
    )


@traceable(run_type="retriever", name="fetch_venturebeat_ai")
def fetch_venturebeat_ai(query: str = "", limit: int = 10, *,
                         intent: str = "", threshold: float | None = None):
    """VentureBeat AI 카테고리 RSS (영문 AI 비즈니스·산업 분석)."""
    return _fetch_rss(
        "https://venturebeat.com/category/ai/feed",
        "VentureBeat AI",
        query, limit, intent=intent, threshold=threshold,
    )


@traceable(run_type="retriever", name="fetch_naver_api")
def fetch_naver_api(query="AI", limit=10, *,
                    intent: str = "", threshold: float | None = None):
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return fetch_naver_crawl(query, limit, intent=intent, threshold=threshold)
    search_q, semantic_q = _resolve_queries(query, intent)
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    fetch_pool = min(30, max(limit, 10))
    params = {"query": search_q, "display": fetch_pool, "sort": "date"}
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
    return rank_by_relevance(semantic_q, items, top_k=limit, threshold=threshold)


def _parse_naver_sds(soup: BeautifulSoup, fetch_pool: int) -> list[dict]:
    items: list[dict] = []
    seen_links: set[str] = set()
    for span in soup.select("span.sds-comps-text-type-headline1"):
        a = span.find_parent("a", href=True)
        if not a:
            continue
        href = a.get("href", "")
        if not href or href in seen_links:
            continue
        seen_links.add(href)
        title = span.get_text(strip=True)

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
            "title": title, "link": href, "summary": summary,
            "published": published, "source": "네이버 뉴스 크롤링",
        })
        if len(items) >= fetch_pool:
            break
    return items


def _parse_naver_legacy(soup: BeautifulSoup, fetch_pool: int) -> list[dict]:
    items: list[dict] = []
    nodes = soup.select("ul.list_news li.bx") or soup.select("div.group_news li")
    for li in nodes[:fetch_pool]:
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
    return items


@traceable(run_type="retriever", name="fetch_naver_crawl")
def fetch_naver_crawl(query="AI", limit=10, *,
                      intent: str = "", threshold: float | None = None):
    """네이버 뉴스 검색 결과 페이지 파싱 (ToS 주의 - 개인 학습 용도로만).
    최근 2년치만 가져오도록 날짜 범위(nso)를 URL에 추가.
    intent가 있으면 자연어 의도로 재랭킹, 없으면 query를 자동으로 키워드/의도 분리."""
    search_q, semantic_q = _resolve_queries(query, intent)
    today = date.today()
    two_years_ago = today - timedelta(days=365 * 2)
    ds = two_years_ago.strftime("%Y.%m.%d")
    de = today.strftime("%Y.%m.%d")
    nso = f"so:dd,p:from{two_years_ago.strftime('%Y%m%d')}to{today.strftime('%Y%m%d')}"
    q_encoded = quote(search_q or "", safe="")
    url = (
        "https://search.naver.com/search.naver?where=news"
        f"&query={q_encoded}&sort=1&pd=3&ds={ds}&de={de}&nso={quote(nso, safe=':,')}"
    )
    # 재랭킹이 의미 있도록 limit보다 큰 풀(최소 20건)에서 의미 정렬 후 상위 limit건만
    fetch_pool = max(limit * 3, 20)
    try:
        r = requests.get(url, headers=UA, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        # 새 구조 (2025~): sds-comps 디자인 시스템. 헤드라인은 span.headline1,
        # 부모 <a>가 링크. 기사 컨테이너는 headline span을 1개만 포함하는 최소 ancestor.
        items = _parse_naver_sds(soup, fetch_pool)
        if not items:
            items = _parse_naver_legacy(soup, fetch_pool)

        if not items:
            return [{"title": "(크롤링 결과 없음 - 네이버 페이지 구조 변경 가능성)",
                     "link": "#", "summary": "", "published": "",
                     "source": "네이버 뉴스 크롤링", "ai_summary": ""}]
        return rank_by_relevance(semantic_q, items, top_k=limit, threshold=threshold)
    except Exception:
        log.exception("fetch_naver_crawl 실패 — query=%r", query)
        return [{"title": "(크롤링 실패 — 서버 로그 확인)", "link": "#",
                 "summary": "", "published": "",
                 "source": "네이버 뉴스 크롤링", "ai_summary": ""}]


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
    except Exception:
        log.exception("analyze 실패 — title=%r", title)
        return {"summary": "(분석 실패 — 서버 로그 확인)",
                "importance": 0, "evaluation": ""}


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
    except Exception:
        log.exception("make_brief 실패")
        return "(브리핑 실패 — 서버 로그 확인)"


# ---------------------------------------------------------------------------
# Self-RAG: critique + source fallback + retry
# ---------------------------------------------------------------------------
_FALLBACK_CHAIN = [
    "geeknews", "naver_api", "naver_crawl",
    "techcrunch_ai", "venturebeat_ai",
]


def expand_sources(sources: list[str]) -> list[str]:
    """폴백 체인: 누락된 소스를 정해진 순서로 덧붙여 retry 시 더 넓게 검색."""
    expanded = list(sources)
    for s in _FALLBACK_CHAIN:
        if s not in expanded:
            expanded.append(s)
    return expanded


@traceable(run_type="chain", name="critique_results",
           metadata={"model": "gpt-4o-mini"})
def critique_results(query: str, top_picks: list[dict],
                     analyzed: list[dict]) -> dict:
    """검색·분석 결과가 사용자 질문에 충분한지 1~5점으로 채점.
    반환: {score, reason, suggest_query}. client가 없거나 호출 실패 시 통과 처리."""
    if client is None:
        return {"score": 5, "reason": "", "suggest_query": ""}
    real = [a for a in (analyzed or [])
            if a.get("link") and a.get("link") != "#"]
    if not real:
        return {"score": 1, "reason": "검색 결과 0건", "suggest_query": ""}
    bullets = "\n".join(
        f"- ({a.get('importance', 0)}/5) {a.get('title', '')}: "
        f"{(a.get('evaluation') or '')[:60]}"
        for a in (top_picks or real[:5])
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "당신은 검색 결과 품질 평가관입니다. 사용자 질문과 "
                    "검색·분석된 기사 목록을 보고 결과가 질문에 답하기에 "
                    "충분한지 1~5점으로 채점하세요.\n"
                    "1=거의 무관, 3=부분 충분, 5=매우 적합. JSON만 응답:\n"
                    '{"score": 1~5 정수, "reason": "한 줄 이유", '
                    '"suggest_query": "더 나은 검색어 또는 빈 문자열"}'
                )},
                {"role": "user",
                 "content": f"[질문]\n{query}\n\n[기사 목록]\n{bullets}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=200,
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        return {
            "score": max(1, min(5, int(data.get("score", 3)))),
            "reason": (data.get("reason") or "").strip(),
            "suggest_query": (data.get("suggest_query") or "").strip(),
        }
    except Exception:
        log.exception("critique_results 실패")
        return {"score": 5, "reason": "", "suggest_query": ""}


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


_SOURCE_TO_NODE = {
    "geeknews":     "fetch_geeknews_node",
    "naver_api":    "fetch_naver_api_node",
    "naver_crawl":  "fetch_naver_crawl_node",
    "techcrunch_ai": "fetch_techcrunch_ai_node",
    "venturebeat_ai": "fetch_venturebeat_ai_node",
}


def route_sources(state: NewsState):
    targets = [_SOURCE_TO_NODE[s] for s in state["sources"]
               if s in _SOURCE_TO_NODE]
    return targets or ["enrich_node"]


def fetch_geeknews_node(state: NewsState):
    return {"items": fetch_geeknews(state.get("query", ""))}


def fetch_naver_api_node(state: NewsState):
    return {"items": fetch_naver_api(state["query"])}


def fetch_naver_crawl_node(state: NewsState):
    return {"items": fetch_naver_crawl(state["query"])}


def fetch_techcrunch_ai_node(state: NewsState):
    return {"items": fetch_techcrunch_ai(state.get("query", ""))}


def fetch_venturebeat_ai_node(state: NewsState):
    return {"items": fetch_venturebeat_ai(state.get("query", ""))}


def _merge_analysis(item: dict, result: dict | None) -> dict:
    """기사 item에 분석 결과(또는 빈 값)를 병합하고 누락 필드를 채운다."""
    merged = dict(item)
    if result and "ai_summary" not in merged:
        merged["ai_summary"] = result["summary"]
        merged["importance"] = result["importance"]
        merged["evaluation"] = result["evaluation"]
    merged.setdefault("ai_summary", "")
    merged.setdefault("importance", 0)
    merged.setdefault("evaluation", "")
    return merged


def enrich_node(state: NewsState):
    items = state["items"]
    if not items:
        return {"analyzed": []}
    if not state.get("do_summarize"):
        return {"analyzed": [_merge_analysis(it, None) for it in items]}
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(analyze, items))
    return {"analyzed": [_merge_analysis(it, r) for it, r in zip(items, results)]}


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
    fetch_nodes = {
        "fetch_geeknews_node": fetch_geeknews_node,
        "fetch_naver_api_node": fetch_naver_api_node,
        "fetch_naver_crawl_node": fetch_naver_crawl_node,
        "fetch_techcrunch_ai_node": fetch_techcrunch_ai_node,
        "fetch_venturebeat_ai_node": fetch_venturebeat_ai_node,
    }
    for name, fn in fetch_nodes.items():
        g.add_node(name, fn)
    g.add_node("enrich_node", enrich_node)
    g.add_node("rank_node", rank_node)
    g.add_node("brief_node", brief_node)
    g.add_conditional_edges(
        START, route_sources,
        list(fetch_nodes) + ["enrich_node"],
    )
    for name in fetch_nodes:
        g.add_edge(name, "enrich_node")
    g.add_edge("enrich_node", "rank_node")
    g.add_edge("rank_node", "brief_node")
    g.add_edge("brief_node", END)
    return g.compile()


news_agent = _build_graph()


def build_news_state(query: str, sources: list[str], do_summarize: bool = True) -> dict:
    """news_agent.invoke()용 초기 NewsState. Flask 라우트와 Discord 봇이 공유.
    sources는 방어적으로 복사 — 호출자가 모듈-레벨 상수를 넘기더라도 변형 안전."""
    return {
        "query": query,
        "sources": list(sources),
        "do_summarize": do_summarize,
        "items": [],
        "analyzed": [],
        "top_picks": [],
        "brief": "",
    }


def sort_for_display(analyzed: list[dict], top_picks: list[dict]) -> list[dict]:
    """top_picks를 먼저, 그 다음 importance 내림차순. placeholder('#') 제거."""
    real = [it for it in analyzed if it.get("link") and it.get("link") != "#"]
    top_links = {t.get("link") for t in top_picks}
    real.sort(key=lambda x: (x.get("link") not in top_links, -x.get("importance", 0)))
    return real


@traceable(run_type="chain", name="reflective_news_agent")
def reflective_news_agent(query: str, sources: list[str],
                          do_summarize: bool = True, *,
                          max_retries: int = 1,
                          critique_threshold: int = 3) -> dict:
    """news_agent를 실행 → critic 점수 < threshold면 query rewrite +
    sources 확장으로 1회 retry. Self-RAG 풀 패턴.

    do_summarize=False면 분석/채점이 의미 없어 critic을 건너뛴다.
    retry 결과가 거의 비어 있으면 원본 결과를 유지(회귀 방지)."""
    result = news_agent.invoke(build_news_state(query, sources, do_summarize))

    if max_retries <= 0 or not do_summarize or client is None:
        return result

    critique = critique_results(
        query, result.get("top_picks", []), result.get("analyzed", []),
    )
    if critique["score"] >= critique_threshold:
        return result

    new_query = critique["suggest_query"] or query
    new_sources = expand_sources(sources)
    if new_query == query and new_sources == list(sources):
        return result   # 변화가 없으면 재시도해도 같은 결과

    log.info(
        "reflective retry — score=%s, sources=%s→%s, query=%r→%r",
        critique["score"], sources, new_sources, query, new_query,
    )
    retry = news_agent.invoke(
        build_news_state(new_query, new_sources, do_summarize)
    )

    retry_real = [a for a in retry.get("analyzed", [])
                  if a.get("link") and a.get("link") != "#"]
    if not retry_real:
        return result   # retry가 빈 결과면 원본 유지

    if retry.get("brief") and critique["reason"]:
        retry["brief"] = (
            f"{retry['brief']}\n\n_검색을 한 번 보강했습니다 — "
            f"이전 결과: {critique['reason']}._"
        )
    return retry


# ---------------------------------------------------------------------------
# ReAct 에이전트 (LLM이 도구 호출을 스스로 결정)
# ---------------------------------------------------------------------------
_TOOL_SOURCE_DISPATCH = {
    "naver": fetch_naver_api,
    "naver_api": fetch_naver_api,
    "naver_crawl": fetch_naver_crawl,
    "geeknews": fetch_geeknews,
    "techcrunch_ai": fetch_techcrunch_ai,
    "venturebeat_ai": fetch_venturebeat_ai,
}


@tool
def search_news(source: str = "geeknews", query: str = "AI", intent: str = "") -> str:
    """뉴스 검색 도구.
    source 옵션:
      - 'geeknews'       : GeekNews RSS (한국 IT/해커뉴스 큐레이션)
      - 'naver' / 'naver_api' : 네이버 검색 API (키 없으면 자동으로 크롤링 폴백)
      - 'naver_crawl'    : 네이버 뉴스 검색 페이지 크롤링 (최근 2년 한정)
      - 'techcrunch_ai'  : TechCrunch AI 카테고리 RSS (영문 AI/스타트업)
      - 'venturebeat_ai' : VentureBeat AI 카테고리 RSS (영문 AI 비즈니스·산업 분석)
    인자:
      - query : 검색 API에 던질 핵심 키워드(공백 구분). 예: "AI 반도체 엔비디아"
      - intent: (선택) 사용자가 알고 싶은 맥락을 자연어 한 문장으로.
                의미 재랭킹에 사용되어, 키워드 매칭만으로는 잡히지 않는
                문맥적 관련성을 확보. 비워두면 query가 사용됨.
    검색된 기사들의 번호/제목/요약/출처를 텍스트로 반환."""
    fetcher = _TOOL_SOURCE_DISPATCH.get(source, fetch_geeknews)
    items = fetcher(query, intent=intent)
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
    "- search_news(source, query, intent): source는 "
    "'geeknews' | 'naver' | 'naver_crawl' | 'techcrunch_ai' | 'venturebeat_ai'.\n"
    "    * **query**: 검색 API에 던질 핵심 키워드만(공백 구분). 예: 'AI 반도체 엔비디아'.\n"
    "      자연어 문장이나 의문어·조사를 절대 query에 넣지 말 것.\n"
    "    * **intent**: 사용자가 알고 싶은 맥락을 **원본 자연어 한 문장**으로 그대로 전달.\n"
    "      예: '엔비디아의 AI 반도체 시장 위치와 최근 동향'. 의미 재랭킹에 쓰여,\n"
    "      키워드만으로는 잡히지 않는 문맥적 관련성을 확보.\n"
    "    * 소스 선택 가이드:\n"
    "      - 'geeknews': 한국 IT/해커뉴스 큐레이션\n"
    "      - 'naver' / 'naver_crawl': 한국 일반 뉴스(국내 동향·기업 소식)\n"
    "      - 'techcrunch_ai' / 'venturebeat_ai': **영문 AI/산업 분석** "
    "(특히 OpenAI·엔비디아·해외 스타트업·시장 분석은 이 두 소스가 강함)\n"
    "    * **키워드 검색(예: 'AI 반도체', 'GPT', '메타버스')**: 한국·영문 양쪽 다 수집. "
    "한국 토픽이면 'geeknews'+'naver', 글로벌 AI 토픽이면 'techcrunch_ai'+'venturebeat_ai'를 "
    "함께 호출. **국제 산업 분석 요청**이면 'techcrunch_ai'+'venturebeat_ai'+'geeknews' 3개 호출.\n"
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
        sources = list(DEFAULT_SOURCES)
    query = request.args.get("q", "AI")
    do_summarize = request.args.get("summarize") == "1" if submitted else True

    result = reflective_news_agent(query, sources, do_summarize)

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
    except Exception:
        # exception 메시지를 사용자에게 노출하지 않음 (API 키·경로 누출 방지).
        log.exception("/agent 실행 실패 — query=%r", user_query)
        final_answer = "(에이전트 실행 실패 — 서버 로그를 확인하세요)"
        trace = []
    return render_template("agent.html", final_answer=final_answer, trace=trace,
                           user_query=user_query, available=True)


if __name__ == "__main__":
    # FLASK_DEBUG=1 일 때만 Werkzeug debugger 활성 — 기본은 안전한 production 모드.
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=5000)
