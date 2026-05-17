"""의미 검색 retrieval 품질을 평가하고 임계값을 튜닝.

Ragas의 LLMContextPrecisionWithoutReference는 `response` 필드와 비교하는데
우리는 reference도 response도 없으므로 query↔context 직접 관련성을 묻는
LLM judge로 평가 (Ragas의 ContextRelevance 패턴과 동일한 구조).

실행: python evaluate_retrieval.py
출력:
  - 콘솔에 threshold별 평균 정밀도 표
  - docs/llm-evaluation/retrieval-report.md 에 상세 리포트 저장
"""
import asyncio
import json
import os
import re
import sys

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

import app  # rank_by_relevance, fetch_geeknews

# 토픽 기반 쿼리 — 이전 큐레이션 데이터셋이 못 측정한 영역
QUERIES = [
    "Claude AI 코딩 도구",
    "Rust 프로그래밍 언어",
    "LLM 평가 방법",
    "보안 취약점",
    "AI 반도체",
    "오픈소스 라이선스",
]

# 평가할 소스 — 'geeknews' (RSS+rank), 'naver_api', 'naver_crawl'
SOURCES = ["geeknews", "naver_api", "naver_crawl"]

# 비교할 임계값 후보
THRESHOLDS = [0.15, 0.20, 0.25, 0.30]

TOP_K = 5  # 평가 시 가져올 문서 수


def _build_context(it: dict) -> str:
    title = it.get("title", "")
    summary = (it.get("summary") or "")[:400]
    return f"{title}\n{summary}".strip()


async def judge_chunk(judge_llm, query: str, chunk: str) -> int:
    """LLM judge가 단일 chunk가 query와 관련 있는지 1(yes)/0(no) 응답."""
    prompt = (
        "당신은 검색 결과 관련성 평가관입니다. 주어진 검색어와 기사 텍스트를 보고 "
        "기사가 검색어 주제와 직접 관련 있는지 판단하세요.\n\n"
        f"[검색어]\n{query}\n\n"
        f"[기사]\n{chunk[:600]}\n\n"
        "관련성 기준:\n"
        "- 기사가 검색어의 주제·키워드·도메인과 직접 연관되면 1\n"
        "- 같은 분야이지만 검색어 주제와 거리가 있거나 우연한 단어 중첩이면 0\n\n"
        '응답은 JSON만: {"relevant": 0 또는 1, "reason": "한 줄 사유"}'
    )
    resp = await judge_llm.ainvoke(prompt)
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    try:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        data = json.loads(m.group(0))
        return int(bool(data.get("relevant", 0)))
    except Exception:
        return 0


def _retrieve(source: str, query: str, threshold: float) -> list[dict]:
    """소스별 retrieval 후 의미 재랭킹 적용.

    app.py가 소스별로 다른 상수를 쓰므로(geeknews → RELEVANCE_THRESHOLD_RSS,
    그 외 → RELEVANCE_THRESHOLD), 두 상수 모두 동일 threshold로 patch해야
    평가 시 threshold 변수가 실제로 반영된다."""
    original_default = app.RELEVANCE_THRESHOLD
    original_rss = app.RELEVANCE_THRESHOLD_RSS
    app.RELEVANCE_THRESHOLD = threshold
    app.RELEVANCE_THRESHOLD_RSS = threshold
    try:
        if source == "geeknews":
            return app.fetch_geeknews(query, limit=TOP_K)
        elif source == "naver_api":
            return app.fetch_naver_api(query, limit=TOP_K)
        elif source == "naver_crawl":
            return app.fetch_naver_crawl(query, limit=TOP_K)
        else:
            return []
    finally:
        app.RELEVANCE_THRESHOLD = original_default
        app.RELEVANCE_THRESHOLD_RSS = original_rss


async def evaluate_threshold(judge_llm, source: str, query: str,
                              threshold: float) -> tuple[int, float, list[int]]:
    """소스 + threshold로 retrieval 후 chunk별 relevance 평균(precision) 계산."""
    items = _retrieve(source, query, threshold)
    # placeholder 결과 (link='#') 제외
    items = [it for it in items if it.get("link") and it.get("link") != "#"]
    if not items:
        return 0, 0.0, []
    contexts = [_build_context(it) for it in items]
    scores = await asyncio.gather(
        *[judge_chunk(judge_llm, query, c) for c in contexts]
    )
    precision = sum(scores) / len(scores) if scores else 0.0
    return len(items), precision, list(scores)


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY 미설정")

    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # {source: {threshold: {query: (count, precision, scores)}}}
    results: dict[str, dict[float, dict[str, tuple[int, float, list[int]]]]] = {
        s: {t: {} for t in THRESHOLDS} for s in SOURCES
    }

    for source in SOURCES:
        print(f"\n##### SOURCE: {source} #####")
        for threshold in THRESHOLDS:
            print(f"\n=== threshold = {threshold} ===")
            for q in QUERIES:
                n, p, scores = await evaluate_threshold(judge_llm, source, q, threshold)
                results[source][threshold][q] = (n, p, scores)
                print(f"  {q:25s} → {n}건 · precision={p:.3f}")

    # 소스 × threshold 평균
    print("\n========== 소스 × threshold 평균 ==========")
    summary_rows = []  # (source, threshold, avg_precision, avg_count)
    for source in SOURCES:
        for threshold in THRESHOLDS:
            precisions = [p for _, p, _ in results[source][threshold].values()]
            counts = [n for n, _, _ in results[source][threshold].values()]
            avg_p = sum(precisions) / len(precisions) if precisions else 0.0
            avg_n = sum(counts) / len(counts) if counts else 0.0
            print(f"  {source:12s} · t={threshold:.2f}  "
                  f"avg_precision={avg_p:.3f}  avg_count={avg_n:.1f}")
            summary_rows.append((source, threshold, avg_p, avg_n))

    # 최적 (source, threshold) 조합 — precision 최대
    best = max(summary_rows, key=lambda r: r[2])
    print(f"\n[최적 조합] source={best[0]}, threshold={best[1]}  "
          f"(precision={best[2]:.3f}, 평균 {best[3]:.1f}건)")
    # threshold만 봤을 때 평균
    by_threshold = {}
    for source, t, p, n in summary_rows:
        by_threshold.setdefault(t, []).append(p)
    print("\n threshold별 (3개 소스 평균):")
    for t, ps in sorted(by_threshold.items()):
        print(f"  t={t:.2f}  →  {sum(ps)/len(ps):.3f}")

    # 마크다운 리포트 저장
    lines = ["# Retrieval (의미 검색) 평가 결과 — 소스 × threshold 비교", "",
             "메트릭: LLM-judge (gpt-4o-mini)로 각 retrieved chunk가 query 주제와 직접 관련 있는지 "
             "0/1 채점, 평균을 precision으로 산출 (Ragas ContextRelevance 패턴).",
             f"평가 소스: {SOURCES}",
             f"top_k={TOP_K}, threshold 후보: {THRESHOLDS}",
             f"쿼리 수: {len(QUERIES)}", ""]

    for source in SOURCES:
        lines += [f"## 소스: `{source}` — 쿼리별 결과", ""]
        lines.append("| 쿼리 | " + " | ".join(f"t={t}" for t in THRESHOLDS) + " |")
        lines.append("|---|" + "|".join(["---"] * len(THRESHOLDS)) + "|")
        for q in QUERIES:
            cells = []
            for t in THRESHOLDS:
                n, p, _ = results[source][t][q]
                cells.append(f"{n}건·{p:.2f}")
            lines.append(f"| {q} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## 소스 × threshold 평균", "",
              "| 소스 | threshold | avg_precision | avg_count/쿼리 |",
              "|---|---|---|---|"]
    for source, t, p, n in summary_rows:
        marker = " ⭐" if (source, t) == (best[0], best[1]) else ""
        lines.append(f"| {source} | {t:.2f}{marker} | {p:.3f} | {n:.1f} |")

    lines += ["", "## threshold별 (3개 소스 평균)", "",
              "| threshold | 평균 precision |", "|---|---|"]
    for t, ps in sorted(by_threshold.items()):
        lines.append(f"| {t:.2f} | {sum(ps)/len(ps):.3f} |")

    lines += ["", f"## 추천 조합: `source={best[0]}`, `threshold={best[1]}`",
              f"- avg precision: {best[2]:.3f}",
              f"- avg count/쿼리: {best[3]:.1f}", ""]

    out_path = os.path.join(os.path.dirname(__file__),
                            "docs", "llm-evaluation", "retrieval-report.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[저장] {out_path}")

    # 자동 업데이트는 하지 않고, 사용자에게 결정 위임
    print(f"\n자동 적용을 원하면 app.py의 RELEVANCE_THRESHOLD를 {best[1]}으로 수정하세요.")


if __name__ == "__main__":
    asyncio.run(main())
