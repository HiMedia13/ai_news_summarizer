"""실데이터에서 importance 평가용 후보 셋을 휴리스틱 stratification으로 구축.

기존 RSS fetcher(geeknews/techcrunch_ai/venturebeat_ai)로 실기사를 끌어와
키워드 휴리스틱으로 6개 버킷(announcement/update/market/gossip/regulation/other)으로
분류한 뒤 버킷별 균등 추출. 결과 JSON의 expected_importance 필드는 비어 있어
사용자가 직접 1~5 라벨링한 후 evaluate_importance.py로 평가.

빈 query로 호출하므로 임베딩 비용 0 (rank_by_relevance가 q=""에서 short-circuit).

실행: python build_realistic_golden.py
출력: realistic-golden-candidates.json
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import app

OUTPUT_PATH = Path(__file__).parent / "realistic-golden-candidates.json"

# 휴리스틱 버킷 — 키워드는 lowercase 매칭. 첫 매치된 버킷으로 분류.
BUCKETS: dict[str, list[str]] = {
    "announcement": [
        "출시", "발표", "공개", "릴리스", "공식",
        "launch", "release", "announc", "unveil", "reveal", "introduc",
    ],
    "update": [
        "업데이트", "패치", "버그", "수정", "마이너", "최적화",
        "update", "patch", "fix", "v0.", "v1.", "v2.", "v3.", "minor",
    ],
    "market": [
        "인하", "가격", "투자", "인수", "매출", "유치", "상장",
        "raise", "fund", "acquir", "price", "valuation", "merger", "ipo",
    ],
    "gossip": [
        "sns", "화제", "트위터", "디자인", "인테리어", "라이프스타일",
        "twitter", "viral", "meme", "photo", "interview", "personal",
    ],
    "regulation": [
        "규제", "법안", "fda", "정부", "재판", "소송",
        "regulation", "policy", "compliance", "court", "ruling", "lawsuit",
    ],
}

PER_SOURCE_LIMIT = 30
TARGET_PER_BUCKET = 5  # 6 buckets × 5 = 30 (희소 버킷은 가용분만, "other"에서 백필)
TOTAL_TARGET = 30


def categorize(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    for bucket, keywords in BUCKETS.items():
        for kw in keywords:
            if kw in text:
                return bucket
    return "other"


def _fetch_safe(name: str, fn) -> list[dict]:
    print(f"수집: {name:18s} (query='', limit={PER_SOURCE_LIMIT})", end=" → ")
    try:
        items = fn(query="", limit=PER_SOURCE_LIMIT)
        print(f"{len(items)}건")
        return items
    except Exception as e:
        print(f"실패: {e}")
        return []


def main() -> None:
    pool: list[dict] = []
    pool += _fetch_safe("geeknews", app.fetch_geeknews)
    pool += _fetch_safe("techcrunch_ai", app.fetch_techcrunch_ai)
    pool += _fetch_safe("venturebeat_ai", app.fetch_venturebeat_ai)

    # link 기반 dedup, placeholder('#') 제외
    seen, unique = set(), []
    for it in pool:
        link = it.get("link", "")
        if not link or link == "#" or link in seen:
            continue
        seen.add(link)
        unique.append(it)
    print(f"\n중복 제거 후 풀: {len(unique)}건")

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for it in unique:
        b = categorize(it.get("title", ""), it.get("summary", "") or "")
        by_bucket[b].append(it)

    print("\n버킷 분포:")
    all_buckets = list(BUCKETS) + ["other"]
    for b in all_buckets:
        print(f"  {b:14s} {len(by_bucket.get(b, [])):3d}건")

    # 버킷별 균등 추출 (부족하면 가용분만)
    candidates: list[dict] = []
    short = []
    used_links: set[str] = set()
    for bucket in all_buckets:
        items = by_bucket.get(bucket, [])
        take = min(TARGET_PER_BUCKET, len(items))
        if take < TARGET_PER_BUCKET and bucket != "other":
            short.append((bucket, len(items)))
        for it in items[:take]:
            used_links.add(it.get("link", ""))
            candidates.append({
                "id": f"r{len(candidates)+1:02d}",
                "source": it.get("source", ""),
                "bucket": bucket,
                "title": it.get("title", ""),
                "summary": (it.get("summary") or "")[:500],
                "link": it.get("link", ""),
                "expected_importance": None,
                "rationale": "",
            })

    # 백필: TOTAL_TARGET 미달이면 "other" 버킷에서 더 가져옴
    if len(candidates) < TOTAL_TARGET:
        extra_needed = TOTAL_TARGET - len(candidates)
        other_pool = [it for it in by_bucket.get("other", [])
                      if it.get("link") not in used_links]
        for it in other_pool[:extra_needed]:
            candidates.append({
                "id": f"r{len(candidates)+1:02d}",
                "source": it.get("source", ""),
                "bucket": "other (backfill)",
                "title": it.get("title", ""),
                "summary": (it.get("summary") or "")[:500],
                "link": it.get("link", ""),
                "expected_importance": None,
                "rationale": "",
            })

    out = {
        "version": 1,
        "purpose": ("휴리스틱 stratification으로 실기사에서 추출한 후보. "
                    "expected_importance를 사람이 1~5로 채워 라벨링한 뒤 "
                    "evaluate_importance.py에서 사용."),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "rubric_reference": "importance-golden.json (의 rubric을 그대로 적용)",
        "buckets": all_buckets,
        "target_per_bucket": TARGET_PER_BUCKET,
        "items": candidates,
    }

    OUTPUT_PATH.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[저장] {OUTPUT_PATH}  ({len(candidates)}건)")
    if short:
        print("주의: 후보가 부족한 버킷 →", short)
    print("\n다음 단계:")
    print("  1) 파일을 열어 각 항목의 expected_importance를 1~5로 채우세요.")
    print("     rubric은 importance-golden.json의 anchor 정의를 그대로 사용.")
    print("  2) python evaluate_importance.py --golden realistic-golden-candidates.json")
    print("     (CLI 옵션은 다음 작업에서 추가 예정)")


if __name__ == "__main__":
    main()
