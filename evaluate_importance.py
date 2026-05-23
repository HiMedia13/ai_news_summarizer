"""importance(1~5) 채점 정확도를 골든셋·LLM-judge·휴리스틱으로 평가.

비교 축 4개:
  - system    vs golden : app.analyze(gpt-4o-mini)가 사람 정답과 얼마나 일치하나
  - judge     vs golden : 강한 모델(gpt-4o)이 정답과 얼마나 일치하나 (셋 난이도 sanity check)
  - heuristic vs golden : 규칙 기반 채점기 정확도 — LLM이 이 규칙을 못 이기면 LLM 가치 의심
  - system    vs judge  : 골든셋 부재/희소 시 추가 reference 비교

메트릭: MAE, exact match, within-1 (|차|<=1).
실행:
  python evaluate_importance.py                                  # 기본 골든셋
  python evaluate_importance.py --golden realistic-golden-candidates.json
출력:
  - 콘솔에 메트릭 요약 + 항목별 표
  - docs/llm-evaluation/importance-report.md 에 마크다운 리포트 저장
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Windows 콘솔 기본 cp949에서 한글·em dash가 깨지지 않도록 stdout을 UTF-8로 강제.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import app

log = logging.getLogger(__name__)

GOLDEN_PATH = Path(__file__).parent / "importance-golden.json"
REPORT_PATH = Path(__file__).parent / "docs" / "llm-evaluation" / "importance-report.md"
JUDGE_MODEL = "gpt-4o"


class JudgeImportance(BaseModel):
    """judge_one 응답 스키마."""

    importance: int = Field(ge=1, le=5, description="1~5 중요도 점수")
    reason: str = Field(default="", description="한 줄 사유")


def _build_judge_messages(rubric_lines: str, title: str, summary: str) -> list[dict]:
    """LLM-judge용 시스템·유저 메시지."""
    system = (
        "당신은 한국어 뉴스 중요도 평가관입니다. 산업·사회 임팩트 기준으로 1~5점 채점.\n"
        + rubric_lines
    )
    user = f"제목: {title}\n\n요약: {summary}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


async def judge_one(judge_llm, rubric_lines: str, title: str, summary: str) -> int:
    try:
        result: JudgeImportance = await judge_llm.with_structured_output(
            JudgeImportance
        ).ainvoke(_build_judge_messages(rubric_lines, title, summary))
        return int(result.importance)
    except Exception:
        log.exception("judge_one 실패 — title=%r", title)
        return 0


def predict_one(item: dict) -> int:
    """app.analyze()를 직접 호출 (동기). 시스템 채점자(gpt-4o-mini, temp=0.3)."""
    r = app.analyze({
        "title": item["title"],
        "summary": item["summary"],
        "link": "ok",  # link='#'이면 분석 스킵되므로 placeholder 회피
    })
    return int(r.get("importance", 0))


# 휴리스틱 채점기 — 규칙 기반 베이스라인.
# LLM이 이 단순 규칙을 못 이기면 LLM 채점의 가치가 의심됨 (sanity check).
# 순서가 중요: 위에서부터 매칭 → return. 1점·5점 같은 명확 신호 먼저.
_HEURISTIC_RULES: list[tuple[int, list[str]]] = [
    (1, ["sns", "viral", "트위터", "twitter", "인테리어", "라이프스타일",
         "사진 공개", "토트백", "데스크 셋업", "키보드 컬렉션", "메뉴 시식"]),
    (5, ["gpt-5", "gpt-6", "claude 4", "claude 5", "gemini 2", "llama 4", "llama 5",
         "ai act", "fda 가이드라인", "규제", "법안", "rubin", "blackwell",
         "공급망 재편", "세대 교체", "양산 시작"]),
    (4, ["인하", "무료 플랜", "전면 확대", "메이저 업데이트", "정책 변화",
         "1m 토큰", "context caching", "온디바이스", "free tier"]),
    (2, ["마이너 패치", "버그 수정", "미세 조정", "ui 개편", "다크 테마",
         "사이드바", "단축키", "minor", "patch"]),
    (3, ["출시", "릴리스", "발표", "release", "launch", "공개",
         "v0.", "v1.", "v2.", "v3.", "안정화", "업데이트"]),
]


def heuristic_one(item: dict) -> int:
    """순수 키워드 매칭. 매치 없으면 3점(실무자 관심사) 기본값."""
    text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    for score, keywords in _HEURISTIC_RULES:
        for kw in keywords:
            if kw.lower() in text:
                return score
    return 3


def metric_block(label: str, preds: list[int], golds: list[int]) -> dict:
    """예측-정답 메트릭. 0은 채점 실패로 간주해 N에서 제외."""
    pairs = [(p, g) for p, g in zip(preds, golds) if p > 0 and g > 0]
    if not pairs:
        return {"label": label, "n": 0, "mae": None,
                "exact_match": None, "within_1": None}
    n = len(pairs)
    diffs = [abs(p - g) for p, g in pairs]
    return {
        "label": label,
        "n": n,
        "mae": sum(diffs) / n,
        "exact_match": sum(1 for d in diffs if d == 0) / n,
        "within_1": sum(1 for d in diffs if d <= 1) / n,
    }


def _format_metric(r: dict) -> str:
    if r["n"] == 0:
        return f"  {r['label']:22s} N=0 (채점 실패)"
    return (f"  {r['label']:22s} n={r['n']:2d}  MAE={r['mae']:.2f}  "
            f"exact={r['exact_match']:.2%}  within-1={r['within_1']:.2%}")


def _build_rubric_lines(rubric: dict) -> str:
    return "\n".join(f"  {k}={v}" for k, v in rubric.items())


def _build_report(items: list[dict], preds: list[int], judges: list[int],
                  heurs: list[int], golds: list[int], rows: list[dict],
                  rubric: dict, golden_path: Path) -> str:
    lines = [
        "# Importance 채점 정확도 평가",
        "",
        f"- 골든셋: `{golden_path.name}` ({len(items)}건)",
        "- 시스템 채점자: `app.analyze` (gpt-4o-mini, temperature=0.3)",
        f"- LLM-judge: `{JUDGE_MODEL}` (temperature=0)",
        "- 휴리스틱: 규칙 기반 키워드 매칭 (sanity baseline)",
        "- 메트릭: MAE(낮을수록 좋음), exact match(정확 일치율), within-1(|차|≤1 일치율)",
        "",
        "## 사용한 rubric (judge에도 동일 적용)",
        "",
        "```",
        _build_rubric_lines(rubric),
        "```",
        "",
        "## 요약 메트릭",
        "",
        "| 비교 | N | MAE | exact match | within-1 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r["n"] == 0:
            lines.append(f"| {r['label']} | 0 | - | - | - |")
        else:
            lines.append(
                f"| {r['label']} | {r['n']} | {r['mae']:.2f} | "
                f"{r['exact_match']:.2%} | {r['within_1']:.2%} |"
            )

    lines += ["", "## 항목별 결과", "",
              "| id | 제목 | 정답 | system(mini) | judge(4o) | heuristic |",
              "|---|---|---|---|---|---|"]
    for it, p, j, h, g in zip(items, preds, judges, heurs, golds):
        diff_sys = "fail" if p == 0 else f"{p-g:+d}"
        diff_jud = "fail" if j == 0 else f"{j-g:+d}"
        diff_heu = f"{h-g:+d}"
        title = it["title"]
        if len(title) > 50:
            title = title[:50] + "…"
        lines.append(
            f"| {it['id']} | {title} | {g} | "
            f"{p if p else 'fail'} ({diff_sys}) | "
            f"{j if j else 'fail'} ({diff_jud}) | "
            f"{h} ({diff_heu}) |"
        )
    lines.append("")
    return "\n".join(lines)


def _resolve_golden_path(arg: str | None) -> Path:
    if arg is None:
        return GOLDEN_PATH
    p = Path(arg)
    if not p.is_absolute():
        p = Path(__file__).parent / p
    return p


async def main() -> None:
    parser = argparse.ArgumentParser(description="importance 채점 정확도 평가")
    parser.add_argument(
        "--golden", default=None,
        help="골든셋 JSON 경로 (기본: importance-golden.json)",
    )
    args = parser.parse_args()
    golden_path = _resolve_golden_path(args.golden)

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY 미설정 — .env 확인")

    if not golden_path.exists():
        sys.exit(f"골든셋 없음 — {golden_path}")

    data = json.loads(golden_path.read_text(encoding="utf-8"))
    # rubric은 기본 골든셋에서, items는 지정 골든셋에서 (라벨링 후보는 rubric 미포함 가능)
    items = data["items"]
    if "rubric" in data:
        rubric = data["rubric"]
    else:
        base = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        rubric = base["rubric"]
        print(f"(rubric 미포함 — {GOLDEN_PATH.name}의 rubric 사용)")

    # 미라벨 항목 가드 — expected_importance가 None이면 평가 불가
    unlabeled = [it["id"] for it in items
                 if it.get("expected_importance") in (None, 0)]
    if unlabeled:
        sys.exit(f"라벨링 미완료 ({len(unlabeled)}건): {unlabeled[:10]}...\n"
                 f"  {golden_path.name}의 expected_importance를 1~5로 채우세요.")

    print(f"골든셋: {golden_path.name}  {len(items)}건 "
          f"(rubric anchors: {list(rubric.keys())})")

    judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    rubric_lines = _build_rubric_lines(rubric)

    print("\n[1] app.analyze() — gpt-4o-mini 채점 중...")
    preds = [predict_one(it) for it in items]

    print(f"\n[2] LLM-judge — {JUDGE_MODEL} 채점 중...")
    judges = list(await asyncio.gather(
        *[judge_one(judge_llm, rubric_lines, it["title"], it["summary"])
          for it in items]
    ))

    print("\n[3] 휴리스틱 채점 (규칙 기반)...")
    heurs = [heuristic_one(it) for it in items]

    golds = [int(it["expected_importance"]) for it in items]

    rows = [
        metric_block("system    vs golden", preds, golds),
        metric_block("judge     vs golden", judges, golds),
        metric_block("heuristic vs golden", heurs, golds),
        metric_block("system    vs judge",  preds, judges),
    ]

    print("\n========== 요약 메트릭 ==========")
    for r in rows:
        print(_format_metric(r))

    print("\n========== 항목별 ==========")
    print(f"  {'id':<5s}{'title':<52s} gold  sys  judge  heur")
    for it, p, j, h, g in zip(items, preds, judges, heurs, golds):
        title = it["title"][:50]
        marker = " " if p == g else ("~" if abs(p - g) <= 1 else "X")
        sys_cell = f"{p}" if p else "fail"
        jud_cell = f"{j}" if j else "fail"
        print(f"  {it['id']:<5s}{title:<52s} {g}    "
              f"{sys_cell:<4s} {jud_cell:<5s} {h}  {marker}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        _build_report(items, preds, judges, heurs, golds, rows, rubric, golden_path),
        encoding="utf-8",
    )
    print(f"\n[저장] {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
