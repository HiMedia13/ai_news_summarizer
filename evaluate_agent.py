"""ReAct 에이전트를 LLM-as-a-judge로 평가

실행: python evaluate_agent.py
LangSmith UI에서 결과 확인: smith.langchain.com → Datasets & Experiments
"""
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import evaluate

from app import react_agent

if react_agent is None:
    raise SystemExit("OPENAI_API_KEY 미설정 - .env에 키를 추가하세요.")

DATASET_NAME = "ai-news-agent-eval"

EXAMPLES = [
    {
        "inputs": {"q": "오늘 GeekNews에서 가장 중요한 뉴스 3건만 골라 종합 브리핑 해줘"},
        "outputs": {
            "expected_tools": ["search_news", "rate_article", "compose_brief"],
            "expected_count": 3,
            "expected_format": "종합 브리핑 + TOP 3 (제목+중요도+링크)",
        },
    },
    {
        "inputs": {"q": "GeekNews 헤드라인만 빨리 보여줘. 평가도 브리핑도 필요없어."},
        "outputs": {
            "expected_tools": ["search_news"],
            "expected_count": None,
            "expected_format": "헤드라인 리스트만 (브리핑/평가 X)",
        },
    },
    {
        "inputs": {"q": "GeekNews에서 가장 중요한 1건만 골라서 그 한 건의 중요도와 평가 알려줘"},
        "outputs": {
            "expected_tools": ["search_news", "rate_article"],
            "expected_count": 1,
            "expected_format": "1개 기사 + 중요도/평가 (브리핑 불필요)",
        },
    },
]


def ensure_dataset(client: Client) -> str:
    """dataset이 없으면 생성, 있으면 examples만 동기화."""
    try:
        ds = client.read_dataset(dataset_name=DATASET_NAME)
        existing = list(client.list_examples(dataset_id=ds.id))
        existing_qs = {e.inputs.get("q") for e in existing}
        new_examples = [e for e in EXAMPLES if e["inputs"]["q"] not in existing_qs]
        if new_examples:
            client.create_examples(
                inputs=[e["inputs"] for e in new_examples],
                outputs=[e["outputs"] for e in new_examples],
                dataset_id=ds.id,
            )
            print(f"[dataset] {len(new_examples)}개 신규 example 추가")
        else:
            print(f"[dataset] '{DATASET_NAME}' 재사용 ({len(existing)}개)")
        return DATASET_NAME
    except Exception:
        ds = client.create_dataset(dataset_name=DATASET_NAME,
                                   description="ReAct 뉴스 에이전트 평가용")
        client.create_examples(
            inputs=[e["inputs"] for e in EXAMPLES],
            outputs=[e["outputs"] for e in EXAMPLES],
            dataset_id=ds.id,
        )
        print(f"[dataset] '{DATASET_NAME}' 새로 생성 ({len(EXAMPLES)}개)")
        return DATASET_NAME


def run_agent(inputs: dict) -> dict:
    """평가 대상 함수: agent 실행 → 답변 + 사용 도구 추출."""
    result = react_agent.invoke(
        {"messages": [("user", inputs["q"])]},
        config={"recursion_limit": 25},
    )
    tools_used = []
    for m in result["messages"]:
        if getattr(m, "type", "") == "ai":
            for tc in getattr(m, "tool_calls", None) or []:
                tools_used.append(tc.get("name", ""))
    final = result["messages"][-1]
    return {"answer": final.content, "tools_used": tools_used}


JUDGE_MODEL = "gpt-4o"
judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)


def quality_judge(outputs: dict, reference_outputs: dict, inputs: dict) -> dict:
    """답변 품질을 0~5점으로 평가."""
    prompt = (
        "당신은 뉴스 큐레이션 에이전트의 답변을 평가하는 심판입니다.\n\n"
        f"질문: {inputs['q']}\n"
        f"기대 형식: {reference_outputs.get('expected_format', '')}\n"
        f"기대 기사 수: {reference_outputs.get('expected_count', 'N/A')}\n\n"
        f"실제 답변:\n{outputs['answer']}\n\n"
        "평가 기준:\n"
        "- 한국어로 자연스럽게 응답했는가?\n"
        "- 사용자가 요청한 형식/개수를 정확히 충족했는가?\n"
        "- 추측이나 환각 없이 검색 결과에 기반했는가?\n\n"
        '응답은 JSON만: {"score": 0~5 정수, "reason": "한 줄 평가"}'
    )
    resp = judge_llm.invoke(prompt)
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    try:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        data = json.loads(m.group(0))
        score = max(0, min(5, int(data.get("score", 0))))
        return {"key": "quality", "score": score / 5.0,
                "comment": data.get("reason", "")}
    except Exception as e:
        return {"key": "quality", "score": 0.0,
                "comment": f"(judge 파싱 실패: {e}) 원문: {text[:200]}"}


def tool_coverage(outputs: dict, reference_outputs: dict) -> dict:
    """기대 도구를 모두 호출했는지 (recall)."""
    expected = set(reference_outputs.get("expected_tools") or [])
    used = set(outputs.get("tools_used") or [])
    if not expected:
        return {"key": "tool_coverage", "score": 1.0, "comment": "기대 도구 없음"}
    recall = len(expected & used) / len(expected)
    missing = expected - used
    extra = used - expected
    parts = [f"recall={recall:.2f}"]
    if missing:
        parts.append(f"missing={sorted(missing)}")
    if extra:
        parts.append(f"extra={sorted(extra)}")
    return {"key": "tool_coverage", "score": recall, "comment": " | ".join(parts)}


def count_match(outputs: dict, reference_outputs: dict) -> dict:
    """rate_article 호출 횟수가 expected_count와 일치하는지."""
    expected_count = reference_outputs.get("expected_count")
    if expected_count is None:
        return {"key": "count_match", "score": 1.0, "comment": "N/A"}
    rate_calls = sum(1 for t in outputs.get("tools_used", []) if t == "rate_article")
    matched = 1.0 if rate_calls == expected_count else 0.0
    return {"key": "count_match", "score": matched,
            "comment": f"rate_article {rate_calls}회 (기대 {expected_count})"}


def main():
    client = Client()
    dataset_name = ensure_dataset(client)

    print(f"\n[evaluate] '{dataset_name}' 에 ReAct 에이전트 실행 + judge({JUDGE_MODEL}) 채점 중...")
    results = evaluate(
        run_agent,
        data=dataset_name,
        evaluators=[quality_judge, tool_coverage, count_match],
        experiment_prefix=f"react-agent-judge-{JUDGE_MODEL}",
        max_concurrency=2,
    )

    print("\n========== 평가 요약 ==========")
    rows = list(results)
    for r in rows:
        ex = r["example"]
        run = r["run"]
        evals = {e.key: e for e in r["evaluation_results"]["results"]}
        print(f"\nQ: {ex.inputs['q'][:60]}")
        used = (run.outputs or {}).get("tools_used", [])
        print(f"  tools_used: {used}")
        for key in ("quality", "tool_coverage", "count_match"):
            ev = evals.get(key)
            if ev is not None:
                print(f"  {key:14s}: score={ev.score:.2f}  {ev.comment or ''}")

    print("\n결과는 LangSmith UI(smith.langchain.com → Datasets & Experiments)에서 "
          "그래프와 함께 확인할 수 있습니다.")


if __name__ == "__main__":
    main()
