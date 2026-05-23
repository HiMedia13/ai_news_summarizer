"""ReAct 에이전트를 LLM-as-a-judge로 평가

실행: python evaluate_agent.py
LangSmith UI에서 결과 확인: smith.langchain.com → Datasets & Experiments
"""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langsmith import Client
from langsmith.evaluation import evaluate

from app import react_agent

if react_agent is None:
    raise SystemExit("OPENAI_API_KEY 미설정 - .env에 키를 추가하세요.")

log = logging.getLogger(__name__)

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
    """평가 대상 함수: agent 실행 → 답변 + 사용 도구 + 검색 컨텍스트 추출."""
    result = react_agent.invoke(
        {"messages": [("user", inputs["q"])]},
        config={"recursion_limit": 25},
    )
    tools_used = []
    retrieval_chunks = []
    messages = result.get("messages") or []
    for m in messages:
        if getattr(m, "type", "") == "ai":
            for tc in getattr(m, "tool_calls", None) or []:
                tools_used.append(tc.get("name", ""))
        elif getattr(m, "type", "") == "tool":
            # search_news 출력만 retrieval context로 취급
            if getattr(m, "name", "") == "search_news":
                content = m.content if isinstance(m.content, str) else str(m.content)
                retrieval_chunks.append(content)
    if not messages:
        log.warning("run_agent: react_agent가 빈 messages 반환 — query=%r", inputs.get("q"))
        return {"answer": "", "tools_used": [], "retrieval": ""}
    final = messages[-1]
    return {
        "answer": getattr(final, "content", "") or "",
        "tools_used": tools_used,
        "retrieval": "\n\n".join(retrieval_chunks),
    }


class QualityRating(BaseModel):
    """quality_judge 응답 스키마 (0~5 정수)."""
    score: int = Field(ge=0, le=5, description="0~5 정수 점수")
    reason: str = Field(default="", description="한 줄 평가")


class JudgeRating(BaseModel):
    """faithfulness/answer_relevancy/contextual_relevancy 응답 스키마 (0~1 실수)."""
    score: float = Field(ge=0.0, le=1.0, description="0.0~1.0 실수 점수")
    reason: str = Field(default="", description="한 줄 평가")


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
    try:
        messages = [{"role": "user", "content": prompt}]
        result: QualityRating = (
            judge_llm.with_structured_output(QualityRating).invoke(messages)
        )
        return {"key": "quality", "score": result.score / 5.0,
                "comment": result.reason}
    except Exception as e:
        log.exception("quality_judge 실패")
        return {"key": "quality", "score": 0.0,
                "comment": f"(judge 파싱 실패: {e})"}


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


def _judge_json(prompt: str, key: str) -> dict:
    """공통 헬퍼: judge LLM에 prompt 보내고 0~1 점수 dict 반환."""
    try:
        messages = [{"role": "user", "content": prompt}]
        result: JudgeRating = (
            judge_llm.with_structured_output(JudgeRating).invoke(messages)
        )
        return {"key": key, "score": result.score, "comment": result.reason}
    except Exception as e:
        log.exception("_judge_json 실패 — key=%s", key)
        return {"key": key, "score": 0.0,
                "comment": f"(judge 파싱 실패: {e})"}


def faithfulness(outputs: dict, inputs: dict) -> dict:
    """답변이 retrieval context에 충실한가 (환각 없는가) — 0~1 점수."""
    retrieval = outputs.get("retrieval", "")
    answer = outputs.get("answer", "")
    if not retrieval.strip():
        return {"key": "faithfulness", "score": 0.0,
                "comment": "retrieval context 없음 (search_news 미호출)"}
    prompt = (
        "당신은 RAG 환각 검사관입니다. 답변의 모든 주장이 검색 컨텍스트로 뒷받침되는지 평가하세요.\n\n"
        f"[검색 컨텍스트]\n{retrieval[:4000]}\n\n"
        f"[답변]\n{answer}\n\n"
        "평가 기준:\n"
        "- 답변에 등장하는 사실/주장이 컨텍스트로부터 직접 추론 가능한가?\n"
        "- 컨텍스트에 없는 내용을 답변이 추가했는가? (환각)\n"
        "- 1.0=모든 주장 충실, 0.5=절반 환각, 0.0=대부분 환각/근거 없음\n\n"
        '응답은 JSON만: {"score": 0.0~1.0 실수, "reason": "한 줄 평가"}'
    )
    return _judge_json(prompt, "faithfulness")


def answer_relevancy(outputs: dict, inputs: dict) -> dict:
    """답변이 사용자 질문에 직접 부합하는가 — 0~1 점수."""
    question = inputs.get("q", "")
    answer = outputs.get("answer", "")
    prompt = (
        "당신은 답변 관련성 검사관입니다. 답변이 질문에 직접 답하고 있는지 평가하세요.\n\n"
        f"[질문]\n{question}\n\n"
        f"[답변]\n{answer}\n\n"
        "평가 기준:\n"
        "- 답변이 질문의 의도(요청 형식·범위·개수)에 직접 부합하는가?\n"
        "- 동문서답이거나 회피·우회 답변은 아닌가?\n"
        "- 1.0=완전 부합, 0.5=부분 부합, 0.0=무관/회피\n\n"
        '응답은 JSON만: {"score": 0.0~1.0 실수, "reason": "한 줄 평가"}'
    )
    return _judge_json(prompt, "answer_relevancy")


def contextual_relevancy(outputs: dict, inputs: dict) -> dict:
    """검색 결과(retrieval)가 질문과 관련 있는가 — retriever 자체 품질."""
    question = inputs.get("q", "")
    retrieval = outputs.get("retrieval", "")
    if not retrieval.strip():
        return {"key": "contextual_relevancy", "score": 0.0,
                "comment": "retrieval context 없음"}
    prompt = (
        "당신은 검색 품질 평가관입니다. 검색된 기사들이 질문과 관련 있는지 평가하세요.\n\n"
        f"[질문]\n{question}\n\n"
        f"[검색 결과]\n{retrieval[:4000]}\n\n"
        "평가 기준:\n"
        "- 검색된 기사들이 질문의 주제·키워드와 관련 있는가?\n"
        "- 무관한 기사가 섞여 있는 비율이 얼마나 되는가?\n"
        "- 1.0=모두 관련, 0.5=절반만 관련, 0.0=대부분 무관\n\n"
        '응답은 JSON만: {"score": 0.0~1.0 실수, "reason": "한 줄 평가"}'
    )
    return _judge_json(prompt, "contextual_relevancy")


def main():
    try:
        client = Client()
    except Exception as e:
        raise SystemExit(
            "LangSmith Client 초기화 실패. LANGSMITH_API_KEY가 .env에 있는지 "
            f"확인하세요.\n원인: {e}"
        ) from e
    dataset_name = ensure_dataset(client)

    print(f"\n[evaluate] '{dataset_name}' 에 ReAct 에이전트 실행 + judge({JUDGE_MODEL}) 채점 중...")
    results = evaluate(
        run_agent,
        data=dataset_name,
        evaluators=[
            quality_judge, tool_coverage, count_match,
            faithfulness, answer_relevancy, contextual_relevancy,
        ],
        experiment_prefix=f"react-agent-judge-{JUDGE_MODEL}",
        max_concurrency=2,
    )

    print("\n========== 평가 요약 ==========")
    rows = list(results)
    metric_keys = ("quality", "tool_coverage", "count_match",
                   "faithfulness", "answer_relevancy", "contextual_relevancy")
    # 평균 누적용
    sums = {k: 0.0 for k in metric_keys}
    counts = {k: 0 for k in metric_keys}

    # 마크다운 리포트 생성
    report_lines = ["# 에이전트 평가 결과", "", f"Judge 모델: `{JUDGE_MODEL}`",
                    f"평가 대상: ReAct 에이전트 (gpt-4o-mini)",
                    f"데이터셋: `{dataset_name}` ({len(rows)}개 예시)", ""]

    for r in rows:
        ex = r["example"]
        run = r["run"]
        evals = {e.key: e for e in r["evaluation_results"]["results"]}
        q = ex.inputs["q"]
        used = (run.outputs or {}).get("tools_used", [])
        print(f"\nQ: {q[:60]}")
        print(f"  tools_used: {used}")
        report_lines += [f"## Q: {q}", "", f"- tools_used: `{used}`", ""]
        for key in metric_keys:
            ev = evals.get(key)
            if ev is not None:
                print(f"  {key:22s}: score={ev.score:.2f}  {ev.comment or ''}")
                report_lines.append(f"- **{key}**: `{ev.score:.2f}` — {ev.comment or ''}")
                if ev.score is not None:
                    sums[key] += ev.score
                    counts[key] += 1
        report_lines.append("")

    # 평균 요약
    print("\n========== 메트릭 평균 ==========")
    report_lines += ["## 메트릭 평균", ""]
    for key in metric_keys:
        if counts[key]:
            avg = sums[key] / counts[key]
            print(f"  {key:22s}: {avg:.2f}")
            report_lines.append(f"- **{key}**: `{avg:.2f}`")
    report_lines.append("")

    out_path = os.path.join(os.path.dirname(__file__),
                            "docs", "llm-evaluation", "evaluation-report.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[저장] {out_path}")

    print("\n결과는 LangSmith UI(smith.langchain.com → Datasets & Experiments)에서 "
          "그래프와 함께 확인할 수 있습니다.")


if __name__ == "__main__":
    main()
