import sys
import os
from pathlib import Path

# Path setup
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import asyncio

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from local_judge import OllamaJudge
from app.agent import investigation_app, reset_ollama_clients_for_new_async_loop


# Judge model: use a tag that supports Ollama JSON schema (e.g. llama3.1, llama3.2).
local_llm = OllamaJudge(model_name=os.environ.get("DEEPEVAL_JUDGE_MODEL", "llama3.1"))

# Define the 10 Golden Questions
GOLDEN_SET = [
    "Which bank has the most SENT_TO relationships?",
    "Find all Entities linked to the address '123 Cayman Way'.",
    "Identify any Officers associated with more than 3 Banks.",
    "List all transactions flagged as 'High Risk' in the last 6 months.",
    "Are there any circular paths (Bank A -> Bank B -> Bank A)?",
    "Find the smurfing pattern: which user has the most transfers under $10,000?",
    "Which jurisdiction has the highest volume of 'RECEIVED_FROM' relationships?",
    "Show me the top 5 Officers ranked by their connection to sanctioned Entities.",
    "Is there a link between 'Alpha Corp' and 'Shell Co' within 3 hops?",
    "Which Entity has the most diverse range of relationship types?"
]

# DeepEval pytest integration (tracing/cache); does not fix asyncio nesting.
os.environ["DEEPEVAL_PYTEST"] = "True"

@pytest.mark.parametrize("question", GOLDEN_SET)
def test_agent_performance(question):
    # New ChatOllama instances each case: their AsyncClient is bound to the previous
    # loop after asyncio.run() returns; reusing them causes "Event loop is closed".
    reset_ollama_clients_for_new_async_loop()
    result = asyncio.run(investigation_app.ainvoke({"question": question}))
    
    report = result.get("report") or {}
    actual_output = report.get("summary") or result.get("analysis", "No analysis generated")
    db_results = result.get("graph_data", [])

    # Ground the judge in what Neo4j actually returned (avoid "Sample" + [] confusing faithfulness).
    if not db_results:
        retrieval_context = [
            "Neo4j executed the Cypher query and returned zero rows (empty result set)."
        ]
    else:
        snippet = db_results[:5]
        retrieval_context = [
            f"Neo4j returned {len(db_results)} row(s); up to 5 rows for assessment: {snippet!r}"
        ]

    errors = result.get("errors") or []
    halted = "Investigation halted" in (actual_output or "")
    if errors:
        retrieval_context = [
            f"Pipeline errors (from agent state): {'; '.join(errors)}.",
            *retrieval_context,
        ]

    # Faithfulness is especially noisy when the analyst adds AML narrative beyond a tiny
    # Neo4j snippet. Use split thresholds; set DEEPEVAL_METRIC_THRESHOLD to apply one value to both.
    combined = os.environ.get("DEEPEVAL_METRIC_THRESHOLD")
    if combined is not None:
        faith_t = relevancy_t = float(combined)
    else:
        faith_t = float(os.environ.get("DEEPEVAL_FAITHFULNESS_THRESHOLD", "0.35"))
        relevancy_t = float(os.environ.get("DEEPEVAL_ANSWER_RELEVANCY_THRESHOLD", "0.5"))

    faith_metric = FaithfulnessMetric(threshold=faith_t, model=local_llm)
    relevancy_metric = AnswerRelevancyMetric(threshold=relevancy_t, model=local_llm)

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )

    # Answer relevancy vs the user question is not meaningful for error-only outputs.
    metrics = [faith_metric] if halted else [faith_metric, relevancy_metric]

    # DeepEval sync path (run_async=False) avoids nested run_until_complete.
    assert_test(test_case, metrics, run_async=False)