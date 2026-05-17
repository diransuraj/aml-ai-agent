import asyncio
import json
import re
import logging
from typing import Any, Dict, List, Optional, Set, TypedDict

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_neo4j import Neo4jGraph
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import settings
from app.core.vector_store import retriever
from app.schemas.report import InvestigationReport, RiskLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
  question: str
  cypher_query: str
  graph_data: List[dict]
  analysis: str
  report: Optional[Dict[str, Any]]
  errors: List[str]
  retry_count: int
  rag_context: str


def _make_generator_llm() -> ChatOllama:
  return ChatOllama(
    model=settings.OLLAMA_GENERATOR_MODEL,
    temperature=0,
    num_ctx=4096,
    base_url=settings.OLLAMA_BASE_URL,
  )


def _make_analyst_llm() -> ChatOllama:
  return ChatOllama(
    model=settings.OLLAMA_ANALYST_MODEL,
    temperature=0.1,
    num_ctx=4096,
    base_url=settings.OLLAMA_BASE_URL,
  )


generator_llm = _make_generator_llm()
analyst_llm = _make_analyst_llm()
structured_analyst = analyst_llm.with_structured_output(InvestigationReport)


def reset_ollama_clients_for_new_async_loop() -> None:
  """Recreate LangChain Ollama clients (required between isolated asyncio.run calls)."""
  global generator_llm, analyst_llm, structured_analyst
  generator_llm = _make_generator_llm()
  analyst_llm = _make_analyst_llm()
  structured_analyst = analyst_llm.with_structured_output(InvestigationReport)


neo4j_graph = Neo4jGraph(
  url=settings.NEO4J_URI,
  username=settings.NEO4J_USER,
  password=settings.NEO4J_PASSWORD,
)

SCHEMA = neo4j_graph.schema
MAX_RETRIES = 3


def _report_to_state(report: InvestigationReport) -> Dict[str, Any]:
  payload = report.model_dump()
  return {"report": payload, "analysis": report.summary}


def _halted_report(errors: List[str]) -> Dict[str, Any]:
  report = InvestigationReport(
    summary=f"Investigation halted. Errors: {'; '.join(errors)}",
    risk_level=RiskLevel.UNKNOWN,
    evidence_nodes=[],
  )
  return _report_to_state(report)


def extract_defined_variables(query: str) -> Set[str]:
  node_vars = re.findall(r"\(\s*(\w+)(?:\s*:\s*\w+)?\s*\)", query)
  rel_vars = re.findall(r"\[\s*(\w+)(?:\s*:\s*\w+)?\s*\]", query)
  aliases = re.findall(r"AS\s+(\w+)", query, re.IGNORECASE)
  return {v for v in (node_vars + rel_vars + aliases) if v}


def get_return_clause_variables(query: str) -> Set[str]:
  query_upper = query.upper()
  return_pos = query_upper.find("RETURN")
  if return_pos == -1:
    return set()

  after_return = query[return_pos + 6 :]
  end_delimiters = ["WITH", "ORDER BY", "SKIP", "LIMIT", "MATCH", "CREATE", "MERGE"]
  end_pos = len(after_return)
  for kw in end_delimiters:
    pos = after_return.upper().find(kw)
    if pos != -1 and pos < end_pos:
      end_pos = pos

  return_clause = after_return[:end_pos]
  return_refs = re.findall(r"\b(\w+)(?:\.\w+)?\b", return_clause)
  keywords = {"AS", "SUM", "COUNT", "AVG", "DISTINCT", "COLLECT"}
  return {v for v in return_refs if v.upper() not in keywords and not v.isdigit()}


async def generate_query(state: AgentState):
  retries = state.get("retry_count", 0)
  error_msg = state["errors"][-1] if state.get("errors") else None

  rag_context = state.get("rag_context", "")
  if not rag_context:
    try:
      docs = await asyncio.to_thread(retriever.invoke, state["question"])
      rag_context = "\n".join([d.page_content for d in docs])
    except Exception as e:
      logger.error("RAG retrieval failed: %s", e)
      rag_context = "No specific policy context found."

  system_msg = (
    f"You are a Neo4j Cypher expert. Schema:\n{SCHEMA}\n\n"
    "CRITICAL RULES:\n"
    "1. ONLY output raw Cypher—no markdown, no backticks, no text.\n"
    "2. NEVER use SQL keywords like 'FROM' or 'GROUP BY'.\n"
    "3. SCOPE RULE: All variables in RETURN/ORDER BY must be defined in MATCH or created as an ALIAS.\n"
    "4. RANKING RULE: To find the 'most' or 'top', you MUST use this exact pattern:\n"
    "   MATCH (n:Label)-[r:REL]->() \n"
    "   RETURN n.property, count(r) AS frequency \n"
    "   ORDER BY frequency DESC \n"
    "   LIMIT 1\n"
    "5. Do NOT use the variable 'frequency' in ORDER BY unless you have 'AS frequency' in the RETURN clause."
  )

  user_content = (
    f"TEMPLATES:\n{rag_context}\n\n"
    f"QUESTION: {state['question']}\n"
  )

  ranking_keywords = ["most", "highest", "top", "rank", "count", "many"]
  if any(word in state["question"].lower() for word in ranking_keywords):
    user_content += (
      "\nINSTRUCTION: This is a ranking query. "
      "You MUST use 'count(r) AS frequency' in the RETURN clause "
      "and 'ORDER BY frequency DESC' to sort the results.\n"
    )

  if error_msg:
    user_content += (
      f"\nPREVIOUS ERROR (fix this in the new query): {error_msg}\n"
      "Output ONLY the corrected Cypher."
    )

  try:
    response = await asyncio.wait_for(
      generator_llm.ainvoke(
        [SystemMessage(content=system_msg), HumanMessage(content=user_content)]
      ),
      timeout=30.0,
    )
    clean_query = re.sub(r"```(?:cypher)?|```", "", response.content).strip()
  except asyncio.TimeoutError:
    return {"errors": ["LLM generation timed out"], "retry_count": retries + 1}

  defined = extract_defined_variables(clean_query)
  referenced = get_return_clause_variables(clean_query)
  undefined = referenced - defined

  if undefined and clean_query:
    logger.warning("Undefined variables detected: %s", undefined)
    return {
      "cypher_query": clean_query,
      "errors": [f"Variable(s) {undefined} used in RETURN but not defined in MATCH"],
      "retry_count": retries + 1,
      "rag_context": rag_context,
    }

  logger.info("Attempt %s Cypher: %s", retries + 1, clean_query)
  return {
    "cypher_query": clean_query,
    "retry_count": retries + 1,
    "errors": [],
    "rag_context": rag_context,
  }


async def execute_query(state: AgentState):
  if state.get("errors"):
    return state

  try:
    results = await asyncio.wait_for(
      asyncio.to_thread(neo4j_graph.query, state["cypher_query"]),
      timeout=20.0,
    )
    return {"graph_data": results, "errors": []}
  except asyncio.TimeoutError:
    return {"errors": ["Neo4j query timed out"]}
  except Exception as e:
    return {"errors": [f"Neo4j error: {str(e)}"]}


async def handle_error(state: AgentState):
  """Self-correction node: surfaces the failure before rewriting Cypher."""
  errors = state.get("errors") or []
  last_error = errors[-1] if errors else "Unknown execution error"
  attempt = state.get("retry_count", 0)
  logger.warning(
    "Cypher self-correction (attempt %s/%s): %s",
    attempt,
    MAX_RETRIES,
    last_error,
  )
  return {
    "errors": [f"[retry {attempt}/{MAX_RETRIES}] {last_error}"],
  }


def route_after_execution(state: AgentState) -> str:
  if state.get("errors") and state.get("retry_count", 0) < MAX_RETRIES:
    return "error_handler"
  return "analyst"


async def analyze_results(state: AgentState):
  if state.get("errors"):
    return _halted_report(state["errors"])

  data = state.get("graph_data", [])
  if not data:
    report = InvestigationReport(
      summary="Investigation complete: no matching records found.",
      risk_level=RiskLevel.LOW,
      evidence_nodes=[],
    )
    return _report_to_state(report)

  prompt = (
    f"User question: {state['question']}\n"
    f"Neo4j result rows (JSON): {json.dumps(data[:20], default=str)}\n\n"
    "Produce an AML investigation report. "
    "Set evidence_nodes to concrete entity names, IDs, or labels from the data. "
    "Base risk_level only on what the data supports."
  )

  try:
    report = await structured_analyst.ainvoke(
      [
        SystemMessage(
          content=(
            "You are a Senior AML Officer. Respond only with the structured report fields."
          )
        ),
        HumanMessage(content=prompt),
      ]
    )
    if not isinstance(report, InvestigationReport):
      report = InvestigationReport.model_validate(report)
    return _report_to_state(report)
  except Exception as e:
    logger.exception("Structured analysis failed: %s", e)
    report = InvestigationReport(
      summary=f"Investigation completed with unstructured fallback. Data rows: {len(data)}.",
      risk_level=RiskLevel.UNKNOWN,
      evidence_nodes=[],
    )
    return _report_to_state(report)


workflow = StateGraph(AgentState)
workflow.add_node("generator", generate_query)
workflow.add_node("executor", execute_query)
workflow.add_node("error_handler", handle_error)
workflow.add_node("analyst", analyze_results)

workflow.set_entry_point("generator")
workflow.add_edge("generator", "executor")
workflow.add_conditional_edges(
  "executor",
  route_after_execution,
  {"error_handler": "error_handler", "analyst": "analyst"},
)
workflow.add_edge("error_handler", "generator")
workflow.add_edge("analyst", END)

investigation_app = workflow.compile()
