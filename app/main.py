from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent import investigation_app
from app.schemas.report import InvestigationReport, RiskLevel

app = FastAPI(
  title="AML AI Agent API",
  description="Neo4j + LangGraph AML investigation service",
  version="1.0.0",
)


class InvestigationRequest(BaseModel):
  query: str = Field(..., min_length=1, description="Natural-language investigation question.")
  depth: int = Field(default=3, ge=1, le=10, description="Reserved for future hop limits.")


class InvestigationResponse(BaseModel):
  status: str
  report: Optional[InvestigationReport] = None
  generated_cypher: Optional[str] = None
  database_results: Optional[List[Dict[str, Any]]] = None
  errors: Optional[List[str]] = None
  message: Optional[str] = None


@app.get("/health")
async def health_check():
  return {"status": "online"}


@app.post("/investigate", response_model=InvestigationResponse)
async def investigate(req: InvestigationRequest) -> InvestigationResponse:
  initial_state = {
    "question": req.query,
    "cypher_query": "",
    "graph_data": [],
    "analysis": "",
    "report": None,
    "errors": [],
    "retry_count": 0,
    "rag_context": "",
  }

  try:
    final_state = await investigation_app.ainvoke(initial_state)
  except Exception as e:
    raise HTTPException(status_code=503, detail=f"Investigation service unavailable: {e}") from e

  report_data = final_state.get("report")
  report = InvestigationReport.model_validate(report_data) if report_data else None

  if final_state.get("errors") and report and report.risk_level == RiskLevel.UNKNOWN:
    return InvestigationResponse(
      status="partial",
      report=report,
      generated_cypher=final_state.get("cypher_query"),
      database_results=final_state.get("graph_data"),
      errors=final_state.get("errors"),
      message="Investigation completed with pipeline errors after retries.",
    )

  return InvestigationResponse(
    status="success",
    report=report,
    generated_cypher=final_state.get("cypher_query"),
    database_results=final_state.get("graph_data"),
    errors=final_state.get("errors") or None,
  )
