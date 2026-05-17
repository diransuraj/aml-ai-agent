from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
  LOW = "low"
  MEDIUM = "medium"
  HIGH = "high"
  CRITICAL = "critical"
  UNKNOWN = "unknown"


class InvestigationReport(BaseModel):
  """Structured AML investigation output returned by the API."""

  summary: str = Field(..., description="Executive summary of findings for the investigator.")
  risk_level: RiskLevel = Field(..., description="Overall risk classification.")
  evidence_nodes: List[str] = Field(
    default_factory=list,
    description="Entity or node identifiers cited as evidence (names, IDs, or labels from graph results).",
  )
