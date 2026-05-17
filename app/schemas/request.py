from pydantic import BaseModel, Field

class InvestigationRequest(BaseModel):
    query: str = Field(..., description="The natural language question from the investigator.")
    user_context: str = Field(default="Senior AML Investigator", description="Role of the person asking.")