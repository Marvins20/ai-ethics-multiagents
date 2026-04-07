import uuid
from datetime import datetime
from pydantic import BaseModel


class AuditSubmit(BaseModel):
    project_description: str
    pre_analysis: dict | None = None  # pre-approved analysis_result from decompose phase


class AuditJobResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    status: str
    input_text: str
    result: dict | None
    error_message: str | None
    langgraph_thread_id: str | None
    created_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}


class AuditJobSummary(BaseModel):
    id: uuid.UUID
    status: str
    created_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}
