import uuid
from datetime import datetime
from pydantic import BaseModel


class QuestionnaireItemOut(BaseModel):
    statement: str
    options: list[str]
    model_config = {"from_attributes": True}


class QuestionnaireJobResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    status: str  # pending | running | completed | failed
    result: list[QuestionnaireItemOut] | None = None
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    model_config = {"from_attributes": True}


class QuestionnaireResponseCreate(BaseModel):
    questionnaire_job_id: uuid.UUID
    question_idx: int
    resposta: str


class QuestionnaireResponseOut(BaseModel):
    id: uuid.UUID
    questionnaire_job_id: uuid.UUID
    project_id: uuid.UUID
    question_idx: int
    user_id: uuid.UUID
    resposta: str | None
    created_at: datetime
    updated_at: datetime
    model_config = {"from_attributes": True}
