import uuid
from datetime import datetime
from pydantic import BaseModel


class ChecklistResponseCreate(BaseModel):
    checklist_id: uuid.UUID
    question_id: uuid.UUID
    resposta: str | None = None


class ChecklistResponseUpdate(BaseModel):
    resposta: str | None = None


class ChecklistResponseOut(BaseModel):
    id: uuid.UUID
    checklist_id: uuid.UUID
    project_id: uuid.UUID
    question_id: uuid.UUID
    user_id: uuid.UUID
    resposta: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
