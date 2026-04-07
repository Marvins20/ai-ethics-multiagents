import uuid
from datetime import datetime
from pydantic import BaseModel


class PairingJobResponse(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    status: str
    result: list | None = None
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class EvaluatorAssignmentOut(BaseModel):
    project_id: uuid.UUID
    evaluator_id: uuid.UUID
    similarity_score: float
    load_at_assignment: int
    assigned_at: datetime
    evaluator_nome: str | None = None
    evaluator_email: str | None = None

    model_config = {"from_attributes": True}


class AutoavaliacaoCompleteResponse(BaseModel):
    complete: bool


class MyEvaluationOut(BaseModel):
    project_id: uuid.UUID
    titulo: str
    status: str
    departamento: str | None = None
    similarity_score: float
    assigned_at: datetime
