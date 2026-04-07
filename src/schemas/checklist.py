import uuid
from pydantic import BaseModel


# ── Question ──────────────────────────────────────────────────────────────────

class QuestionCreate(BaseModel):
    descricao: str
    ordem: int = 0


class QuestionUpdate(BaseModel):
    descricao: str | None = None
    ordem: int | None = None


class QuestionResponse(BaseModel):
    id: uuid.UUID
    group_id: uuid.UUID
    descricao: str
    ordem: int

    model_config = {"from_attributes": True}


# ── QuestionGroup ─────────────────────────────────────────────────────────────

class QuestionGroupCreate(BaseModel):
    nome: str
    ordem: int = 0


class QuestionGroupUpdate(BaseModel):
    nome: str | None = None
    ordem: int | None = None


class QuestionGroupResponse(BaseModel):
    id: uuid.UUID
    checklist_id: uuid.UUID
    nome: str
    ordem: int
    questions: list[QuestionResponse] = []

    model_config = {"from_attributes": True}


# ── Checklist ─────────────────────────────────────────────────────────────────

class ChecklistCreate(BaseModel):
    nome: str
    descricao: str | None = None


class ChecklistUpdate(BaseModel):
    nome: str | None = None
    descricao: str | None = None


class ChecklistResponse(BaseModel):
    id: uuid.UUID
    project_type_id: uuid.UUID
    nome: str
    descricao: str | None
    groups: list[QuestionGroupResponse] = []

    model_config = {"from_attributes": True}
