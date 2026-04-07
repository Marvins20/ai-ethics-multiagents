import uuid
from pydantic import BaseModel


class ProjectTypeCreate(BaseModel):
    nome: str


class ProjectTypeUpdate(BaseModel):
    nome: str | None = None


class ProjectTypeResponse(BaseModel):
    id: uuid.UUID
    nome: str

    model_config = {"from_attributes": True}
