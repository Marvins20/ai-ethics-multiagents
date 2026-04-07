import uuid
from datetime import datetime
from pydantic import BaseModel


class NormCreate(BaseModel):
    titulo: str
    arquivo: str | None = None
    publico: bool = True
    incluir_avaliacao: bool = False
    obrigatorio: bool = False


class NormUpdate(BaseModel):
    titulo: str | None = None
    arquivo: str | None = None
    publico: bool | None = None
    incluir_avaliacao: bool | None = None
    obrigatorio: bool | None = None


class NormResponse(BaseModel):
    id: uuid.UUID
    titulo: str
    arquivo: str | None
    publico: bool
    incluir_avaliacao: bool
    obrigatorio: bool
    created_at: datetime

    model_config = {"from_attributes": True}
