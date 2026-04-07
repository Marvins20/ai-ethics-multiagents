import uuid
from datetime import datetime, date
from pydantic import BaseModel, EmailStr

from ..db.enums import TituloEnum, DepartamentoEnum


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    nome: str | None = None
    matricula: str | None = None
    data_nascimento: date | None = None
    telefone: str | None = None
    departamento: DepartamentoEnum | None = None
    curso_programa: str | None = None
    curriculo_lattes: str | None = None
    titulo: TituloEnum | None = None
    descricao: str | None = None
    areas_atuacao: list[str] | None = None
    projetos_anteriores: str | None = None


class UserUpdate(BaseModel):
    email: EmailStr | None = None
    password: str | None = None
    nome: str | None = None
    matricula: str | None = None
    data_nascimento: date | None = None
    telefone: str | None = None
    departamento: DepartamentoEnum | None = None
    curso_programa: str | None = None
    curriculo_lattes: str | None = None
    titulo: TituloEnum | None = None
    descricao: str | None = None
    areas_atuacao: list[str] | None = None
    projetos_anteriores: str | None = None


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    nome: str | None = None
    matricula: str | None = None
    titulo: str | None = None
    departamento: str | None = None
    curso_programa: str | None = None
    curriculo_lattes: str | None = None
    telefone: str | None = None
    data_nascimento: date | None = None
    privilegio: str | None = None
    descricao: str | None = None
    areas_atuacao: list[str] | None = None
    projetos_anteriores: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
