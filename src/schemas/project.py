import uuid
from datetime import datetime
from pydantic import BaseModel, model_validator

from ..db.enums import (
    ProjectStatusEnum,
    DepartamentoEnum,
    AreaCnpqEnum,
    SubAreaCnpqEnum,
    SUBAREA_TO_AREA,
)


class ProjectCreate(BaseModel):
    titulo: str
    project_type_id: uuid.UUID | None = None
    orientador_id: uuid.UUID | None = None
    departamento: DepartamentoEnum | None = None
    area_primaria: AreaCnpqEnum | None = None
    subarea_primaria: SubAreaCnpqEnum | None = None
    area_secundaria: AreaCnpqEnum | None = None
    subarea_secundaria: SubAreaCnpqEnum | None = None
    desenho: str | None = None
    palavras_chave: list[str] | None = None
    envolve_humanos: bool = False
    tema: str | None = None
    problema: str | None = None
    pergunta_de_pesquisa: str | None = None
    hipotese: str | None = None
    justificativa_da_hipotese: str | None = None
    objetivo: str | None = None
    objetivo_especifico: str | None = None
    metodo: str | None = None
    classificacao_da_pesquisa: str | None = None
    riscos: str | None = None
    beneficios: str | None = None

    @model_validator(mode="after")
    def check_subarea_belongs_to_area(self) -> "ProjectCreate":
        _validate_subarea_pair(self.area_primaria, self.subarea_primaria, "primaria")
        _validate_subarea_pair(self.area_secundaria, self.subarea_secundaria, "secundaria")
        return self


class ProjectUpdate(BaseModel):
    titulo: str | None = None
    project_type_id: uuid.UUID | None = None
    orientador_id: uuid.UUID | None = None
    departamento: DepartamentoEnum | None = None
    area_primaria: AreaCnpqEnum | None = None
    subarea_primaria: SubAreaCnpqEnum | None = None
    area_secundaria: AreaCnpqEnum | None = None
    subarea_secundaria: SubAreaCnpqEnum | None = None
    desenho: str | None = None
    palavras_chave: list[str] | None = None
    envolve_humanos: bool | None = None
    tema: str | None = None
    problema: str | None = None
    pergunta_de_pesquisa: str | None = None
    hipotese: str | None = None
    justificativa_da_hipotese: str | None = None
    objetivo: str | None = None
    objetivo_especifico: str | None = None
    metodo: str | None = None
    classificacao_da_pesquisa: str | None = None
    riscos: str | None = None
    beneficios: str | None = None

    @model_validator(mode="after")
    def check_subarea_belongs_to_area(self) -> "ProjectUpdate":
        _validate_subarea_pair(self.area_primaria, self.subarea_primaria, "primaria")
        _validate_subarea_pair(self.area_secundaria, self.subarea_secundaria, "secundaria")
        return self


class ProjectStatusUpdate(BaseModel):
    status: ProjectStatusEnum


class ProjectMemberAdd(BaseModel):
    user_id: uuid.UUID


class ProjectResponse(BaseModel):
    id: uuid.UUID
    titulo: str
    status: ProjectStatusEnum
    criador_id: uuid.UUID | None
    orientador_id: uuid.UUID | None
    project_type_id: uuid.UUID | None
    audit_id: uuid.UUID | None
    departamento: DepartamentoEnum | None
    area_primaria: AreaCnpqEnum | None
    subarea_primaria: SubAreaCnpqEnum | None
    area_secundaria: AreaCnpqEnum | None
    subarea_secundaria: SubAreaCnpqEnum | None
    desenho: str | None
    palavras_chave: list[str] | None
    envolve_humanos: bool
    tema: str | None
    problema: str | None
    pergunta_de_pesquisa: str | None
    hipotese: str | None
    justificativa_da_hipotese: str | None
    objetivo: str | None
    objetivo_especifico: str | None
    metodo: str | None
    classificacao_da_pesquisa: str | None
    riscos: str | None
    beneficios: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_subarea_pair(
    area: AreaCnpqEnum | None,
    subarea: SubAreaCnpqEnum | None,
    label: str,
) -> None:
    if subarea is None:
        return
    expected_area = SUBAREA_TO_AREA.get(subarea)
    if area is not None and expected_area != area:
        raise ValueError(
            f"subarea_{label} '{subarea}' does not belong to area_{label} '{area}'"
        )
